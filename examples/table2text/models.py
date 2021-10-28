import torch
from torch import nn
from transformers import GPT2PreTrainedModel, GPT2LMHeadModel


class _View(nn.Module):
    def __init__(self, shape):
        super(_View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)


class PrefixTuner(GPT2PreTrainedModel):
    """A minimalistic implementation of the core components."""

    def __init__(self, config, model_args, gpt2=None):
        super(PrefixTuner, self).__init__(config=config)

        # Instantiate a GPT-2, and DON'T optimizer it!
        if gpt2 is None:
            self.gpt2 = GPT2LMHeadModel.from_pretrained(
                model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir,
            )
        else:
            self.gpt2 = gpt2

        self.register_buffer('extra_prefix_ids', torch.arange(model_args.prefix_len))
        # TODO: Also introduce the easier net.
        self.extra_prefix_net = nn.Sequential(
            nn.Embedding(model_args.prefix_len, config.n_embd),
            nn.Linear(config.n_embd, model_args.mid_dim),
            nn.Tanh(),
            nn.Linear(model_args.mid_dim, config.n_layer * 2 * config.n_embd),
            _View((-1, model_args.prefix_len, config.n_layer * 2, config.n_head, config.n_embd // config.n_head)),
            nn.Dropout(model_args.prefix_dropout),
        )

    def make_past_key_values(self, bsz=None):
        extra_prefix_ids = self.extra_prefix_ids[None, :].expand(bsz, -1)
        past_key_values = self.extra_prefix_net(extra_prefix_ids)
        # (n_layer, batch_size, n_head, prefix_len, n_embed // n_head).
        # e.g., (2, 1, 12, 5, 64,).
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2, dim=0)
        return past_key_values

    def state_dict(self):
        """Avoid storing GPT-2, since it's not even trained."""
        return self.extra_prefix_net.state_dict()

    def load_state_dict(self, state_dict):
        """Avoid loading GPT-2, since it's not even trained."""
        self.extra_prefix_net.load_state_dict(state_dict)

    @property
    def major_device(self):
        """Returns the device where the parameters are on."""
        return next(self.parameters()).device

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        past_key_values = self.make_past_key_values(bsz=input_ids.size(0))
        return self.gpt2(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def generate(self, input_ids, num_beams, **kwargs):
        # Additional files also changed:
        # src/transformers/generation_utils.py
        # src/transformers/models/gpt2/modeling_gpt2.py

        # --- lxuechen: This part is really error-prone!
        #   A sanity check is to optimize the model for a few updates and check if the beam-search generations changed.
        #   The confusing logic in generation_utils:
        #       1) `past` is used in `GPT2LMHeadModel:prepare_inputs_for_generation`,
        #       2) it's converted to `past_key_values` in that function,
        #       3) `past_key_values` is then updated in forward due to return_dict,
        #       4) `past` is set to `past_key_values` in `generation_utils:_update_model_kwargs_for_generation`

        # This is expansion step is important for generation, since otherwise the shapes are wrong.
        past_key_values = self.make_past_key_values(bsz=input_ids.size(0) * num_beams)
        # ---

        return self.gpt2.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            past_key_values=past_key_values,

            use_cache=True,
            position_ids=None,

            # --- lxuechen: These arguments I created to make sure prefix-tuning works correctly.
            #   The logic: At beginning, past=None, and then it gets replaced with past_key_values.
            #              Can't directly give in past, since otherwise, input_ids gets truncated to the last index.
            use_past_key_values_as_past_at_init=True,
            nullify_attention_mask=True,
            # ---

            **kwargs
        )
