import transformers
from ml_swissknife import utils


class BackwardHookMode(metaclass=utils.ContainerMeta):
    ghost_norm = "ghost_norm"
    ghost_grad = "ghost_grad"
    default = "default"


class ClippingMode(metaclass=utils.ContainerMeta):
    default = "default"  # Global fixed.
    ghost = "ghost"  # Global fixed clipping with ghost clipping.
    per_layer = "per_layer"  # Per layer fixed clipping.
    per_layer_percentile = "per_layer_percentile"  # Clip gradient per-layer based on gradient norm percentile.


class AccountingMode(metaclass=utils.ContainerMeta):
    rdp = "rdp"
    glw = "glw"
    all_ = "all"


SUPPORTED_TRANSFORMERS = (
    transformers.models.openai.modeling_openai.OpenAIGPTLMHeadModel,
    transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModel,
    transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel,
    transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModel,
    transformers.models.bert.modeling_bert.BertForSequenceClassification,
    transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification,
    transformers.models.albert.modeling_albert.AlbertForSequenceClassification,
    transformers.models.bart.modeling_bart.BartForConditionalGeneration,
    transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
    transformers.models.opt.modeling_opt.OPTForCausalLM,
    transformers.models.vit.modeling_vit.ViTForImageClassification,
    transformers.models.deit.modeling_deit.DeiTForImageClassification,
    transformers.models.beit.modeling_beit.BeitForImageClassification,
)
