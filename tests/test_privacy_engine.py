"""Test gradient accumulation of privacy engine.

Compare the accumulated gradient with multiple virtual steps against processing the examples one-by-one.
You will need a GPU to run this!

python tests/test_privacy_engine.py
pytest -s tests
"""
import contextlib
import copy
import gc
import itertools
import os

import pytest
import torch
import torch.nn.functional as F
import tqdm
import transformers
from ml_swissknife import utils
from torch import optim

from private_transformers import PrivacyEngine, freeze_isolated_params_for_vit, supported_layers_grad_samplers

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# Don't mess up my disk space on cluster.
if os.path.exists("/nlp/scr/lxuechen/cache/private_transformers"):
    CACHE_DIR = "/nlp/scr/lxuechen/cache/private_transformers"
else:
    CACHE_DIR = None


def _make_classification_data(num_micro_batches=2, micro_batch_size=4, seq_len=128, num_labels=2):
    return tuple(
        dict(
            input_ids=torch.randint(low=1, high=100, size=(micro_batch_size, seq_len)),
            labels=torch.randint(low=0, high=num_labels, size=(micro_batch_size,)),
        )
        for _ in range(num_micro_batches)
    )


def _make_generation_data(num_micro_batches=4, micro_batch_size=4, seq_len=128):
    """Make a batch of plain sequences.

    Tuple of multiple micro batches.
    """
    return tuple(
        dict(input_ids=torch.randint(low=1, high=100, size=(micro_batch_size, seq_len)))
        for _ in range(num_micro_batches)
    )


def _make_encoder_decoder_data(num_micro_batches=4, micro_batch_size=4, seq_len=64, target_seq_len=64):
    """Make a batch of sequences."""
    return tuple(
        dict(
            input_ids=torch.randint(low=1, high=100, size=(micro_batch_size, seq_len)),
            decoder_input_ids=torch.randint(low=1, high=100, size=(micro_batch_size, target_seq_len)),
        )
        for _ in range(num_micro_batches)
    )


def _make_image_classification_data(
    num_micro_batches=4, micro_batch_size=4,
    num_labels=10, num_channels=3, height=224, width=224,
):
    return tuple(
        dict(
            pixel_values=torch.randn(micro_batch_size, num_channels, height, width),
            labels=torch.randint(size=(micro_batch_size,), low=0, high=num_labels),
        )
        for _ in range(num_micro_batches)
    )


def _prepare_inputs(batch: dict):
    return {key: value.to(DEVICE) for key, value in batch.items()}


@pytest.mark.parametrize(
    'clipping_mode,model_name_or_path',
    itertools.product(["ghost", "default"], ['roberta-base', 'bert-base-cased', 'albert-base-v2'])
)
def test_classification(clipping_mode: str, model_name_or_path: str):
    if clipping_mode == "ghost" and 'albert' in model_name_or_path:
        pytest.skip("Ghost clipping does not support parameter sharing which occurs in ALBERT.")

    gc.collect()
    torch.cuda.empty_cache()

    lr = 1e-4
    num_labels = 2
    num_micro_batches = 4
    micro_batch_size = 4
    seq_len = 128
    batch_size = num_micro_batches * micro_batch_size
    max_grad_norm = 1

    # Set up model -- disable dropout to remove randomness.
    config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        attention_probs_dropout_prob=0.,
        hidden_dropout_prob=0.,
        classifier_dropout_prob=0.,  # Important for ALBERT, since otherwise randomness causes gradient difference.
        return_dict=True,
        padding_idx=-1,
        # roberta sets `pad_token_id` to 1 by default, whereas it's 0 for bert.
        #   Uncomment the following line, if you want consistency (it's not totally necessary).
        # pad_token_id=-1,
    )

    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model.requires_grad_(True).train()

    param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    num_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'Number of trainable parameters: {num_trainable_params / 1e6:.4f} million')

    # Make data.
    batches = _make_classification_data(
        micro_batch_size=micro_batch_size, num_micro_batches=num_micro_batches, seq_len=seq_len, num_labels=num_labels,
    )

    # 1: Compute updates with my engine.
    clone1 = copy.deepcopy(model).to(DEVICE)
    clone1_params = [param for param in clone1.parameters() if param.requires_grad]

    optimizer = optim.Adam(params=clone1_params, lr=lr)
    privacy_engine = PrivacyEngine(
        module=clone1,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        noise_multiplier=0.,  # Remove noise to test gradient clipping & accumulation.
        sample_size=1000,  # Any number suffices for testing.
        epochs=1,  # Any number suffices for testing.
        numerical_stability_constant=0.,  # Important!
        clipping_mode=clipping_mode,
    )
    privacy_engine.attach(optimizer=optimizer)
    optimizer.zero_grad()  # Clears summed_grad, so don't run unless necessary.
    for i, batch in enumerate(batches, 1):
        batch = _prepare_inputs(batch)
        logits = clone1(**batch).logits
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels, reduction="none")

        del batch
        if i == len(batches):
            optimizer.step(loss=loss)
        else:
            optimizer.virtual_step(loss=loss)

    result1 = [param.grad for param in clone1_params]
    privacy_engine.detach()  # Restore`hooks_mode`.
    del clone1, loss, logits, labels, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # 2: Compute grad and clip one-by-one.
    clone2 = copy.deepcopy(model).to(DEVICE)
    clone2_params = [param for param in clone2.parameters() if param.requires_grad]

    optimizer = torch.optim.Adam(params=clone2_params, lr=lr)
    summed_grad = [torch.zeros_like(param) for param in clone2_params]
    for i, batch in tqdm.tqdm(enumerate(batches, 1), desc="over batches"):
        batch = _prepare_inputs(batch)
        for input_ids, labels in tqdm.tqdm(utils.zip_(batch["input_ids"], batch["labels"]), desc="over samples"):
            optimizer.zero_grad(set_to_none=True)  # Clear previous grad each time!
            input_ids = input_ids[None, :]
            labels = labels[None]

            logits = clone2(input_ids=input_ids, labels=labels).logits
            loss = F.cross_entropy(logits, labels, reduction="none").sum()
            loss.backward()

            with torch.no_grad():
                flat_unclipped_grad = torch.cat(tuple(param.grad.flatten() for param in clone2_params))
                factor = torch.clamp_max(max_grad_norm / flat_unclipped_grad.norm(), 1.)
                for si, pi in utils.zip_(summed_grad, clone2_params):
                    si.add_(factor * pi.grad)

    result2 = [grad / batch_size for grad in summed_grad]
    del clone2, loss, logits, labels, optimizer

    del model
    gc.collect()
    torch.cuda.empty_cache()

    for g1, g2, name in utils.zip_(result1, result2, param_names):
        try:
            torch.testing.assert_allclose(g1, g2, atol=1e-5, rtol=1e-6)
        except AssertionError as e:
            print(f"failed at {name}")
            raise e


@pytest.mark.parametrize(
    'clipping_mode,tie_word_embeddings,model_name_or_path',
    # Testing two OPT models, since the 350m one has a different LayerNorm placement.
    tuple(
        itertools.product(
            ["ghost", "default"],
            [False, True],
            ['gpt2', 'openai-gpt', 'facebook/opt-125m', 'facebook/opt-350m']
        )
    )
)
def test_generation(clipping_mode, tie_word_embeddings, model_name_or_path):
    gc.collect()
    torch.cuda.empty_cache()

    lr = 1e-4
    num_micro_batches = 4
    micro_batch_size = 4
    seq_len = 128
    batch_size = num_micro_batches * micro_batch_size
    max_grad_norm = 1

    # Catch expected failures.
    with pytest.raises(ValueError) if clipping_mode == "ghost" and tie_word_embeddings else contextlib.nullcontext():

        # Set up model.
        config = transformers.AutoConfig.from_pretrained(model_name_or_path, cache_dir=CACHE_DIR)
        config.tie_word_embeddings = tie_word_embeddings
        # Branch out due to weird inconsistency with naming.
        # OPT is AutoModelForCausalLM; GPT is AutoModelWithLMHead.
        if 'opt' in model_name_or_path:
            # Remove potential causes of randomness.
            config.activation_dropout = config.attention_dropout = config.dropout = config.layerdrop = 0.
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name_or_path, config=config, cache_dir=CACHE_DIR
            )
        else:
            # Remove potential causes of randomness.
            config.attn_pdrop = config.embd_pdrop = config.resid_pdrop = 0.
            model = transformers.AutoModelWithLMHead.from_pretrained(
                model_name_or_path, config=config, cache_dir=CACHE_DIR
            )
        model.train()  # Needed to ensure privacy engine works.
        param_names = [name for name, param in model.named_parameters() if param.requires_grad]

        # Make data.
        batches = _make_generation_data(
            num_micro_batches=num_micro_batches, micro_batch_size=micro_batch_size, seq_len=seq_len
        )

        # 1: Compute updates with my engine.
        clone1 = copy.deepcopy(model).to(DEVICE)
        optimizer = optim.Adam(params=clone1.parameters(), lr=lr)
        privacy_engine = PrivacyEngine(
            module=clone1,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            noise_multiplier=0.,  # Remove noise to test gradient clipping & accumulation.
            sample_size=1000,  # Any number suffices for testing.
            epochs=1,  # Any number suffices for testing.
            numerical_stability_constant=0.,  # Important!
            clipping_mode=clipping_mode,
        )
        privacy_engine.attach(optimizer=optimizer)
        optimizer.zero_grad()  # Clears summed_grad, so don't run unless necessary.

        for i, batch in enumerate(batches, 1):
            batch = _prepare_inputs(batch)
            shifted_logits = clone1(**batch).logits[..., :-1, :].permute(0, 2, 1)
            shifted_labels = batch["input_ids"][..., 1:]
            loss = F.cross_entropy(shifted_logits, shifted_labels, reduction="none")
            loss = loss.mean(dim=1)  # Average over sequence.

            del batch
            if i == len(batches):
                optimizer.step(loss=loss)
            else:
                optimizer.virtual_step(loss=loss)
        result1 = [param.grad for param in clone1.parameters()]
        privacy_engine.detach()  # Restore`hooks_mode`.
        del clone1, loss, shifted_labels, shifted_logits, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # 2: Compute grad and clip one-by-one.
        clone2 = copy.deepcopy(model).to(DEVICE)
        optimizer = torch.optim.Adam(params=clone2.parameters(), lr=lr)
        summed_grad = [torch.zeros_like(param) for param in clone2.parameters()]
        for i, batch in tqdm.tqdm(enumerate(batches, 1), desc="over batches"):
            batch = _prepare_inputs(batch)
            for input_ids in tqdm.tqdm(batch["input_ids"], desc="over samples"):
                optimizer.zero_grad()  # Clear previous grad each time!
                input_ids = input_ids[None, :]
                shifted_logits = clone2(input_ids=input_ids).logits[..., :-1, :].permute(0, 2, 1)
                shifted_labels = input_ids[..., 1:]
                loss = F.cross_entropy(shifted_logits, shifted_labels, reduction="none")
                loss = loss.mean()
                loss.backward()

                with torch.no_grad():
                    flat_unclipped_grad = torch.cat(tuple(param.grad.flatten() for param in clone2.parameters()))
                    factor = torch.clamp_max(max_grad_norm / flat_unclipped_grad.norm(), 1.)
                    for si, pi in utils.zip_(summed_grad, list(clone2.parameters())):
                        si.add_(factor * pi.grad)
        result2 = [si / batch_size for si in summed_grad]
        del clone2, loss, shifted_labels, shifted_logits, optimizer

        gc.collect()
        torch.cuda.empty_cache()

        wrong_names = []
        for r1, r2, param_name in utils.zip_(result1, result2, param_names):
            if not torch.allclose(r1, r2, atol=1e-5, rtol=1e-6):
                wrong_names.append(param_name)
        if len(wrong_names) > 0:
            raise AssertionError(
                f"The following parameters have wrong gradients: \n{wrong_names}"
            )


@pytest.mark.parametrize(
    'clipping_mode,tie_word_embeddings,model_name_or_path',
    tuple(itertools.product(["ghost", "default"], [True], ['facebook/bart-base', 't5-base']))
)
def test_conditional_generation(clipping_mode: str, tie_word_embeddings, model_name_or_path):
    if 't5' in model_name_or_path:
        torch.set_default_dtype(torch.float32)  # Unfortunately, can't run double precision on T5.
    gc.collect()
    torch.cuda.empty_cache()

    lr = 1e-4
    num_micro_batches = 4
    micro_batch_size = 4
    seq_len = 128
    batch_size = num_micro_batches * micro_batch_size
    max_grad_norm = 1

    # Catch expected failures.
    with pytest.raises(ValueError) if clipping_mode == "ghost" and tie_word_embeddings else contextlib.nullcontext():

        # Set up model.
        config = transformers.AutoConfig.from_pretrained(model_name_or_path, cache_dir=CACHE_DIR)
        if 'bart' in model_name_or_path:
            config.tie_word_embeddings = tie_word_embeddings
            config.dropout = config.attention_dropout = config.activation_dropout = config.classifier_dropout = 0
        else:  # t5
            config.dropout_rate = 0.0

        model = transformers.AutoModelWithLMHead.from_pretrained(model_name_or_path, config=config, cache_dir=CACHE_DIR)
        if 'bart' in model_name_or_path:
            # TODO: Write per-sample grad hooks to enable optimizing the learned positional embedding layer in BART.
            model.model.encoder.embed_positions.requires_grad_(False)
            model.model.decoder.embed_positions.requires_grad_(False)
        model.train()  # Needed to ensure privacy engine works.

        # Make data.
        batches = _make_encoder_decoder_data(
            num_micro_batches=num_micro_batches, micro_batch_size=micro_batch_size, seq_len=seq_len
        )

        # 1: Compute updates with my engine.
        clone1 = copy.deepcopy(model).to(DEVICE)
        clone1_params = [param for param in clone1.parameters() if param.requires_grad]
        optimizer = optim.Adam(params=clone1.parameters(), lr=lr)
        privacy_engine = PrivacyEngine(
            module=clone1,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            noise_multiplier=0.,  # Remove noise to test gradient clipping & accumulation.
            sample_size=1000,  # Any number suffices for testing.
            epochs=1,  # Any number suffices for testing.
            numerical_stability_constant=0.,  # Important!
            clipping_mode=clipping_mode,
        )
        privacy_engine.attach(optimizer=optimizer)
        optimizer.zero_grad()  # Clears summed_grad, so don't run unless necessary.

        for i, batch in enumerate(batches, 1):
            batch = _prepare_inputs(batch)
            shifted_logits = clone1(**batch).logits[..., :-1, :].permute(0, 2, 1)
            shifted_labels = batch["decoder_input_ids"][..., 1:]
            loss = F.cross_entropy(shifted_logits, shifted_labels, reduction="none")
            loss = loss.mean(dim=1)  # Average over sequence.

            del batch
            if i == len(batches):
                optimizer.step(loss=loss)
            else:
                optimizer.virtual_step(loss=loss)
        # Collect grads.
        result1 = torch.cat([param.grad.flatten() for param in clone1_params])
        privacy_engine.detach()  # Restore`hooks_mode`.
        del clone1, loss, shifted_labels, shifted_logits, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # 2: Compute grad and clip one-by-one.
        clone2 = copy.deepcopy(model).to(DEVICE)
        clone2_params = [param for param in clone2.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(params=clone2.parameters(), lr=lr)
        summed_grad = [torch.zeros_like(param) for param in clone2_params]
        for i, batch in tqdm.tqdm(enumerate(batches, 1), desc="over batches"):
            batch = _prepare_inputs(batch)
            for input_ids, decoder_input_ids in tqdm.tqdm(
                utils.zip_(batch["input_ids"], batch["decoder_input_ids"]),
                desc="over samples"
            ):
                optimizer.zero_grad()  # Clear previous grad each time!
                input_ids = input_ids[None, :]
                decoder_input_ids = decoder_input_ids[None, :]
                outputs = clone2(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
                shifted_logits = outputs.logits[..., :-1, :].permute(0, 2, 1)
                shifted_labels = decoder_input_ids[..., 1:]
                loss = F.cross_entropy(shifted_logits, shifted_labels, reduction="none")
                loss = loss.mean()
                loss.backward()

                with torch.no_grad():
                    flat_unclipped_grad = torch.cat(tuple(param.grad.flatten() for param in clone2_params))
                    factor = torch.clamp_max(max_grad_norm / flat_unclipped_grad.norm(), 1.)
                    for si, pi in utils.zip_(summed_grad, [param for param in clone2_params]):
                        si.add_(factor * pi.grad)
        # Collect grads.
        result2 = torch.cat([si.flatten() for si in summed_grad]) / batch_size
        del clone2, loss, shifted_labels, shifted_logits, optimizer

        gc.collect()
        torch.cuda.empty_cache()

        if 't5' in model_name_or_path:
            # Loosen tolerance, since T5 only runs in half or single.
            torch.testing.assert_allclose(result1, result2, atol=1e-2, rtol=1e-3)
        else:
            torch.testing.assert_allclose(result1, result2, atol=1e-5, rtol=1e-6)

    if 't5' in model_name_or_path:
        torch.set_default_dtype(torch.float64)  # Revert to double precision for other tests.


def test_t5_layer_norm(batch_size=16, hidden_size=128, seq_len=4):
    t5_layer_norm = transformers.models.t5.modeling_t5.T5LayerNorm(hidden_size=hidden_size).to(DEVICE)
    l1, l2 = tuple(copy.deepcopy(t5_layer_norm) for _ in range(2))

    inputs = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)
    targets = torch.randn(batch_size, seq_len, hidden_size, device=DEVICE)

    grad1 = []
    for i, t in utils.zip_(inputs, targets):
        i, t = i[None, :], t[None, :]
        l1.zero_grad()
        (.5 * (l1(i) - t) ** 2.).sum().backward()
        grad1.append(l1.weight.grad.detach().clone())
    grad1 = torch.stack(grad1)

    l2.zero_grad()
    outputs = l2(inputs)
    grad_outputs = (outputs - targets)
    supported_layers_grad_samplers._compute_t5_layer_norm_grad_sample(l2, (inputs,), (grad_outputs,))
    grad2 = l2.weight.grad_sample
    torch.testing.assert_allclose(grad1, grad2, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    'clipping_mode,model_name_or_path,num_labels',
    tuple(
        itertools.product(
            ["ghost", "default"],
            ['google/vit-base-patch16-224', 'facebook/deit-tiny-patch16-224', 'microsoft/beit-base-patch16-224'],
            [10]
        )
    )
)
def test_image_classification(clipping_mode: str, model_name_or_path: str, num_labels: int):
    lr = 1e-4
    num_micro_batches = 4
    micro_batch_size = 4
    batch_size = num_micro_batches * micro_batch_size
    max_grad_norm = 1

    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    config.hidden_dropout_prob = config.attention_probs_dropout_prob = config.drop_path_rate = 0.
    config.num_labels = num_labels
    model = transformers.AutoModelForImageClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True  # Default pre-trained model has 1k classes; we only have 10.
    )
    freeze_isolated_params_for_vit(model)
    model.train()

    batches = _make_image_classification_data(
        num_micro_batches=num_micro_batches, micro_batch_size=micro_batch_size, num_labels=num_labels,
    )
    param_names = [name for name, param in model.named_parameters() if param.requires_grad]

    # 1: Compute updates with my engine.
    clone1 = copy.deepcopy(model).to(DEVICE)
    clone1_params = [param for param in clone1.parameters() if param.requires_grad]

    optimizer = optim.Adam(params=clone1_params, lr=lr)
    privacy_engine = PrivacyEngine(
        module=clone1,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        noise_multiplier=0.,  # Remove noise to test gradient clipping & accumulation.
        sample_size=1000,  # Any number suffices for testing.
        epochs=1,  # Any number suffices for testing.
        numerical_stability_constant=0.,  # Important!
        clipping_mode=clipping_mode,
    )
    privacy_engine.attach(optimizer=optimizer)
    optimizer.zero_grad()  # Clears summed_grad, so don't run unless necessary.
    for i, batch in enumerate(batches, 1):
        batch = _prepare_inputs(batch)
        logits = clone1(**batch).logits
        loss = F.cross_entropy(logits, batch["labels"], reduction="none")

        del batch
        if i == len(batches):
            optimizer.step(loss=loss)
        else:
            optimizer.virtual_step(loss=loss)

    result1 = [param.grad for param in clone1_params]
    privacy_engine.detach()  # Restore`hooks_mode`.
    del clone1, loss, logits, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # 2: Compute grad and clip one-by-one.
    clone2 = copy.deepcopy(model).to(DEVICE)
    clone2_params = [param for param in clone2.parameters() if param.requires_grad]

    optimizer = torch.optim.Adam(params=clone2_params, lr=lr)
    summed_grad = [torch.zeros_like(param) for param in clone2_params]
    for i, batch in tqdm.tqdm(enumerate(batches, 1), desc="over batches"):
        for pixel_values, labels in tqdm.tqdm(utils.zip_(batch["pixel_values"], batch["labels"]), desc="over samples"):
            optimizer.zero_grad(set_to_none=True)  # Clear previous grad each time!
            batch = {"pixel_values": pixel_values[None, ...], "labels": labels[None, ...]}
            batch = _prepare_inputs(batch)
            logits = clone2(**batch).logits
            loss = F.cross_entropy(logits, batch["labels"], reduction="none").sum()
            loss.backward()

            with torch.no_grad():
                flat_unclipped_grad = torch.cat(tuple(param.grad.flatten() for param in clone2_params))
                factor = torch.clamp_max(max_grad_norm / flat_unclipped_grad.norm(), 1.)
                for si, pi in utils.zip_(summed_grad, clone2_params):
                    si.add_(factor * pi.grad)

    result2 = [grad / batch_size for grad in summed_grad]
    del clone2, loss, logits, labels, optimizer

    del model
    gc.collect()
    torch.cuda.empty_cache()

    wrong_names = []
    for r1, r2, param_name in utils.zip_(result1, result2, param_names):
        if not torch.allclose(r1, r2, atol=1e-5, rtol=1e-6):
            wrong_names.append(param_name)
    if len(wrong_names) > 0:
        raise AssertionError(
            f"The following parameters have wrong gradients: \n{wrong_names}"
        )
