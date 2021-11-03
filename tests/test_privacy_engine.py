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
from torch import optim
import torch.nn.functional as F
import tqdm
import transformers

from private_transformers import PrivacyEngine

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# Don't mess up my disk space on cluster.
if os.path.exists("/nlp/scr/lxuechen/cache/private_transformers"):
    CACHE_DIR = "/nlp/scr/lxuechen/cache/private_transformers"
else:
    CACHE_DIR = None


def _make_gpt2_data(num_micro_batches=4, micro_batch_size=4, seq_len=128):
    """Make a batch of plain sequences.

    Tuple of multiple micro batches.
    """
    return tuple(
        dict(input_ids=torch.randint(low=1, high=100, size=(micro_batch_size, seq_len)))
        for _ in range(num_micro_batches)
    )


def _make_bert_data(num_micro_batches=2, micro_batch_size=4, seq_len=128, num_labels=2):
    return tuple(
        dict(
            input_ids=torch.randint(low=1, high=100, size=(micro_batch_size, seq_len)),
            labels=torch.randint(low=0, high=num_labels, size=(micro_batch_size,)),
        )
        for _ in range(num_micro_batches)
    )


def _prepare_inputs(batch: dict):
    return {key: value.to(DEVICE) for key, value in batch.items()}


@pytest.mark.parametrize(
    'ghost_clipping,model_name_or_path',
    itertools.product([False, True], ['bert-base-cased', 'roberta-base'])
)
def test_bert(ghost_clipping: bool, model_name_or_path: str):
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
        return_dict=True,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model.requires_grad_(True).train()

    # Make data.
    batches = _make_bert_data(
        micro_batch_size=micro_batch_size, num_micro_batches=num_micro_batches, seq_len=seq_len, num_labels=num_labels,
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
        ghost_clipping=ghost_clipping,
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
    # Collect grads.
    result1 = torch.cat([param.grad.flatten() for param in clone1.parameters()])
    privacy_engine.detach()  # Restore`hooks_mode`.
    del clone1, loss, logits, labels, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    # 2: Compute grad and clip one-by-one.
    clone2 = copy.deepcopy(model).to(DEVICE)
    optimizer = torch.optim.Adam(params=clone2.parameters(), lr=lr)
    summed_grad = [torch.zeros_like(param) for param in clone2.parameters()]
    for i, batch in tqdm.tqdm(enumerate(batches, 1), desc="over batches"):
        batch = _prepare_inputs(batch)
        for input_ids, labels in tqdm.tqdm(zip(batch["input_ids"], batch["labels"]), desc="over samples"):
            optimizer.zero_grad()  # Clear previous grad each time!
            input_ids = input_ids[None, :]
            labels = labels[None]

            logits = clone2(input_ids=input_ids, labels=labels).logits
            loss = F.cross_entropy(logits, labels, reduction="none")
            loss.backward()

            with torch.no_grad():
                flat_unclipped_grad = torch.cat(tuple(param.grad.flatten() for param in clone2.parameters()))
                factor = torch.clamp_max(max_grad_norm / flat_unclipped_grad.norm(), 1.)
                for si, pi in zip(summed_grad, list(clone2.parameters())):
                    si.add_(factor * pi.grad)
    # Collect grads.
    result2 = torch.cat([si.flatten() for si in summed_grad]) / batch_size
    del clone2, loss, logits, labels, optimizer

    del model, batches, config
    gc.collect()
    torch.cuda.empty_cache()

    torch.testing.assert_allclose(result1, result2, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize(
    'ghost_clipping,tie_word_embeddings',
    tuple(itertools.product([False, True], [False, True]))
)
def test_gpt2(ghost_clipping, tie_word_embeddings):
    gc.collect()
    torch.cuda.empty_cache()

    lr = 1e-4
    model_name_or_path = 'gpt2'
    num_micro_batches = 4
    micro_batch_size = 4
    seq_len = 128
    batch_size = num_micro_batches * micro_batch_size
    max_grad_norm = 1

    # Catch expected failures.
    with pytest.raises(ValueError) if ghost_clipping and tie_word_embeddings else contextlib.nullcontext():

        # Set up model.
        config = transformers.GPT2Config.from_pretrained(model_name_or_path, cache_dir=CACHE_DIR)
        config.tie_word_embeddings = tie_word_embeddings
        # Remove potential causes of randomness.
        config.attn_pdrop = config.embd_pdrop = config.resid_pdrop = 0.
        gpt2 = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=CACHE_DIR)
        gpt2.train()  # Needed to ensure privacy engine works.

        # Make data.
        batches = _make_gpt2_data(
            num_micro_batches=num_micro_batches, micro_batch_size=micro_batch_size, seq_len=seq_len
        )

        # 1: Compute updates with my engine.
        clone1 = copy.deepcopy(gpt2).to(DEVICE)
        optimizer = optim.Adam(params=clone1.parameters(), lr=lr)
        privacy_engine = PrivacyEngine(
            module=clone1,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            noise_multiplier=0.,  # Remove noise to test gradient clipping & accumulation.
            sample_size=1000,  # Any number suffices for testing.
            epochs=1,  # Any number suffices for testing.
            numerical_stability_constant=0.,  # Important!
            ghost_clipping=ghost_clipping,
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
        # Collect grads.
        result1 = torch.cat([param.grad.flatten() for param in clone1.parameters()])
        privacy_engine.detach()  # Restore`hooks_mode`.
        del clone1, loss, shifted_labels, shifted_logits, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        # 2: Compute grad and clip one-by-one.
        clone2 = copy.deepcopy(gpt2).to(DEVICE)
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
                    for si, pi in zip(summed_grad, list(clone2.parameters())):
                        si.add_(factor * pi.grad)
        # Collect grads.
        result2 = torch.cat([si.flatten() for si in summed_grad]) / batch_size
        del clone2, loss, shifted_labels, shifted_logits, optimizer

        gc.collect()
        torch.cuda.empty_cache()

        torch.testing.assert_allclose(result1, result2, atol=1e-5, rtol=1e-6)
