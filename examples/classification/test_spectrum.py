from swissknife import utils
import torch
from torch.utils.data import DataLoader
import transformers
from transformers.data.data_collator import default_data_collator

from . import common, spectrum
from .common import device
from .run_classification import DynamicDataTrainingArguments


def make_test_model():
    """Make a tiny model with 9.7k params."""
    # Tiny model to test correctness.
    config = transformers.RobertaConfig(
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=8,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.1,
        vocab_size=50265,
    )
    model = transformers.RobertaForSequenceClassification(config=config)
    model.requires_grad_(False)

    model.classifier.requires_grad_(True)
    for name, param in model.named_parameters():
        if 'layer.0' in name:
            param.requires_grad_(True)

    num_tot_params = utils.count_parameters(model)
    num_dif_params = utils.count_parameters(model, only_differentiable=True)
    print(f"Total params: {num_tot_params / 1e6:.4f}m. Differentiable params: {num_dif_params / 1e6:.4f}m")
    return model


def test_make_spectrum_lanczos():
    torch.set_default_dtype(torch.float64)

    model_name_or_path = "distilroberta-base"
    task_name = "sst-2"
    data_dir = "classification/data/original"
    max_seq_length = 128
    max_lanczos_iter = 1000
    tol = 1e-8

    train_batch_size = 100
    max_batches = 2
    max_batches_singleton = max_batches * train_batch_size

    model = make_test_model().to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    data_args = DynamicDataTrainingArguments(
        task_name=task_name,
        data_dir=utils.join(data_dir, common.task_name2suffix_name[task_name]),
        max_seq_length=max_seq_length,
    )
    train_dataset = transformers.GlueDataset(data_args, tokenizer, mode="train")

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=train_batch_size,
        collate_fn=default_data_collator,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    train_loader_singleton = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=default_data_collator,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    lanczos_eigenvals = spectrum.make_spectrum_lanczos(
        model=model,
        loader=train_loader,
        max_batches=max_batches,
        max_lanczos_iter=max_lanczos_iter,
        tol=tol
    )
    exact_eigenvals = spectrum.make_spectrum_exact(
        model=model,
        loader=train_loader_singleton,
        max_batches=max_batches_singleton,
    )


if __name__ == "__main__":
    # python -m classification.test_spectrum
    test_make_spectrum_lanczos()
