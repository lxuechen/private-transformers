from ml_swissknife import utils
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers.data.data_collator import default_data_collator

from . import density
from . import spectrum_utils
from ..run_classification import DynamicDataTrainingArguments
from ..src import common
from ..src.common import device


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

    model.classifier.requires_grad_(False)
    for name, param in model.named_parameters():
        if 'layer.0' in name or 'layer.1' in name:
            param.requires_grad_(True)

    num_tot_params = utils.count_parameters(model)
    num_dif_params = utils.count_parameters(model, only_differentiable=True)
    print(f"Total params: {num_tot_params / 1e6:.4f}m. Differentiable params: {num_dif_params / 1e6:.4f}m")
    return model


def make_loss_fn(scale=1.):
    """Make the usual classification loss with custom scale.

    Non-identity scale helps to avoid tiny eigenvalues which Lanczos fails to report.
    """

    def loss_fn(batch, model):
        device = next(iter(model.parameters())).device
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        losses = F.cross_entropy(logits.logits, batch["labels"], reduction="none") * scale
        return losses

    return loss_fn


def test_make_spectrum_lanczos(
    scale=100.,
    dump_dir="/mnt/disks/disk-2/dump/spectrum/test_spectrum"
):
    utils.makedirs(dump_dir, exist_ok=True)
    torch.set_default_dtype(torch.float64)

    model_name_or_path = "distilroberta-base"
    task_name = "sst-2"
    data_dir = "classification/data/original"
    max_seq_length = 128
    max_lanczos_iter = 1000
    tol = 1e-9

    train_batch_size = 100
    max_batches = 5
    n_total = max_batches * train_batch_size

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

    loss_fn = make_loss_fn(scale=scale)
    lanczos_outputs = spectrum_utils.make_spectrum_lanczos(
        model=model,
        loader=train_loader,
        max_batches=max_batches,
        max_lanczos_iter=max_lanczos_iter,
        tol=tol,
        loss_fn=loss_fn,
        return_dict=True,
    )
    exact_eigenvals = spectrum_utils.make_spectrum_exact(
        model=model,
        loader=train_loader_singleton,
        max_batches=n_total,
        loss_fn=loss_fn,
    )
    torch.save(
        {"lanczos_outputs": lanczos_outputs, "exact_eigenvals": exact_eigenvals},
        utils.join(dump_dir, "eigenvals.pt")
    )

    sigma_squared = 0.001
    lanczos_density, lanczos_grids = density.tridiag_to_density(
        [lanczos_outputs["T"].cpu().numpy()],
        sigma_squared=sigma_squared
    )
    exact_density, exact_grids = density.tridiag_to_density(
        [torch.diag(exact_eigenvals).cpu().numpy()],
        sigma_squared=sigma_squared
    )

    import matplotlib.pyplot as plt
    plt.figure(dpi=300)
    plt.semilogy(lanczos_grids, lanczos_density + 1.0e-7, label='Lanczos')
    plt.semilogy(exact_grids, exact_density + 1.0e-7, label='exact')
    plt.xlabel('$\lambda$')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(utils.join(dump_dir, "vis.png"))


if __name__ == "__main__":
    # python -m classification.spectrum.test_spectrum
    test_make_spectrum_lanczos()
