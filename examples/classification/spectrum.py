import logging
from typing import Callable

import fire
from swissknife import utils
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import transformers
from transformers.data.data_collator import default_data_collator

from . import common, lanczos
from .common import device
from .run_classification import DynamicDataTrainingArguments
from .src.processors import num_labels_mapping


def filter_params(
    model: transformers.RobertaForSequenceClassification,
    filter_criteria=lambda name: "attention" in name or "classifier" in name,
):
    """Set only the attention and lm-head params to require grads."""
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if filter_criteria(name):
            param.requires_grad_(True)
    num_tot_params = utils.count_parameters(model)
    num_dif_params = utils.count_parameters(model, only_differentiable=True)
    print(f"Total params: {num_tot_params / 1e6:.4f}m. Differentiable params: {num_dif_params / 1e6:.4f}m")


def make_loss(batch: dict, model: transformers.RobertaForSequenceClassification):
    batch = {key: value.to(device) for key, value in batch.items()}
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    losses = F.cross_entropy(logits.logits, batch["labels"], reduction="none")
    return losses


def make_matmul_closure(
    model: transformers.RobertaForSequenceClassification,
    loader: DataLoader,
    max_batches: int,
    loss_fn: Callable,
):
    """Make covariance-vector product closure."""
    params = [param for param in model.parameters() if param.requires_grad]  # Collect diff-able.
    shapes = [param.size() for param in params]

    def matmul_closure(vector: torch.Tensor):
        """Compute G^t G v; jpv then vjp.

        This is the uncentered covariance.
        """
        output = torch.zeros_like(vector)
        vectors = utils.flat_to_shape(vector, shapes)

        batch_size = next(iter(loader))["input_ids"].size(0)
        n_total = max_batches * batch_size
        for batch_idx, batch in tqdm.tqdm(enumerate(loader), total=max_batches):
            if batch_idx >= max_batches:
                break

            with torch.enable_grad():
                losses = loss_fn(batch=batch, model=model)
                Gv = utils.jvp(outputs=losses, inputs=params, grad_inputs=vectors)  # (n,).
                GtGv = utils.vjp(outputs=losses, inputs=params, grad_outputs=Gv)  # (d,).

            GtGv = utils.flatten(GtGv)[:, None]
            output += GtGv / n_total

        return output

    return matmul_closure


@torch.no_grad()
def make_spectrum_lanczos(
    model: transformers.RobertaForSequenceClassification,
    loader: DataLoader,
    max_batches: int,
    max_lanczos_iter: int,
    loss_fn: Callable,
    tol=1e-5,
    return_dict=False,
):
    model.to(device).eval()

    numel = sum(param.numel() for param in model.parameters() if param.requires_grad)

    Q, T = lanczos.lanczos_tridiag(
        make_matmul_closure(model=model, loader=loader, max_batches=max_batches, loss_fn=loss_fn),
        max_iter=max_lanczos_iter,
        dtype=torch.get_default_dtype(),
        device=device,
        matrix_shape=(numel,),
        tol=tol,
    )
    if not torch.all(torch.diag(T, diagonal=1) == torch.diag(T, diagonal=-1)):
        logging.warning("Lanczos output failed tri-diagonality check!")

    eigenvals, eigenvecs = torch.linalg.eigh(T)
    logging.warning("Lanczos eigenvalues:")
    logging.warning(eigenvals)
    if return_dict:
        return dict(Q=Q, T=T, eigenvecs=eigenvecs, eigenvals=eigenvals)
    return eigenvals


@torch.no_grad()
def make_spectrum_exact(
    model: transformers.RobertaForSequenceClassification,
    loader: DataLoader,  # Must be singleton.
    max_batches: int,
    loss_fn: Callable,
):
    model.to(device).eval()

    params = [param for param in model.parameters() if param.requires_grad]

    grads = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        with torch.enable_grad():
            losses = loss_fn(batch=batch, model=model)
            grad = torch.autograd.grad(losses.squeeze(), params)
        grads.append(utils.flatten(grad).detach())

    grads = torch.stack(grads)  # (n, d).
    GtG = grads.t() @ grads / max_batches
    eigenvals, eigenvecs = torch.linalg.eigh(GtG)
    logging.warning("Exact eigenvalues:")
    logging.warning(eigenvals)
    return eigenvals


def make_model_and_loader(
    model_name_or_path,
    task_name,
    data_dir,
    max_seq_length,
    batch_size,
):
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
        batch_size=batch_size,
        collate_fn=default_data_collator,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels_mapping[task_name],
        finetuning_task=task_name,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    filter_params(model)

    return model, train_loader


def get_init_eigenvals(
    model_name_or_path,
    task_name,
    data_dir,
    batch_size,
    max_seq_length,

    max_batches,
    max_lanczos_iter,
    tol,
    random_init=False,
):
    model, loader = make_model_and_loader(
        model_name_or_path=model_name_or_path,
        task_name=task_name,
        data_dir=data_dir,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    if random_init:
        model.init_weights()

    return make_spectrum_lanczos(
        model=model,
        loader=loader,
        max_batches=max_batches,
        max_lanczos_iter=max_lanczos_iter,
        loss_fn=make_loss,
        tol=tol,
        return_dict=True
    )


def main(
    model_name_or_path="roberta-base",
    task_name="sst-2",
    data_dir="classification/data/original",
    batch_size=32,
    max_seq_length=128,
    max_lanczos_iter=100,
    max_batches=400,
    tol=1e-7,

    random_init=False,
    dump_path="/mnt/disks/disk-2/dump/spectrum/init_compare/dump.pt",
    dtype="float64",
):
    torch.set_default_dtype(utils.get_dtype(dtype))
    utils.makedirs(utils.dirname(dump_path), exist_ok=True)

    kwargs = dict(
        model_name_or_path=model_name_or_path,
        task_name=task_name,
        data_dir=data_dir,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_lanczos_iter=max_lanczos_iter,
        max_batches=max_batches,
        tol=tol,
        random_init=random_init,
    )
    outputs = get_init_eigenvals(**kwargs)
    torch.save(
        dict(**kwargs, **outputs),
        dump_path,
    )


if __name__ == "__main__":
    # python -m classification.spectrum --max_lanczos_iter 1 --max_batches 1
    fire.Fire(main)
