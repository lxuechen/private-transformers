import logging

import fire
import gpytorch
from swissknife import utils
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import transformers
from transformers.data.data_collator import default_data_collator

from . import common
from .run_classification import DynamicDataTrainingArguments
from .src.processors import num_labels_mapping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def make_matmul_closure(
    model: transformers.RobertaForSequenceClassification,
    loader: DataLoader,
    max_batches: int,
):
    """Make covariance-vector product closure."""
    model.eval()

    params = [param for param in model.parameters() if param.requires_grad]  # Collect diff-able.
    shapes = [param.size() for param in params]

    @torch.enable_grad()
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
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            losses = F.cross_entropy(logits.logits, batch["labels"], reduction="none")
            Gv = utils.jvp(outputs=losses, inputs=params, grad_inputs=vectors)  # (n,).
            GtGv = utils.vjp(outputs=losses, inputs=params, grad_outputs=Gv)  # (d,).

            with torch.no_grad():
                GtGv = utils.flatten(GtGv)[:, None]
                output += GtGv / n_total

        return output

    return matmul_closure


def make_spectrum_lanczos(
    model: transformers.RobertaForSequenceClassification,
    loader: DataLoader,
    max_batches: int,
    max_lanczos_iter: int,
):
    numel = sum(param.numel() for param in model.parameters() if param.requires_grad)

    Q, T = gpytorch.utils.lanczos.lanczos_tridiag(
        make_matmul_closure(model=model, loader=loader, max_batches=max_batches),
        max_iter=max_lanczos_iter,
        dtype=torch.get_default_dtype(),
        device=device,
        matrix_shape=(numel,),
    )
    if not torch.all(torch.diag(T, diagonal=1) == torch.diag(T, diagonal=-1)):
        logging.warning("Lanczos output failed tri-diagonality check!")

    eigenvals, eigenvecs = torch.linalg.eigh(T)
    logging.warning("eigenvalues:")
    logging.warning(eigenvals)
    return eigenvals


def make_spectrum_exact(
    model,
    loader,
):
    pass


def main(
    model_name_or_path="distilroberta-base",
    task_name="sst-2",
    data_dir="classification/data/original",
    train_batch_size=128,
    max_seq_length=128,
    max_lanczos_iter=100,
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
        batch_size=train_batch_size,
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
    model.to(device)
    filter_params(model)


# python -m classification.spectrum
if __name__ == "__main__":
    fire.Fire(main)
