"""
"""

import fire
from swissknife import utils
from torch.utils.data import DataLoader, RandomSampler
import transformers
from transformers.data.data_collator import default_data_collator

from . import common
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


def main(
    model_name_or_path="distilroberta-base",
    task_name="sst-2",
    data_dir="classification/data/original",
    train_batch_size=128,
):
    config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels_mapping[task_name],
        finetuning_task=task_name,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    data_args = DynamicDataTrainingArguments(
        task_name=task_name, data_dir=utils.join(data_dir, common.task_name2suffix_name[task_name])
    )
    train_dataset = transformers.GlueDataset(data_args, tokenizer, mode="train")

    filter_params(model)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=train_batch_size,
        collate_fn=default_data_collator,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    batch = next(iter(train_loader))
    import pdb; pdb.set_trace()


# python -m classification.spectrum
if __name__ == "__main__":
    fire.Fire(main)
