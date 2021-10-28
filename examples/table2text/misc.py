"""Miscellaneous utilities.

Mostly bespoke data loaders at the moment.
"""

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    PreTrainedTokenizer
)

from .compiled_args import DataTrainingArguments
from .data_utils.data_collator import DataCollatorForData2TextLanguageModeling
from .data_utils.language_modeling import LineByLineE2ETextDataset, LineByLineTriplesTextDataset


def get_dataset_with_path(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    file_path: str,
    max_examples: int,
    **_,
):
    if data_args.line_by_line:
        if data_args.task_mode == 'e2e':
            dataset = LineByLineE2ETextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=data_args.block_size,
                bos_tok=tokenizer.bos_token,
                eos_tok=tokenizer.eos_token,
                max_seq_len=data_args.max_seq_len,
                max_examples=max_examples,
            )
        elif data_args.task_mode == 'dart':
            dataset = LineByLineTriplesTextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=data_args.block_size,
                bos_tok=tokenizer.bos_token,
                eos_tok=tokenizer.eos_token,
                max_seq_len=data_args.max_seq_len,
                max_examples=max_examples,
            )
        else:
            raise ValueError(f"Unknown `args.task_mode`: {data_args.task_mode}")

    else:
        raise ValueError("table2text task don't support anything other than line_by_line!")
    return dataset


def get_prompt_dataset(file_path, tokenizer):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    encoded_lines = [
        tokenizer.encode(line.strip(), add_special_tokens=False, return_tensors="pt")
        for line in lines
    ]
    return encoded_lines


def get_all_datasets(config, tokenizer, data_args, model_args, **_):
    kwargs = dict(data_args=data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
    train_dataset = get_dataset_with_path(
        **kwargs, file_path=data_args.train_data_file, max_examples=data_args.max_train_examples
    )
    valid_dataset = get_dataset_with_path(
        **kwargs, file_path=data_args.valid_data_file, max_examples=data_args.max_valid_examples
    )
    eval_dataset = get_dataset_with_path(
        **kwargs, file_path=data_args.eval_data_file, max_examples=data_args.max_eval_examples
    )

    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        if data_args.task_mode == 'e2e' or data_args.task_mode == 'dart':
            data_collator = DataCollatorForData2TextLanguageModeling(
                tokenizer=tokenizer, mlm=False, format_mode=data_args.format_mode
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False,
            )

    return train_dataset, valid_dataset, eval_dataset, data_collator
