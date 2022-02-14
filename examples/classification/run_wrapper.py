"""Wrapper launcher script."""

import os

import fire


def _get_command(
    task_name,
    output_dir,
    model_name_or_path,
    data_dir,
    ghost_clipping,
    non_private,
    target_epsilon,
    few_shot_type,
    per_device_train_batch_size=20,
    eval_steps=10,
):
    task_name_to_factor = {
        "sst-2": 1, "qnli": 2, "qqp": 6, "mnli": 6,
    }
    factor = task_name_to_factor[task_name]

    base_batch_size = 1000
    base_num_train_epochs = 3

    # This batch size selection roughly ensures the sampling rates on different
    # datasets are in the same ballpark.
    batch_size = int(base_batch_size * factor)
    num_train_epochs = int(base_num_train_epochs * factor)

    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    data_dir_suffix = {
        "sst-2": "GLUE-SST-2",
        "mnli": "MNLI",
        "qqp": "QQP",
        "qnli": "QNLI",
    }[task_name]
    data_dir = f"{data_dir}/{data_dir_suffix}"

    template = {
        "sst-2": "*cls**sent_0*_It_was*mask*.*sep+*",
        "mnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qqp": "*cls**sent-_0**mask*,*+sentl_1**sep+*",
    }[task_name]

    # Epochs chosen roughly to match e2e number of updates. We didn't hyperparameter tune on classification tasks :)
    return f'''
python -m classification.run_classification \
  --task_name {task_name} \
  --data_dir {data_dir} \
  --output_dir {output_dir} \
  --overwrite_output_dir \
  --model_name_or_path {model_name_or_path} \
  --few_shot_type {few_shot_type} \
  --num_k 1 \
  --num_sample 1 --seed 0 \
  --template {template} \
  --non_private {non_private} \
  --num_train_epochs {num_train_epochs} \
  --target_epsilon {target_epsilon} \
  --per_device_train_batch_size {per_device_train_batch_size} \
  --gradient_accumulation_steps {gradient_accumulation_steps} \
  --per_device_eval_batch_size 8 \
  --per_example_max_grad_norm 0.1 --ghost_clipping {ghost_clipping} \
  --learning_rate 0.0005 \
  --lr_decay yes \
  --adam_epsilon 1e-08 \
  --weight_decay 0 \
  --max_seq_len 256 \
  --evaluation_strategy steps --eval_steps {eval_steps} --evaluate_before_training True \
  --do_train --do_eval \
  --first_sent_limit 200 --other_sent_limit 200 --truncate_head yes
    '''


def main(
    output_dir,
    task_name,
    few_shot_type="prompt",
    model_name_or_path="roberta-base",
    data_dir="classification/data/original",
    ghost_clipping="yes",
    non_private="no",
    target_epsilon=8,
):
    command = _get_command(
        output_dir=output_dir,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        ghost_clipping=ghost_clipping,
        non_private=non_private,
        target_epsilon=target_epsilon,
        few_shot_type=few_shot_type,
    )
    print('Running command:')
    print(command)
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
