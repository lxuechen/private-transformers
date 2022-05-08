#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python -m classification.run_wrapper \
  --output_dir "/mnt/disks/disk-2/dump/classification/test" \
  --task_name "sst-2" \
  --model_name_or_path "distilroberta-base" \
  --few_shot_type "prompt" \
  --attention_only "yes" \
  --static_lm_head "yes" \
  --static_embedding "no" \
  --per_device_train_batch_size 20 \
  --eval_spectrum "yes" \
  --eval_steps 1
