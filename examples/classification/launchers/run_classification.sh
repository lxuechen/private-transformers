#!/bin/sh
# bash classification/launchers/run_classification.sh
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python -m classification.run_wrapper \
  --output_dir "/mnt/disks/disk-2/dump/classification/test" \
  --task_name "sst-2" \
  --model_name_or_path "distilroberta-base" \
  --few_shot_type "prompt" \
  --attention_only "yes" \
  --static_lm_head "yes" \
  --static_embedding "no" \
  --per_device_train_batch_size 25 \
  --batch_size 1000 \
  --num_train_epochs 32 \
  --eval_spectrum "no" \
  --non_private "no" \
  --eval_steps 50 \
  --max_spectrum_batches 400 \
  --max_lanczos_iter 100 \
  --randomly_initialize "no" \
  --ghost_clipping "no" \
  --store_grads "yes" \
  --orthogonal_projection_path "/mnt/disks/disk-2/dump/classification/test/orthogonal_projection.pt" \
  --orthogonal_projection_rank 100
