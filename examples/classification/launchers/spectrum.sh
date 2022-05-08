#!/bin/sh

python -m classification.spectrum \
  --model_name_or_path "distilroberta-base" \
  --random_init True \
  --dump_path "/mnt/disks/disk-2/dump/spectrum/test" \
  --dtype "float32"
