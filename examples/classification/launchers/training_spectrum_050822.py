"""
Evaluate the spectrum during training.

python -m classification.launchers.training_spectrum_050822
"""

import fire
from swissknife import utils

date = "training_spectrum_050822"


def _get_cmd(randomly_initialize: str, model_name_or_path="distilroberta-base"):
    output_dir = f"/mnt/disks/disk-2/dump/classification/{date}/{model_name_or_path}_{randomly_initialize}"
    return f'''python -m classification.run_wrapper \
  --output_dir "{output_dir}" \
  --task_name "sst-2" \
  --model_name_or_path {model_name_or_path} \
  --few_shot_type "prompt" \
  --attention_only "yes" \
  --static_lm_head "yes" \
  --static_embedding "no" \
  --per_device_train_batch_size 25 \
  --eval_spectrum "yes" \
  --eval_steps 60 \
  --max_spectrum_batches 100 \
  --max_lanczos_iter 100 \
  --randomly_initialize {randomly_initialize}
'''


def main():
    cmds = []
    for randomly_initialize in ('yes', 'no'):
        cmds.append(_get_cmd(randomly_initialize=randomly_initialize))
    utils.gpu_scheduler(cmds, excludeID=[0, 1], excludeUUID=[0, 1])


if __name__ == "__main__":
    fire.Fire(main)
