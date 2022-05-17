"""
python -m classification.launchers.roberta_051622
"""

import fire
from swissknife import utils


def dump_grads():
    cmd = '''python -m classification.run_wrapper \
  --output_dir "/mnt/disks/disk-2/dump/privlm/roberta/sst-2" \
  --task_name "sst-2" \
  --model_name_or_path "distilroberta-base" \
  --few_shot_type "finetune" \
  --attention_only "yes" \
  --static_lm_head "yes" \
  --per_device_train_batch_size 25 \
  --batch_size 1000 \
  --ghost_clipping "no" \
  --num_train_epochs 32 \
  --eval_spectrum "no" \
  --non_private "no" \
  --eval_steps 50 \
  --randomly_initialize "no" \
  --store_grads "yes"'''
    utils.gpu_scheduler(commands=[cmd])


def get_bases():
    """Perform PCA for grad near local optimum."""
    cmd = '''python -m classification.numerical --task "qr"\
        --grads_dir "/mnt/disks/disk-2/dump/privlm/roberta/sst-2/grad_trajectory" \
        --dump_dir "/mnt/disks/disk-2/dump/privlm/roberta/sst-2/orthproj" \
        --num_ckpts 2000 \
        --varname "flat_grad" \
        --num_power_iteration 100 \
        --k 2000'''
    utils.gpu_scheduler(commands=[cmd])


def main(
    task='dump_grads',
):
    utils.runs_tasks(
        task=task,
        task_names=("dump_grads", "get_bases"),
        task_callables=(dump_grads, get_bases)
    )


if __name__ == "__main__":
    fire.Fire(main)
