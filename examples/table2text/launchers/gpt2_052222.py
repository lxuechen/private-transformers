"""
Same eigen-analysis for GPT-2.
"""

import fire
from swissknife import utils


def dump_grads(run=True):
    cmd = '''python -m table2text.launchers.e2e_run_wrapper \
        --output_dir "/mnt/disks/disk-2/dump/privlm2/gpt2/e2e" \
        --model_name_or_path "distilgpt2" \
        --num_train_epochs 50 \
        --store_grads "yes" \
        --eval_epochs 10
        '''
    if run:
        utils.gpu_scheduler(commands=[cmd])
    return cmd


def get_bases(seed=42, run=True):
    """Perform PCA for grad near local optimum."""
    cmd = f'''python -m classification.numerical \
        --grads_dir "/mnt/disks/disk-2/dump/privlm2/gpt2/e2e/grad_trajectory" \
        --dump_dir "/mnt/disks/disk-2/dump/privlm2/gpt2/e2e/orthproj_{seed}" \
        --n 4000 \
        --k 1000 \
        --num_power_iteration 400 \
        --seed {seed}'''
    if run:
        utils.gpu_scheduler(commands=[cmd])
    return cmd


# python -m table2text.launchers.gpt2_052222 --task "dump_and_pca"
def dump_and_pca():
    procs = utils.gpu_scheduler(commands=[dump_grads(run=False)])
    for proc in procs:
        proc.wait()  # Wait for training to finish, then run parallel seed.
    cmds = [get_bases(seed=seed, run=False) for seed in (42, 9009, 101)]
    utils.gpu_scheduler(commands=cmds)


def main(task):
    utils.runs_tasks(
        task=task,
        task_names=("dump_grads", "get_bases", "dump_and_pca"),
        task_callables=(dump_grads, get_bases, dump_and_pca)
    )


if __name__ == "__main__":
    fire.Fire(main)
