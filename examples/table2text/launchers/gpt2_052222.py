"""
Same eigen-analysis for GPT-2.
"""

import fire
from swissknife import utils


def dump_grads(run=True):
    cmd = '''python -m table2text.launchers.e2e_run_wrapper \
        --output_dir "/mnt/disks/disk-2/dump/privlm2/gpt2/e2e" \
        --model_name_or_path "distilgpt2" \
        --num_train_epochs 100 \
        --store_grads "yes" \
        --eval_epochs 20 \
        --learning_rate 1e-4
        '''
    if run:
        utils.gpu_scheduler(commands=[cmd])
    return cmd


# python -m table2text.launchers.gpt2_052222 --task "get_bases"
def get_bases(seed=42, run=True, start_index=0, n=4000, k=500):
    """Perform PCA for grad near local optimum."""
    cmd = f'''python -m classification.numerical \
        --grads_dir "/mnt/disks/disk-2/dump/privlm2/gpt2/e2e/grad_trajectory" \
        --dump_dir "/mnt/disks/disk-2/dump/privlm2/gpt2/e2e/orthproj_{seed}" \
        --n {n} \
        --k {k} \
        --num_power_iteration 101 \
        --seed {seed} \
        --start_index {start_index}'''
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


def retrain(seeds=(42, 9008, 0), run=True, global_step=10):
    cmds = []
    for seed in seeds:
        for rank in (10, 20, 50, 100, None):
            output_dir = f"/mnt/disks/disk-2/dump/privlm2/gpt2_retrain_{rank}_{seed}/e2e"
            cmd = f'''python -m table2text.launchers.e2e_run_wrapper \
                --model_name_or_path "distilgpt2" \
                --output_dir {output_dir} \
                --seed {seed} \
                --max_generations 50000'''
            if rank is not None:
                cmd += f' --orthogonal_projection_path ' \
                       f'"/mnt/disks/disk-2/dump/privlm/gpt2/e2e/orthproj_42/all/global_step_{global_step:06d}.pt"'
                cmd += f' --orthogonal_projection_rank {rank}'
            cmds.append(cmd)
    if run:
        utils.gpu_scheduler(commands=cmds)
    return cmds


def main(task):
    utils.runs_tasks(
        task=task,
        task_names=("dump_grads", "get_bases", "dump_and_pca", "retrain"),
        task_callables=(dump_grads, get_bases, dump_and_pca, retrain)
    )


if __name__ == "__main__":
    fire.Fire(main)
