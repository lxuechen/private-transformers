import fire
from swissknife import utils


def _get_cmd(rank, seed):
    output_dir = f"/mnt/disks/disk-2/dump/classification/lowrank_rank_{rank}_seed_{seed}"
    cmd = f'''python -m classification.run_wrapper \
        --output_dir {output_dir} \
        --task_name "sst-2" \
        --model_name_or_path "distilroberta-base" \
        --few_shot_type "prompt" \
        --attention_only "yes" \
        --static_lm_head "yes" \
        --static_embedding "no" \
        --per_device_train_batch_size 25 \
        --batch_size 1000 \
        --num_train_epochs 8 \
        --eval_spectrum "no" \
        --non_private "no" \
        --eval_steps 25 \
        --randomly_initialize "no" \
        --ghost_clipping "no" \
        --seed {seed}'''
    if rank is not None:
        cmd += ' --orthogonal_projection_path "/mnt/disks/disk-2/dump/classification/test/orthogonal_projection.pt"'
        cmd += f' --orthogonal_projection_rank {rank}'
    return cmd


def main(
    seeds=(0, 1, 2),
):
    # python -m classification.launchers.lowrank_051322
    cmds = []
    for seed in seeds:
        for rank in (1, 10, 100, None):
            cmds.append(_get_cmd(rank=rank, seed=seed))
    utils.gpu_scheduler(commands=cmds, excludeID=(0,), excludeUUID=(0,))


if __name__ == "__main__":
    fire.Fire(main)
