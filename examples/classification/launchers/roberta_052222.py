import fire
from swissknife import utils


# python -m classification.launchers.roberta_052222 --task dump_grads_prompt
def dump_grads_prompt(run=True):
    cmd = '''python -m classification.run_wrapper \
  --output_dir "/mnt/disks/disk-2/dump/privlm2/roberta_prompt/sst-2" \
  --task_name "sst-2" \
  --model_name_or_path "distilroberta-base" \
  --few_shot_type "prompt" \
  --attention_only "yes" \
  --static_lm_head "yes" \
  --per_device_train_batch_size 25 \
  --batch_size 1000 \
  --ghost_clipping "no" \
  --num_train_epochs 64 \
  --eval_spectrum "no" \
  --non_private "no" \
  --eval_steps 50 \
  --randomly_initialize "no" \
  --store_grads "yes"'''
    if run:
        utils.gpu_scheduler(commands=[cmd])
    return cmd


# python -m classification.launchers.roberta_052222 --task get_bases_prompt
def get_bases_prompt(seed=42, run=True):
    """Perform PCA for grad near local optimum."""
    cmd = f'''python -m classification.numerical \
        --grads_dir "/mnt/disks/disk-2/dump/privlm2/roberta_prompt/sst-2/grad_trajectory" \
        --dump_dir "/mnt/disks/disk-2/dump/privlm2/roberta_prompt/sst-2/orthproj_{seed}" \
        --n 4000 \
        --k 1000 \
        --num_power_iteration 400 \
        --seed {seed}'''
    if run:
        utils.gpu_scheduler(commands=[cmd])
    return cmd


# TODO: Decide global_step.
# python -m classification.launchers.roberta_052222 --task retrain_prompt
def retrain_prompt(seeds=(42, 9008, 0), run=True, global_step=6):
    cmds = []
    for seed in seeds:
        for rank in (10, 20, 50, 100, None):
            output_dir = f"/mnt/disks/disk-2/dump/privlm2/roberta_prompt_retrain_{rank}_{seed}/sst-2"
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
          --ghost_clipping "no" \
          --num_train_epochs 4 \
          --eval_spectrum "no" \
          --non_private "no" \
          --eval_steps 25 \
          --randomly_initialize "no" \
          --seed {seed}'''
            if rank is not None:
                cmd += f' --orthogonal_projection_path ' \
                       f'"/mnt/disks/disk-2/dump/privlm/roberta2_prompt/sst-2/orthproj/global_step_{global_step:06d}.pt"'
                cmd += f' --orthogonal_projection_rank {rank}'
            cmds.append(cmd)

    if run:
        utils.gpu_scheduler(commands=cmds)
    return cmds


# python -m classification.launchers.roberta_052222 --task retrain_all
def retrain_all():
    cmds = retrain_prompt(run=False)
    utils.gpu_scheduler(commands=cmds)


# python -m classification.launchers.roberta_052222 --task dump_and_pca
def dump_and_pca():
    procs = utils.gpu_scheduler(commands=[dump_grads_prompt(run=False)])
    for proc in procs:
        proc.wait()  # Wait for training to finish, then run parallel seed.
    cmds = [get_bases_prompt(seed=seed, run=False) for seed in (42, 9009, 101)]
    utils.gpu_scheduler(commands=cmds)


def main(
    task='dump_grads',
):
    utils.runs_tasks(
        task=task,
        task_names=(
            "dump_grads_prompt",
            "get_bases_prompt",
            "retrain_prompt",
            "retrain_all",
            "dump_and_pca"
        ),
        task_callables=(
            dump_grads_prompt,
            get_bases_prompt,
            retrain_prompt,
            retrain_all,
            dump_and_pca
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
