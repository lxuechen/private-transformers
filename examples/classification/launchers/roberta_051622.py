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


# python -m classification.launchers.roberta_051622 --task dump_grads_prompt
def dump_grads_prompt():
    cmd = '''python -m classification.run_wrapper \
  --output_dir "/mnt/disks/disk-2/dump/privlm/roberta_prompt/sst-2" \
  --task_name "sst-2" \
  --model_name_or_path "distilroberta-base" \
  --few_shot_type "prompt" \
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


# python -m classification.launchers.roberta_051622 --task get_bases
def get_bases():
    """Perform PCA for grad near local optimum."""
    # TODO: Get this to 2k.
    cmd = '''python -m classification.numerical --task "qr" \
        --grads_dir "/mnt/disks/disk-2/dump/privlm/roberta/sst-2/grad_trajectory" \
        --dump_dir "/mnt/disks/disk-2/dump/privlm/roberta/sst-2/orthproj" \
        --num_ckpts 1000 \
        --varname "flat_grad" \
        --num_power_iteration 500 \
        --k 1000'''
    utils.gpu_scheduler(commands=[cmd])


# python -m classification.launchers.roberta_051622 --task get_bases_prompt
def get_bases_prompt():
    """Perform PCA for grad near local optimum."""
    # TODO: Get this to 2k.
    cmd = '''python -m classification.numerical --task "qr" \
        --grads_dir "/mnt/disks/disk-2/dump/privlm/roberta_prompt/sst-2/grad_trajectory" \
        --dump_dir "/mnt/disks/disk-2/dump/privlm/roberta_prompt/sst-2/orthproj" \
        --num_ckpts 1000 \
        --varname "flat_grad" \
        --num_power_iteration 500 \
        --k 1000'''
    utils.gpu_scheduler(commands=[cmd])


# python -m classification.launchers.roberta_051622 --task retrain
def retrain(seeds=(42, 9008,), run=True):
    cmds = []
    for seed in seeds:
        for rank in (10, 20, 50, 100, None):
            output_dir = f"/mnt/disks/disk-2/dump/privlm/roberta_retrain_{rank}_{seed}/sst-2"
            cmd = f'''python -m classification.run_wrapper \
          --output_dir {output_dir} \
          --task_name "sst-2" \
          --model_name_or_path "distilroberta-base" \
          --few_shot_type "finetune" \
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
                cmd += f' --orthogonal_projection_path "/mnt/disks/disk-2/dump/privlm/roberta/sst-2/orthproj/global_step_000002.pt"'
                cmd += f' --orthogonal_projection_rank {rank}'
            cmds.append(cmd)
    if run:
        utils.gpu_scheduler(commands=cmds)
    return cmds


# python -m classification.launchers.roberta_051622 --task retrain_prompt
def retrain_prompt(seeds=(42, 9008,), run=True):
    cmds = []
    for seed in seeds:
        for rank in (10, 20, 50, 100, None):
            output_dir = f"/mnt/disks/disk-2/dump/privlm/roberta_prompt_retrain_{rank}_{seed}/sst-2"
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
                cmd += f' --orthogonal_projection_path "/mnt/disks/disk-2/dump/privlm/roberta_prompt/sst-2/orthproj/global_step_000002.pt"'
                cmd += f' --orthogonal_projection_rank {rank}'
            cmds.append(cmd)

    if run:
        utils.gpu_scheduler(commands=cmds)
    return cmds

# python -m classification.launchers.roberta_051622 --task retrain_all
def retrain_all():
    cmds = retrain(run=False) + retrain_prompt(run=False)
    utils.gpu_scheduler(commands=cmds)


def main(
    task='dump_grads',
):
    utils.runs_tasks(
        task=task,
        task_names=(
            "dump_grads", "dump_grads_prompt", "get_bases", "get_bases_prompt", "retrain", "retrain_prompt", "retrain_all"
        ),
        task_callables=(
            dump_grads, dump_grads_prompt, get_bases, get_bases_prompt, retrain, retrain_prompt, retrain_all
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
