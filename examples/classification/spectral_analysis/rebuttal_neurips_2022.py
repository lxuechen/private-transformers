"""Experiments ran pre- and post-rebuttals."""
import os

import fire
from ml_swissknife import utils


def run_save_grads(num_train_epochs=60):
    commands = []
    for model_name_or_path in ("roberta-base", "roberta-large"):
        output_dir = utils.join("/mnt/data1/dump/", 'rebuttal', f'run-{model_name_or_path}')
        command = f'''python -m classification.run_wrapper \
            --output_dir {output_dir} \
            --task_name "sst-2" \
            --model_name_or_path "{model_name_or_path}" \
            --attention_only "yes" \
            --static_lm_head "yes" \
            --num_train_epochs {num_train_epochs} \
            --eval_spectrum "no" \
            --non_private "no" \
            --eval_steps 50 \
            --randomly_initialize "no" \
            --per_device_train_batch_size 25 \
            --batch_size 1000 \
            --clipping_mode "default" \
            --store_grads "yes"'''
        commands.append(command)
    utils.gpu_scheduler(commands, excludeID=(0,), excludeUUID=(0,))
    return commands


def run_pca():
    # python -m classification.spectral_analysis.rebuttal_neurips_2022 --task "run_pca"
    command = 'python -m classification.spectral_analysis.main \
        --task "pca" \
        --n 2000 \
        --k 500 \
        --batch_size 40 \
        --train_dir "/mnt/data1/dump/rebuttal/run-roberta-base" \
        --num_power_iteration 10'
    os.system(command)

    command = 'python -m classification.spectral_analysis.main \
        --task "pca" \
        --n 2000 \
        --k 500 \
        --train_dir "/mnt/data1/dump/rebuttal/run-roberta-large" \
        --batch_size 10 \
        --num_power_iteration 10'
    os.system(command)


def run_retrain(
    # Setup for Roberta-base.
    # seeds=(42, 9008, 0),
    # model_name_or_paths=("roberta-base",),
    # ranks=(10, 20, 50, 100, None),

    # Setup for Roberta-large.
    seeds=(42, 9008),
    model_name_or_paths=("roberta-large",),
    ranks=(10, 20, 100, None),

    run=True,
):
    # python -m classification.spectral_analysis.rebuttal_neurips_2022 --task "run_retrain"
    commands = []
    for seed in seeds:
        for model_name_or_path in model_name_or_paths:
            for rank in ranks:
                if model_name_or_path == "roberta-base":
                    output_dir = f"/mnt/data1/dump/rebuttal/roberta_prompt_retrain_{rank}_{seed}/sst-2"
                else:
                    output_dir = f"/mnt/data1/dump/rebuttal/roberta_prompt_large_retrain_{rank}_{seed}/sst-2"
                cmd = f'''python -m classification.run_wrapper \
                    --output_dir {output_dir} \
                    --task_name "sst-2" \
                    --model_name_or_path {model_name_or_path} \
                    --few_shot_type "prompt" \
                    --attention_only "yes" \
                    --static_lm_head "yes" \
                    --per_device_train_batch_size 25 \
                    --batch_size 1000 \
                    --clipping_mode "default" \
                    --num_train_epochs 4 \
                    --eval_spectrum "no" \
                    --non_private "no" \
                    --eval_steps 25 \
                    --randomly_initialize "no" \
                    --seed {seed}'''
                if rank is not None:
                    if model_name_or_path == "roberta-base":
                        cmd += (
                            f' --orthogonal_projection_path '
                            f'"/mnt/data1/dump/rebuttal/run-roberta-base/orthproj/all/global_step_000010.pt"'
                        )
                    else:
                        cmd += (
                            f' --orthogonal_projection_path '
                            f'"/mnt/data1/dump/rebuttal/run-roberta-large/orthproj/all/global_step_000004.pt"'
                        )
                    cmd += f' --orthogonal_projection_rank {rank}'
                commands.append(cmd)
    if run:
        utils.gpu_scheduler(commands=commands, excludeID=(), excludeUUID=())
    return commands


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
