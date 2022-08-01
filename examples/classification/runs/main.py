import fire
from ml_swissknife import utils


def run_pca():
    # python -m classification.runs.main --task "run_pca"
    commands = []
    command = 'python -m classification.numerical \
        --task "pca" \
        --n 2000 \
        --k 500 \
        --batch_size 40 \
        --chunk_size 50 \
        --train_dir "/home/t-lc/dump/privlm/rebuttal/run-roberta-base" \
        --num_power_iteration 10'
    commands.append(command)

    command = 'python -m classification.numerical \
        --task "pca" \
        --n 2000 \
        --k 500 \
        --train_dir "/home/t-lc/dump/privlm/rebuttal/run-roberta-large" \
        --batch_size 20 \
        --chunk_size 25 \
        --num_power_iteration 10'
    commands.append(command)

    utils.gpu_scheduler(commands, excludeID=(0,), log=False)


def run_retrain(seeds=(42, 9008, 0), model_name_or_paths=("roberta-base",), run=True):
    # python -m classification.runs.main --task "run_retrain"
    commands = []
    for seed in seeds:
        for model_name_or_path in model_name_or_paths:
            for rank in (10, 20, 50, 100, None):
                output_dir = f"/home/t-lc/dump/privlm/rebuttal/roberta_prompt_retrain_{rank}_{seed}/sst-2"
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
                    cmd += (
                        f' --orthogonal_projection_path '
                        f'"/home/t-lc/dump/privlm/rebuttal/run-roberta-base/orthproj/all/global_step_000010.pt"'
                    )
                    cmd += f' --orthogonal_projection_rank {rank}'
                commands.append(cmd)
    if run:
        utils.gpu_scheduler(commands=commands, excludeID=(0,), excludeUUID=(0,))
    return commands


def main(task):
    utils.runs_tasks(task=task, task_names=("run_pca", "run_retrain"), task_callables=(run_pca, run_retrain))


if __name__ == "__main__":
    fire.Fire(main)
