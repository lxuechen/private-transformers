"""Save the gradients obtained along the trajectory.

python -m classification.runs.save_grads
"""

from ml_swissknife import utils


def _get_cmd(model_name_or_path: str, num_train_epochs=60):
    base_dir = "/home/t-lc/dump/privlm"
    output_dir = utils.join(base_dir, 'rebuttal', f'run-{model_name_or_path}')
    return f'''python -m classification.run_wrapper \
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


if __name__ == "__main__":
    cmds = []
    for model_name_or_path in ("roberta-base", "roberta-large"):
        cmds.append(_get_cmd(model_name_or_path=model_name_or_path))
    utils.gpu_scheduler(cmds, excludeID=(0,), excludeUUID=(0,))
