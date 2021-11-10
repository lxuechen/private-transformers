"""
Run with SGD (as opposed to Adam).

python -m table2text.launchers.main_111021 --test_run False
"""

import logging
import os

import fire
from swissknife import utils
from swissknife import wrapper

from . import shared


def main(
    out_path="./table2text/scripts/main_111021.sh",
    seeds=(0, 1, 2),

    test_run=True,
    **additional_kwargs,  # Feed in things like `--skip_generation yes`.
):
    if test_run:
        logging.warning('Test run...')

        train_dir = "/nlp/scr/lxuechen/private-lm/test"
        kwargs = shared.get_best_hyper_params(
            tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=3,
            seed=0, model_name_or_path="gpt2", date="111021", gpu="3090", **additional_kwargs,
        )
        command = shared.get_command(
            **kwargs, mode=wrapper.Mode.local, logs=False, train_dir=train_dir,
        )
        os.system(f'export CUDA_VISIBLE_DEVICES=0 ; {command}')
        return
    else:
        logging.warning('Launch run...')

        base_dir = "/nlp/scr/lxuechen/private-lm/"
        commands = []
        for seed in seeds:
            # Vary learning rate.
            for learning_rate in (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4,):
                kwargs = shared.get_best_hyper_params(
                    tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=8,
                    seed=seed, model_name_or_path="gpt2", date="111021", gpu="a100", optimizer="sgd",
                    learning_rate=learning_rate,
                )
                command = shared.get_command(
                    **kwargs, evaluate_before_training="no", ghost_clipping="no",
                    base_dir=base_dir, mode=wrapper.Mode.gvm,
                )
                commands.append(command)

        utils.gpu_scheduler(commands)


if __name__ == "__main__":
    fire.Fire(main)
