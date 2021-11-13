"""
Run with various epsilon and delta.

python -m table2text.launchers.main_111221 --test_run False
"""

import logging
import os

import fire
from swissknife import wrapper, utils

from . import shared


def main(
    seeds=(0, 1, 2, 3, 4),

    test_run=True,
    **additional_kwargs,  # Feed in things like `--skip_generation yes`.
):
    if test_run:
        logging.warning('Test run...')

        train_dir = "/nlp/scr/lxuechen/private-lm/test"
        kwargs = shared.get_best_hyper_params(
            tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=3,
            seed=0, model_name_or_path="gpt2", date="111221",
        )
        command = shared.get_command(
            **kwargs, mode=wrapper.Mode.local, logs=False, train_dir=train_dir,
            **additional_kwargs
        )
        os.system(command)
        return
    else:
        logging.warning('Launch run...')

        base_dir = "/nlp/scr/lxuechen/private-lm/"
        commands = []
        for seed in seeds:
            # Vary delta.
            for target_epsilon in (3, 8):
                for target_delta in (1e-5, 1e-6, 1e-7):
                    kwargs = shared.get_best_hyper_params(
                        tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=target_epsilon,
                        seed=seed, model_name_or_path="gpt2", date="111221", target_delta=target_delta,
                        gpu="a100",
                    )
                    command = shared.get_command(
                        **kwargs, evaluate_before_training="no", ghost_clipping="no",
                        base_dir=base_dir, mode=wrapper.Mode.gvm,
                    )
                    commands.append(command)

        utils.gpu_scheduler(commands)


if __name__ == "__main__":
    fire.Fire(main)
