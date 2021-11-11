"""
Run with various epsilon and delta.

python -m table2text.launchers.main_110921_v5 --test_run False
bash table2text/scripts/main_110921_v5.sh

Patching failed jobs (vary delta) due to memory.
"""

import logging
import os

import fire
from swissknife import wrapper

from . import shared


def main(
    out_path="./table2text/scripts/main_110921_v5.sh",
    seeds=(0, 1, 2),

    test_run=True,
    **additional_kwargs,  # Feed in things like `--skip_generation yes`.
):
    if test_run:
        logging.warning('Test run...')

        train_dir = "/nlp/scr/lxuechen/private-lm/test"
        kwargs = shared.get_best_hyper_params(
            tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=3,
            seed=0, model_name_or_path="gpt2", date="110921", gpu="3090", **additional_kwargs,
        )
        command = shared.get_command(
            **kwargs, mode=wrapper.Mode.local, logs=False, train_dir=train_dir,
        )
        os.system(command)
        return
    else:
        logging.warning('Launch run...')

        base_dir = "/nlp/scr/lxuechen/private-lm/"
        commands = []
        for seed in seeds:
            # Vary epsilon.
            for target_epsilon in (0.1,):
                kwargs = shared.get_best_hyper_params(
                    tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=target_epsilon,
                    seed=seed, model_name_or_path="gpt2", date="110921", gpu="3090",
                )
                kwargs.pop("optimizer")
                command = shared.get_command(
                    **kwargs, evaluate_before_training="no", ghost_clipping="no",
                    base_dir=base_dir, mode=wrapper.Mode.submit, hold_job=False, priority="high", gpu="3090"
                )
                commands.append(command)

        with open(out_path, 'w') as f:
            f.write('\n'.join(commands))


if __name__ == "__main__":
    fire.Fire(main)
