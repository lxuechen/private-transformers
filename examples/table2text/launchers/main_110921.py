"""
Run with various epsilon and delta.

python -m table2text.launchers.main_110921
"""

import os

import fire
from swissknife import wrapper

from . import shared


def main(
    base_dir="/nlp/scr/lxuechen/private-lm/main_110921",
    out_path="./table2text/scripts/main_110921.sh",
    seeds=(0,),

    test_run=True,
    **additional_kwargs,  # Feed in things like `--skip_generation yes`.
):
    if test_run:
        kwargs = shared.get_best_hyper_params(
            tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=3,
            seed=0, model_name_or_path="distilgpt2", date="110921",
        )
        command = shared.get_command(
            **kwargs, base_dir=base_dir, mode=wrapper.Mode.local, logs=False,
            *additional_kwargs
        )
        os.system(command)
        return

    commands = []
    for seed in seeds:
        # Vary epsilon.
        for target_epsilon in (0.1, 0.5, 1, 2, 3, 5, 8):
            kwargs = shared.get_best_hyper_params(
                tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=target_epsilon,
                seed=seed, model_name_or_path="gpt2", date="110921",
            )
            command = shared.get_command(
                **kwargs,
                base_dir=base_dir, mode=wrapper.Mode.submit, hold_job=False, priority="standard"
            )
            commands.append(command)

        # Vary delta.
        for target_epsilon in (3, 8):
            for target_delta in (1e-5, 1e-6, 1e-7):
                kwargs = shared.get_best_hyper_params(
                    tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=target_epsilon,
                    seed=seed, model_name_or_path="gpt2", date="110921", target_delta=target_delta,
                )
                command = shared.get_command(
                    **kwargs,
                    base_dir=base_dir, mode=wrapper.Mode.submit, hold_job=False, priority="standard"
                )
                commands.append(command)

    with open(out_path, 'w') as f:
        f.write('\n'.join(commands))


if __name__ == "__main__":
    fire.Fire(main)
