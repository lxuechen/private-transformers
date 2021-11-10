"""

python -m table2text.launchers.main_110921
"""

import os

import fire
from swissknife import wrapper

from . import shared


def main():
    kwargs = shared.get_best_hyper_params(
        tuning_mode="full", task_mode="e2e", non_private="no", target_epsilon=3,
        seed=0, model_name_or_path="gpt2", date="110921",
    )
    command = shared.get_command(**kwargs, mode=wrapper.Mode.local, logs=False)
    print(command)
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
