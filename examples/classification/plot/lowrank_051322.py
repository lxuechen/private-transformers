"""
"""

import fire
from swissknife import utils


def main(
    seeds=(0, 1, 2),
    ranks=(1, 10, 100, None),
    base_dir="/Users/xuechenli/Desktop/dump_a100/classification"
):
    errorbars = []
    for rank in ranks:
        results = []
        for seed in seeds:
            path = utils.join(base_dir, f'lowrank_rank_{rank}_seed_{seed}', 'log_history.json')
            record = utils.jload(path)
            results.append([dumpi['dev']['eval_acc'] for dumpi in record])
            steps = [dumpi['step'] for dumpi in record]

        label = f"subspace rank={rank}" if rank is not None else "original"
        mu, si = utils.average_over_seed(results)
        errorbar = dict(x=steps, y=mu, yerr=si, label=label, marker='x')
        errorbars.append(errorbar)

    utils.plot_wrapper(
        errorbars=errorbars,
        options=dict(xlabel="step", ylabel="SST-2 classification accuracy (dev)")
    )


if __name__ == "__main__":
    fire.Fire(main)
