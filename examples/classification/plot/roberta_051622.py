"""
Plot 1) spectral decay, 2) retrain curves.

python -m classification.plot.roberta_051622
"""

import fire

from swissknife import utils


def plot1():
    """Eigenvalues."""
    pass


def plot2(
    seeds=(42,),
    ranks=(10, 20, 50, 100, None),
    base_dir="/Users/xuechenli/Desktop/dump_a100/privlm",
    dump_dir="./classification/plot",
):
    """Retrain."""
    errorbars = []
    for rank in ranks:
        results = []
        for seed in seeds:
            output_dir = utils.join(
                f"{base_dir}/roberta_retrain_{rank}_{seed}/sst-2",
                'log_history.json'
            )
            record = utils.jload(output_dir)
            results.append([dumpi['dev']['eval_acc'] for dumpi in record])
            if rank == 100:
                print(results)
            steps = [dumpi['step'] for dumpi in record]

        label = f"subspace rank={rank}" if rank is not None else "original"
        mu, si = utils.average_over_seed(results)
        errorbar = dict(x=steps, y=mu, yerr=si, label=label, marker='x')
        errorbars.append(errorbar)

    img_path = utils.join(dump_dir, 'plot2')
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(xlabel="step", ylabel="SST-2 classification accuracy (dev)")
    )


def plot_all():
    plot1()
    plot2()


def main(task="plot2"):
    utils.runs_tasks(
        task=task,
        task_names=("plot_all", "plot1", "plot2"),
        task_callables=(plot_all, plot1, plot2)
    )


if __name__ == "__main__":
    fire.Fire(main)
