"""
Plot 1) spectral decay, 2) retrain curves.
"""

import fire
import numpy as np
import scipy.stats
from swissknife import utils
import torch


# python -m classification.plot.roberta_051622 --task plot1
def plot1(
    dump_dir="./classification/plot",
    ckpt_path=f"/Users/xuechenli/Desktop/dump_a100/privlm/roberta_prompt/sst-2/eigenvalues.pt-small",
    k=500,
):
    """Eigenvalues.

    Run on gvm.
    """
    state_dicts = torch.load(ckpt_path)
    eigenvalues = state_dicts["eigenvalues"]

    x = np.arange(1, k + 1)
    g = np.sqrt(eigenvalues[:k])
    logg = np.log(g)
    logx = np.log(x)

    linfit = scipy.stats.linregress(logx, logg)
    g_linfit = np.exp(logx * linfit.slope + linfit.intercept)

    print("slope:", linfit.slope)
    print("R value:", linfit.rvalue)

    plots = [
        dict(x=x, y=g, marker='+', linewidth=0),
        dict(x=x, y=g_linfit,
             label=f"linear fit: $\log y = {linfit.slope:.2f} \log x {linfit.intercept:.2f} $ ($R^2={linfit.rvalue ** 2.:.3f}$)")
    ]
    utils.plot_wrapper(
        img_path=utils.join(dump_dir, "eigenvalue-linfit"),
        suffixes=(".png", ".pdf"),
        plots=plots,
        options=dict(xlabel="$k$", ylabel="$\lambda(H^\\top H)^{1/2}$", xscale='log', yscale='log')
    )


# python -m classification.plot.roberta_051622 --task plot2
def plot2(
    seeds=(42, 101, 20598, 90828, 9008),
    ranks=(10, 20, 100, None),
    base_dir="/Users/xuechenli/Desktop/dump_a100/privlm",
    dump_dir="./classification/plot",
    markers=('x', '^', '+', 'o'),
):
    """Retrain.

    Run locally.
    """
    errorbars = []
    for rank, marker in utils.zip_(ranks, markers):
        results = []
        for seed in seeds:
            output_dir = utils.join(
                f"{base_dir}/roberta_prompt_retrain_{rank}_{seed}/sst-2",
                'log_history.json'
            )
            record = utils.jload(output_dir)
            results.append([dumpi['dev']['eval_acc'] for dumpi in record])
            steps = [dumpi['step'] for dumpi in record]

        label = f"subspace rank={rank}" if rank is not None else "original"
        mu, si = utils.average_over_seed(results)
        errorbar = dict(x=steps, y=mu, yerr=si, label=label, marker=marker)
        errorbars.append(errorbar)

    img_path = utils.join(dump_dir, 'plot2')
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(xlabel="iteration", ylabel="SST-2 classification accuracy (dev)")
    )


# python -m classification.plot.roberta_051622 --task plot_all
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
