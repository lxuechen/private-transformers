"""
Plot 1) spectral decay, 2) retrain curves.
"""

import math

import fire
import numpy as np
import scipy.stats
import torch
from ml_swissknife import utils

from ..spectrum import density


def plot1(
    ckpt_path: str,  # Path to eigenvalues.
    dump_dir="./classification/plots",
    k=500,
    **kwargs,
):
    """Eigenvalues.

    Run on gvm.
    """
    state_dicts = torch.load(ckpt_path)
    eigenvalues = state_dicts["eigenvalues"].numpy()

    # Linear fit.
    x = np.arange(1, k + 1)
    g = np.sqrt(eigenvalues[:k])
    logg = np.log(g)
    logx = np.log(x)

    linfit = scipy.stats.linregress(logx, logg)
    g_linfit = np.exp(logx * linfit.slope + linfit.intercept)

    print("slope:", linfit.slope)
    print("R value:", linfit.rvalue)

    plots = [
        dict(x=x, y=g, marker='+', linewidth=0, label="estimated values", markersize=8, alpha=0.8),
        dict(x=x, y=g_linfit,
             label=f"linear fit: $\log y = {linfit.slope:.2f} \log x {linfit.intercept:.2f} $ ($R^2="
                   f"{linfit.rvalue ** 2.:.3f}$)"),
    ]
    utils.plot_wrapper(
        img_path=utils.join(dump_dir, "eigenvalue-linfit"),
        suffixes=(".png", ".pdf"),
        plots=plots,
        options=dict(xlabel="$k$", ylabel="$\lambda(H^\\top H)^{1/2}$", xscale='log', yscale='log')
    )

    # Spectral density.
    sigma_squared = 1e-6
    evals = np.sqrt(eigenvalues[None, :k])
    den, gri = density.eigv_to_density(evals, sigma_squared=sigma_squared, grid_len=300000, grid_expand=3e-4)
    utils.plot_wrapper(
        img_path=utils.join(dump_dir, 'eigenvalue-density'),
        suffixes=(".png", ".pdf"),
        plots=[dict(x=gri, y=den, label=f"bandwidth $\sigma={math.sqrt(sigma_squared):.5f}$")],
        options=dict(xlabel="$\lambda(H^\\top H)^{1/2}$", ylabel="Density of KDE",
                     ylim=dict(bottom=1e-10, top=2e2),
                     xscale="log", yscale='log')
    )


def plot2(
    base_dir,
    seeds=(42, 9008, 0),
    ranks=(10, 20, 100, None),
    dump_dir="./classification/plots",
    markers=('x', '^', '+', 'o'),
    **kwargs,
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


def plot_all(**kwargs):
    # rebuttal roberta-base experiments.
    # python -m classification.runs.rebuttal_plots_neurips_2022 --task "plot_all" --base_dir "/home/t-lc/dump/privlm/rebuttal" --ckpt_path "/home/t-lc/dump/privlm/rebuttal/run-roberta-base/orthproj/eigenvalues/global_step_000010.evals"
    plot1(**kwargs)
    plot2(**kwargs)


def main(task="plot_all", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("plot_all", "plot1", "plot2"),
        task_callables=(plot_all, plot1, plot2),
        **kwargs,
    )


if __name__ == "__main__":
    fire.Fire(main)
