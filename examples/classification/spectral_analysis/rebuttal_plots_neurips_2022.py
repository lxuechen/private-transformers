# Copyright (c) Xuechen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Plot 1) spectral decay, 2) retrain curves.
"""

import math

import fire
import numpy as np
import scipy.stats
import torch
from ml_swissknife import utils

from . import density


def plot1(
    ckpt_path: str,  # Path to eigenvalues.
    dump_dir="./classification/plots",
    img_name="",
    k=500,
    **kwargs,
):
    """Eigenvalues.

    Run on gvm.
    """
    # Roberta-large
    # python -m classification.spectral_analysis.rebuttal_plots_neurips_2022 --task "plot1" --ckpt_path "/mnt/data1/dump/rebuttal/run-roberta-large/orthproj/eigenvalues/global_step_000005.evals" --img_name "large" --k 100
    if img_name != "":
        img_name = f'-{img_name}'

    state_dicts = torch.load(ckpt_path)
    eigenvalues = state_dicts["eigenvalues"].numpy()
    eigenvalues = -np.sort(-eigenvalues)
    k = min(k, len(eigenvalues))

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
        img_path=utils.join(dump_dir, f"eigenvalue-linfit{img_name}"),
        suffixes=(".png", ".pdf"),
        plots=plots,
        options=dict(xlabel="$k$", ylabel="$\lambda(H^\\top H)^{1/2}$", xscale='log', yscale='log')
    )

    # Spectral density.
    sigma_squared = 1e-6
    evals = np.sqrt(eigenvalues[None, :k])
    den, gri = density.eigv_to_density(evals, sigma_squared=sigma_squared, grid_len=300000, grid_expand=3e-4)
    utils.plot_wrapper(
        img_path=utils.join(dump_dir, f'eigenvalue-density{img_name}'),
        suffixes=(".png", ".pdf"),
        plots=[dict(x=gri, y=den, label=f"bandwidth $\sigma={math.sqrt(sigma_squared):.5f}$")],
        options=dict(xlabel="$\lambda(H^\\top H)^{1/2}$", ylabel="Density of KDE",
                     ylim=dict(bottom=1e-10, top=2e2),
                     xscale="log", yscale='log')
    )


def plot2(
    base_dir: str,
    img_name="",
    seeds=(42, 9008, 0),
    ranks=(10, 20, 100, None),
    dump_dir="./classification/plots",
    markers=('x', '^', '+', 'o'),
    roberta_large=False,
    **kwargs,
):
    """Retrain.

    Run locally.
    """
    # Roberta-large
    # python -m classification.spectral_analysis.rebuttal_plots_neurips_2022 --task "plot2" --img_name "large" --base_dir "/mnt/data1/dump/rebuttal" --roberta_large True
    if img_name != "":
        img_name = f'-{img_name}'

    errorbars = []
    for rank, marker in utils.zip_(ranks, markers):
        results = []
        for seed in seeds:
            if roberta_large:
                output_dir = utils.join(
                    f"{base_dir}/roberta_prompt_large_retrain_{rank}_{seed}/sst-2",
                    'log_history.json'
                )
            else:
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

    img_path = utils.join(dump_dir, f'plot2{img_name}')
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(xlabel="iteration", ylabel="SST-2 classification accuracy (dev)")
    )


def plot_all(**kwargs):
    # rebuttal roberta-base experiments.
    # python -m classification.spectral_analysis.rebuttal_plots_neurips_2022 --task "plot_all" --base_dir "/mnt/data1/dump/rebuttal" --ckpt_path "/mnt/data1/dump/rebuttal/run-roberta-base/orthproj/eigenvalues/global_step_000010.evals"
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
