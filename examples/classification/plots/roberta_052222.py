import math

import fire
import numpy as np
import scipy.stats
from swissknife import utils
import torch

from ..spectrum import density


def plot_helper(
    dump_dir="./classification/plot",
    ckpt_path=f"/Users/xuechenli/Desktop/dump_a100/privlm2/roberta_prompt/sst-2/orthproj_9009/eigenvalues"
              f"/global_step_{50:06d}.evals",
    k=500,
    batch_size=1000,
):
    state_dicts = torch.load(ckpt_path)
    eigenvalues = state_dicts["eigenvalues"]
    eigenvalues /= batch_size
    eigenvalues = eigenvalues.numpy()

    # Linear fit.
    x = np.arange(1, k + 1)
    g = np.sqrt(eigenvalues[:k])
    logg = np.log(g)
    logx = np.log(x)

    linfit = scipy.stats.linregress(logx, logg)
    g_linfit = np.exp(logx * linfit.slope + linfit.intercept)

    print("slope:", linfit.slope)
    print("R value:", linfit.rvalue)

    linfit_sign = "+" if linfit.intercept > 0 else ''
    plots = [
        dict(x=x, y=g, marker='+', linewidth=0, label="estimated values", markersize=8, alpha=0.8),
        dict(x=x, y=g_linfit,
             label=f"linear fit: $\log y = {linfit.slope:.2f} \log x {linfit_sign}{linfit.intercept:.2f} $ ($R^2="
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


def make_suppl_plots():
    base_dir = "/Users/xuechenli/Desktop/dump_a100/privlm2/roberta_prompt/sst-2"

    # Less samples.
    ckpt_path = f"{base_dir}/orthproj_42_300/eigenvalues/global_step_{10:06d}.evals"
    dump_dir = f"./classification/plots/less_samples_trim_front"
    plot_helper(dump_dir=dump_dir, ckpt_path=ckpt_path)

    # More samples.
    ckpt_path = f"{base_dir}/orthproj_9009/eigenvalues/global_step_{10:06d}.evals"
    dump_dir = f"./classification/plots/npi_{10:06d}"
    plot_helper(dump_dir=dump_dir, ckpt_path=ckpt_path)

    ckpt_path = f"{base_dir}/orthproj_9009/eigenvalues/global_step_{50:06d}.evals"
    dump_dir = f"./classification/plots/npi_{50:06d}"
    plot_helper(dump_dir=dump_dir, ckpt_path=ckpt_path)

    ckpt_path = f"{base_dir}/orthproj_9009/eigenvalues/global_step_{100:06d}.evals"
    dump_dir = f"./classification/plots/npi_{100:06d}"
    plot_helper(dump_dir=dump_dir, ckpt_path=ckpt_path)


# python -m classification.plots.roberta_052222
def main(task="make_suppl_plots", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("make_suppl_plots",),
        task_callables=(make_suppl_plots,),
        **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
