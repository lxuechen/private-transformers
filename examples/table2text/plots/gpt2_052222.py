import math

import fire
import numpy as np
import scipy.stats
from swissknife import utils
import torch

from .. import density


def plot_spectrum(
    dump_dir="./table2text/plots",
    ckpt_path=f"/Users/xuechenli/Desktop/dump_a100/privlm2/gpt2/e2e/orthproj_42/eigenvalues/global_step_{10:06d}.evals",
    k=500,
    batch_size=1024,  # What's actually used in training.
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
             label=f"linear fit: $\log y = {linfit.slope:.4f} \log x {linfit_sign}{linfit.intercept:.2f} $ ($R^2="
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


def plot_retrain():
    base_dir = "/Users/xuechenli/Desktop/dump_a100/privlm2"

    errorbars = []
    for rank, marker in utils.zip_((10, 50, 100, 500, None), ('x', '+', 'o', 'v', '^')):
        x = []
        y = []
        for seed in (42, 9008, 0):
            path = utils.join(base_dir, f"gpt2_retrain_{rank}_{seed}", "e2e", "log_history.json")
            record = utils.jload(path)
            x.append([recordi['epoch'] for recordi in record])
            y.append([recordi['eval']['model']['tok_logprobs'] for recordi in record])
        x = np.array(x)
        y = np.array(y)

        label = "original" if rank is None else f"{rank}"
        errorbars.append(
            dict(x=np.mean(x, axis=0), y=np.mean(y, axis=0), yerr=np.std(y, axis=0), label=label, marker=marker)
        )

    img_path = "./table2text/plots/retrain"
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(xlabel="epoch", ylabel="Per-token NLL"),
    )


def make_suppl_plots():
    plot_spectrum()
    plot_retrain()


# TODO: Merge plotter.
# python -m table2text.plots.gpt2_052222
def main(task="make_suppl_plots", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("make_suppl_plots",),
        task_callables=(make_suppl_plots,),
        **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
