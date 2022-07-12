"""
Toy example on geometric median estimation in the paper.

CUDA_VISIBLE_DEVICES=3 python geometric_median.py --img_dir "/mnt/disks/disk-2/dump/spectrum/geometric_median"
"""
import dataclasses
import logging
import math
import sys
from typing import Tuple

import fire
import numpy as np
import torch
import tqdm
from ml_swissknife import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclasses.dataclass
class Data:
    beta_train: torch.Tensor
    beta_test: torch.Tensor
    Ar: torch.Tensor  # A^{1/2}.
    sensitivity: float

    def __post_init__(self):
        self.n_train, self.d = self.beta_train.size()
        self.n_test = self.beta_test.shape[0]


class Modes(metaclass=utils.ContainerMeta):
    const = "const"
    quarter = "quarter"
    sqrt = "sqrt"
    linear = "linear"
    quadratic = "quadratic"


def make_data(
    betas=None,
    n_train=100000, n_test=100000, d=10, dmin=1, mu_beta=0.2, si_beta=0.1,
    mode="linear",
    g0=1.,
):
    if betas is None:
        beta_train, beta_test = make_beta(
            n_train=n_train, n_test=n_test, d=d, dmin=dmin, mu_beta=mu_beta, si_beta=si_beta
        )
    else:
        beta_train, beta_test = betas
        n_train, d = beta_train.size()
        n_test, _ = beta_test.size()

    if mode == Modes.const:
        Ar = g0 * torch.arange(1, d + 1, device=device)
    elif mode == Modes.quarter:
        Ar = g0 * torch.arange(1, d + 1, device=device) ** -.25
    elif mode == Modes.sqrt:
        Ar = g0 * torch.arange(1, d + 1, device=device) ** -.5
    elif mode == Modes.linear:
        Ar = g0 * torch.arange(1, d + 1, device=device) ** -1.
    elif mode == Modes.quadratic:
        Ar = g0 * torch.arange(1, d + 1, device=device) ** -2.
    else:
        raise ValueError(f"Unknown mode: {mode}")

    sensitivity = 2 * g0 / n_train

    return Data(beta_train=beta_train, beta_test=beta_test, Ar=Ar, sensitivity=sensitivity)


def make_beta(n_train, n_test, d, dmin, mu_beta, si_beta):
    if d < dmin:
        raise ValueError(f"d < dmin")

    beta_train = mu_beta + torch.randn(size=(n_train, d), device=device) * si_beta
    beta_train[:, dmin:] = 0.  # Ensure init distance to opt is the same.

    beta_test = mu_beta + torch.randn(size=(n_test, d), device=device) * si_beta
    beta_test[:, dmin:] = 0.  # Same distribution as train.

    return beta_train, beta_test


def evaluate(data: Data, beta: torch.Tensor) -> Tuple:
    """Compute loss 1 / n sum_i | A^{1/2} (beta - beta_i) |_2 for train and test."""

    def compute_loss(samples):
        res = data.Ar[None, :] * (beta - samples)  # (n, d).
        return res.norm(2, dim=1).mean(dim=0).item()

    return tuple(
        compute_loss(samples=samples)
        for samples in (data.beta_train, data.beta_test)
    )


def train_one_step(data: Data, beta, lr, epsilon, delta, weight_decay):
    res = data.Ar[None, :] * (beta - data.beta_train)  # (n, d).
    grad = data.Ar * (res / res.norm(2, dim=1, keepdim=True)).mean(dim=0)

    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * data.sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta = beta - lr * (grad_priv + weight_decay * beta)
    return beta


@torch.no_grad()
def train(data: Data, num_steps, eval_steps, lr, weight_decay, epsilon, delta, tag, verbose, seed):
    utils.manual_seed(seed)

    per_step_epsilon, per_step_delta = make_per_step_privacy_spending(
        target_epsilon=epsilon, target_delta=delta, num_steps=num_steps
    )

    beta = torch.zeros(size=(1, data.d,), device=device)
    beta_avg = beta.clone()

    for global_step in range(0, num_steps):
        if global_step % eval_steps == 0:
            tr_loss, te_loss = evaluate(data=data, beta=beta_avg)
            if verbose:
                logging.warning(
                    f"tag: {tag}, global_step: {global_step}, lr: {lr:.6f}, num_steps: {num_steps}, "
                    f"train_loss: {tr_loss:.6f}, test_loss: {te_loss:.6f}"
                )

        beta = train_one_step(
            data=data,
            beta=beta,
            lr=lr, weight_decay=weight_decay,
            epsilon=per_step_epsilon, delta=per_step_delta,
        )
        beta_avg = beta_avg * global_step / (global_step + 1) + beta / (global_step + 1)

    final_tr_loss, final_te_loss = evaluate(data=data, beta=beta_avg)
    if verbose:
        logging.warning(
            f"tag: {tag}, final, lr: {lr:.6f}, num_steps: {num_steps}, "
            f"train_loss: {final_tr_loss:.6f}, te_loss: {final_te_loss:.6f}"
        )

    return beta_avg, (final_tr_loss, final_te_loss)


def make_per_step_privacy_spending(
    target_epsilon, target_delta, num_steps, threshold=1e-4,
):
    per_step_delta = target_delta / (num_steps + 1)

    def adv_composition(per_step_epsilon):
        total_epsilon = (
            math.sqrt(2 * num_steps * math.log(1 / per_step_delta)) * per_step_epsilon +
            num_steps * per_step_epsilon * (math.exp(per_step_epsilon) - 1)
        )
        return total_epsilon

    minval, maxval = 1e-6, 5
    while maxval - minval > threshold:
        midval = (maxval + minval) / 2
        eps = adv_composition(midval)
        if eps > target_epsilon:
            maxval = midval
        else:
            minval = midval
    per_step_epsilon = minval
    return per_step_epsilon, per_step_delta


def main(
    img_dir=None, eval_steps=10000, weight_decay=0, epsilon=2, delta=1e-6,
    n_train=10000, n_test=10000, dmin=1, mu_beta=1., si_beta=1, g0=3.,
    seeds=(42, 96, 10000, 999, 101),  # Some arbitrary numbers.
    modes=(Modes.const, Modes.sqrt, Modes.linear),  # A subset of all possible modes for visualization.
    verbose=False,
    quick=False,  # Use small data if True.
):
    if quick:
        dims = (10, 50,)
        num_steps_list = (10, 20,)
        lrs = (1e-4, 3e-4,)
    else:
        dims = (20, 50, 100, 200, 500, 1000, 2000)
        num_steps_list = (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120)
        lrs = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3,)

    tr_losses = {mode: [] for mode in modes}
    te_losses = {mode: [] for mode in modes}
    for dim in tqdm.tqdm(dims, desc="dims"):
        betas = make_beta(n_train=n_train, n_test=n_test, d=dim, dmin=dmin, mu_beta=mu_beta, si_beta=si_beta)
        data = tuple(make_data(betas=betas, mode=mode, g0=g0) for mode in modes)

        tr_loss = {mode: [sys.maxsize] for mode in modes}
        te_loss = {mode: [sys.maxsize] for mode in modes}
        for this_data, this_mode in tqdm.tqdm(utils.zip_(data, modes), desc="modes", total=len(data)):

            # Hyperparameter tuning.
            for num_steps in num_steps_list:
                for lr in lrs:
                    kwargs = dict(
                        data=this_data,
                        num_steps=num_steps,
                        lr=lr,

                        eval_steps=eval_steps,
                        weight_decay=weight_decay,
                        epsilon=epsilon,
                        delta=delta,
                        tag=this_mode,
                        verbose=verbose,
                    )

                    tr_results = []
                    te_results = []
                    for seed in seeds:
                        _, (a, b) = train(**kwargs, seed=seed)
                        tr_results.append(a)
                        te_results.append(b)

                    if np.mean(tr_results) < np.mean(tr_loss[this_mode]):
                        tr_loss[this_mode] = tr_results
                        te_loss[this_mode] = te_results

        # update after hp tuning.
        for this_mode in modes:
            tr_losses[this_mode].append(tr_loss[this_mode])
            te_losses[this_mode].append(te_loss[this_mode])

    raw_data = dict(tr_losses=tr_losses, te_losses=te_losses, modes=modes, dims=dims)

    if img_dir is not None:
        utils.jdump(raw_data, utils.join(img_dir, 'toyplot.json'))

        plot_modes = modes
        linestyles = ("-", "--", ":", "-.")
        markers = ("o", "+", "x", "^")

        tr_plotting = dict(
            errorbars=tuple(
                dict(
                    x=dims,
                    y=np.mean(np.array(tr_losses[this_mode]), axis=1),
                    yerr=np.std(np.array(tr_losses[this_mode]), axis=1),
                    label=this_mode, marker=markers[mode_idx],
                    linestyle=linestyles[mode_idx]
                )
                for mode_idx, this_mode in enumerate(plot_modes)
            ),
            options=dict(xlabel="$d$", ylabel="train loss")
        )
        utils.plot_wrapper(
            img_path=utils.join(img_dir, 'trplot'),
            suffixes=('.png', '.pdf'),
            **tr_plotting,
        )

        te_plotting = dict(
            errorbars=tuple(
                dict(
                    x=dims,
                    y=np.mean(np.array(te_losses[this_mode]), axis=1),
                    yerr=np.std(np.array(te_losses[this_mode]), axis=1),
                    label=this_mode, marker=markers[mode_idx],
                    linestyle=linestyles[mode_idx]
                )
                for mode_idx, this_mode in enumerate(plot_modes)
            ),
            options=dict(xlabel="$d$", ylabel="test loss")
        )
        utils.plot_wrapper(
            img_path=utils.join(img_dir, 'teplot'),
            suffixes=('.png', '.pdf'),
            **te_plotting,
        )


if __name__ == "__main__":
    fire.Fire(main)
