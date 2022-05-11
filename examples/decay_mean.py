"""
mean estimation with decaying importance.
"""
import dataclasses
import logging
import math
import sys

import fire
import numpy as np
from swissknife import utils
import torch
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclasses.dataclass
class Data:
    beta: torch.Tensor
    Ar: torch.Tensor  # A^{1/2}.
    sensitivity: float


def make_data(
    beta=None,
    n=100000, d=10, dmin=1, mu_beta=0.2, si_beta=0.1,
    mode="linear",
    G0=1.,
):
    if beta is None:
        beta = make_beta(n=n, d=d, dmin=dmin, mu_beta=mu_beta, si_beta=si_beta)
    d = beta.size(1)

    if mode == "const":
        Ar = G0 * torch.arange(1, d + 1, device=device)
    elif mode == "sqrt":
        Ar = G0 * torch.arange(1, d + 1, device=device) ** -.2  # TODO: Not actually sqrt.
    elif mode == "linear":
        Ar = G0 * torch.arange(1, d + 1, device=device) ** -1.
    elif mode == "quadratic":
        Ar = G0 * torch.arange(1, d + 1, device=device) ** -2.
    else:
        raise ValueError(f"Unknown mode: {mode}")

    sensitivity = 2 * G0 / n

    return Data(beta=beta, Ar=Ar, sensitivity=sensitivity)


def make_beta(n, d, dmin, mu_beta, si_beta):
    if d < dmin:
        raise ValueError(f"d < dmin")
    beta = mu_beta + torch.randn(size=(n, d), device=device) * si_beta
    beta[:, dmin:] = 0.  # Ensure init distance to opt is the same.
    return beta


def evaluate(data, beta):
    # 1 / n sum_i | A^{1/2} (beta_i - beta) |
    res = (data.Ar[None, :] * (data.beta - beta))  # (n, d).
    return res.norm(2, dim=1).mean(dim=0).item()


def train_one_step(data, beta, lr, epsilon, delta, weight_decay):
    res = data.Ar[None, :] * (beta - data.beta)  # (n, d).
    grad = data.Ar * (res / res.norm(2, dim=1, keepdim=True)).mean(dim=0)

    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * data.sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta = beta - lr * (grad_priv + weight_decay * beta)
    return beta


@torch.no_grad()
def train(data, num_steps, eval_steps, lr, weight_decay, epsilon, delta, tag, verbose):
    per_step_epsilon, per_step_delta = make_per_step_privacy_spending(
        target_epsilon=epsilon, target_delta=delta, num_steps=num_steps
    )
    beta = torch.zeros(size=(1, data.beta.size(1),), device=device)
    beta_avg = beta.clone()
    for global_step in range(0, num_steps):
        if global_step % eval_steps == 0:
            mdist = evaluate(data=data, beta=beta_avg)
            if verbose:
                logging.warning(
                    f"tag: {tag}, global_step: {global_step}, lr: {lr:.6f}, num_steps: {num_steps}, mdist: {mdist:.6f}"
                )

        beta = train_one_step(
            data=data,
            beta=beta,
            lr=lr, weight_decay=weight_decay,
            epsilon=per_step_epsilon, delta=per_step_delta,
        )
        beta_avg = beta_avg * global_step / (global_step + 1) + beta / (global_step + 1)

    mdist = evaluate(data=data, beta=beta_avg)
    if verbose:
        logging.warning(f"tag: {tag}, final, lr: {lr:.6f}, num_steps: {num_steps}, mdist: {mdist:.6f}")

    return beta_avg, mdist


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
    per_step_epsilon = midval
    return per_step_epsilon, per_step_delta


def main(
    img_dir=None, eval_steps=10000, weight_decay=0, epsilon=3, delta=1e-6, seeds=(42, 96, 10000),
    verbose=False, quick=False,
):
    if quick:
        dims = (10, 50,)
        num_steps_list = (10, 20,)
        lrs = (1e-4, 3e-4,)
    else:
        dims = (10, 50, 100, 500, 1000)
        num_steps_list = (10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120)
        lrs = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3,)

    modes = ("const", "sqrt", "linear", "quadratic")
    num_modes = len(modes)
    # tuple of best results; each result is of size (len(dims), len(seeds)).
    losses = tuple([] for _ in range(num_modes))

    for dim in tqdm.tqdm(dims, desc="dims"):
        beta = make_beta(n=100000, d=dim, dmin=1, mu_beta=0.2, si_beta=0.1)
        data = tuple(make_data(beta=beta, mode=mode, G0=1.) for mode in modes)

        loss = [[sys.maxsize] for _ in range(num_modes)]  # a list (of best results over seed) for each mode.
        for idx, (this_data, this_tag) in tqdm.tqdm(enumerate(utils.zip_(data, modes)), desc="modes", total=len(data)):
            # hp tuning; 1) num_steps, 2) lr.
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
                        tag=this_tag,
                        verbose=verbose,
                    )

                    results = [train(**kwargs)[1] for seed in seeds]
                    if np.mean(results) < np.mean(loss[idx]):
                        loss[idx] = results

        # update after hp tuning.
        for this_losses, this_loss in utils.zip_(losses, loss):
            this_losses.append(this_loss)

    raw_data = dict(losses=losses, modes=modes, dims=dims)

    if img_dir is not None:
        utils.jdump(raw_data, utils.join(img_dir, 'toyplot.json'))
        img_path = utils.join(img_dir, 'toy.png')
    else:
        img_path = None

    losses = [np.array(this_losses) for this_losses in losses]  # for using np.mean, np.std.
    linestyles = ("-", "--", ":", "-.")
    markers = ("o", "+", "x", "^")
    plotting = dict(
        errorbars=tuple(
            dict(x=dims, y=np.mean(arr, axis=1), yerr=np.std(arr, axis=1), label=mode, marker=marker,
                 linestyle=linestyle)
            for arr, mode, linestyle, marker in utils.zip_(losses, modes, linestyles, markers)
        ),
        options=dict(xlabel="$d$", ylabel="$\mathbb{E}[ F(\\bar{x}) ]$")
    )
    utils.plot_wrapper(img_path=img_path, **plotting)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python decay_mean.py --img_dir "/mnt/disks/disk-2/dump/spectrum/toy4"
    fire.Fire(main)
