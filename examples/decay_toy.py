"""
Toy example of spectral decay.
"""
import logging
import math

import fire
import numpy as np
from swissknife import utils
import torch
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
import dataclasses


@dataclasses.dataclass
class Data:
    x: torch.Tensor
    y: torch.Tensor
    sensitivity: float
    beta_opt: torch.Tensor

    R: float
    C: float
    G0: float

    mu_x: torch.Tensor
    si_x: torch.Tensor


def make_data(mode="decay", n=100000, d=50):
    R = 1
    C = 1
    sum_sqrt = torch.sum(torch.arange(1, d + 1) ** -.5)  # sum_j 1 / sqrt(j)
    G0 = 1 / 2 * C ** 2 / d

    # beta_* = (0.25, 0.25, ..., 0.25) / sqrt(d).
    # |beta_*|_2 \le 0.5
    # |beta|_2 \le 0.5
    beta_opt = torch.full(
        fill_value=0.25 / math.sqrt(d), size=(d,), device=device
    )
    mu_x = torch.zeros(size=(d,), device=device)
    # decay variance (G0, G0 * 2 ** -0.5, ..., G0 * d ** -0.5) =>
    #   |x|_2^2 concentrates to G0 * sum_sqrt; smaller than C^2, so almost no clipping.
    # si_x_decay = math.sqrt(G0) * torch.sqrt(torch.arange(1, d + 1, device=device) ** -.5)  # standard deviation.
    # TODO: This is a weird spectrum.
    si_x_decay = math.sqrt(G0 * 2) * torch.sqrt(torch.arange(1, d + 1, device=device) ** -1.)  # standard deviation.
    # constant variance (G0, G0, ..., G0).
    si_x_const = math.sqrt(G0) * torch.ones(size=(d,), device=device)
    sensitivity = 2 / n * C ** 2 * R

    if mode == "decay":
        si_x = si_x_decay
    elif mode == "const":
        si_x = si_x_const
    else:
        raise ValueError

    x = mu_x[None, :] + si_x[None, :] * torch.randn(size=(n, d), device=device)
    num_clipped = (x.norm(2, dim=1) > C).sum(dim=0)
    logging.warning(f"Create data in mode: {mode}; number of examples clipped: {num_clipped}")

    x = x * torch.clamp_max(C / x.norm(2, dim=1, keepdim=True), max=1.)  # Almost no clipping happening here.
    y = x @ beta_opt  # no noise.

    return Data(
        x=x, y=y,
        sensitivity=sensitivity, beta_opt=beta_opt,
        R=R, C=C, G0=G0,
        mu_x=mu_x, si_x=si_x,
    )


def train_one_step(data, beta, lr, epsilon, delta, weight_decay):
    residuals = (data.x @ beta - data.y)
    grad = (residuals[:, None] * data.x).mean(dim=0)  # avg gradient.
    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * data.sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta -= lr * (grad_priv + weight_decay * beta)
    beta = beta * torch.clamp_max(.5 * data.R / beta.norm(2), max=1.)  # projection of beta into the radius-R ball.
    return beta


def evaluate(data, beta):
    ypred = data.x @ beta
    mse = .5 * ((data.y - ypred) ** 2).mean(dim=0)
    dis = (beta - data.beta_opt).norm(2)
    return mse.item(), dis.item()


@torch.no_grad()
def train(data, num_steps, eval_steps, lr, epsilon, delta, weight_decay):
    beta = torch.zeros(size=(data.x.size(1),), device=device)
    beta_avg = beta.clone()
    for global_step in range(0, num_steps):
        if global_step % eval_steps == 0:
            mse, dis = evaluate(data=data, beta=beta_avg)
            logging.warning(f"global_step: {global_step}, mse: {mse:.6f}, iterate dist: {dis:.6f}")
        beta = train_one_step(
            data=data, beta=beta, lr=lr, epsilon=epsilon, delta=delta, weight_decay=weight_decay
        )
        beta_avg = beta_avg * global_step / (global_step + 1) + beta / (global_step + 1)
    mse, dis = evaluate(data=data, beta=beta_avg)
    logging.warning(f"final, mse: {mse:.6f}, iterate dist: {dis:.6f}")
    return beta_avg, mse, dis


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
    img_dir=None,
    eval_steps=10000, weight_decay=1e-7,
    epsilon=3, delta=1e-6,
):
    dims = (2, 5, 10, 20, 50, 100,)
    num_steps_list = (100, 400, 700, 1000, 1300, 1600, 1900, 2200, 3000, 5000,)
    lrs = (1e-1, 2e-1, 5e-1, 1, 2, 5,)
    seeds = (42, 96, 10000)

    losses_decay = []
    losses_const = []
    for dim in tqdm.tqdm(dims, desc="dims"):
        data_decay = make_data(mode='decay', d=dim)
        data_const = make_data(mode="const", d=dim)

        loss_decay = utils.MinMeter()
        loss_const = utils.MinMeter()
        for num_steps in tqdm.tqdm(num_steps_list, desc="num steps"):
            per_step_epsilon, per_step_delta = make_per_step_privacy_spending(
                target_epsilon=epsilon, target_delta=delta, num_steps=num_steps,
            )
            for lr in lrs:
                kwargs = dict(
                    num_steps=num_steps, eval_steps=eval_steps, lr=lr, weight_decay=weight_decay,
                    epsilon=per_step_epsilon, delta=per_step_delta,
                )

                mses_decay = []
                mses_const = []
                for seed in seeds:
                    _, mse_decay, _ = train(data_decay, **kwargs)
                    _, mse_const, _ = train(data_const, **kwargs)
                    mses_decay.append(mse_decay)
                    mses_const.append(mse_const)
                loss_decay.step(np.mean(mses_decay))
                loss_const.step(np.mean(mses_const))

        losses_decay.append(loss_decay.item())
        losses_const.append(loss_const.item())

    plotting = dict(
        plots=[
            dict(x=dims, y=losses_decay, label='decay', marker="x"),
            dict(x=dims, y=losses_const, label='const', marker="x"),
        ],
        options=dict(xlabel="$d$", ylabel="$\mathbb{E}[ F(\\bar{x}) ]$")
    )

    if img_dir is not None:
        utils.jdump(plotting, utils.join(img_dir, 'toyplot.json'))
        img_path = utils.join(img_dir, 'toy.png')
    else:
        img_path = None

    utils.plot_wrapper(
        img_path=img_path, **plotting,
    )


if __name__ == "__main__":
    # python decay_toy.py --img_dir "/mnt/disks/disk-2/dump/spectrum/toy"
    fire.Fire(main)
