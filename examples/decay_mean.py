"""
mean estimation with decaying importance.
"""
import dataclasses
import logging
import math

import fire
import numpy as np
from swissknife import utils
import torch
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclasses.dataclass
class Data:
    beta_opt: torch.Tensor
    beta: torch.Tensor
    Ar: torch.Tensor  # A^{1/2}.
    G0: torch.Tensor  # max eigenvalue of A^{1/2}.
    sensitivity: float


def make_data(n=100000, d=10, dmin=10, mu_beta=0.1, si_beta=0.1, mode="linear"):
    assert d >= dmin

    beta_opt = torch.full(fill_value=mu_beta, device=device, size=(n, d))
    beta = beta_opt + torch.randn_like(beta_opt) * si_beta
    beta[:, dmin:] = 0.  # Ensure init distance to opt is the same.

    if mode == "constant":
        Ar = torch.arange(1, d + 1, device=device)
    elif mode == "sqrt":
        Ar = torch.arange(1, d + 1, device=device) ** -.5
    elif mode == "linear":
        Ar = torch.arange(1, d + 1, device=device) ** -1.
    elif mode == "quadratic":
        Ar = torch.arange(1, d + 1, device=device) ** -2.
    else:
        raise NotImplementedError

    G0 = Ar[0]
    sensitivity = 2 * G0 / n

    return Data(beta=beta, beta_opt=beta_opt, Ar=Ar, G0=G0, sensitivity=sensitivity)


def evaluate(data, beta):
    res = (data.Ar[None, :] * (data.beta - beta))  # (n, d).
    return res.norm(2, dim=1).mean(dim=0).item()


def train_one_step(data, beta, lr, epsilon, delta, weight_decay):
    # res = (data.Ar[None, :] * (data.beta - beta[None, :]))  # (n, d).
    # grad = data.Ar * (res / res.norm(2, dim=1, keepdim=True)).mean(dim=0)

    assert beta.dim() == 2
    assert data.Ar.dim() == 1

    beta = beta.clone().requires_grad_(True)
    with torch.enable_grad():
        loss = (data.Ar[None, :] * (beta - data.beta)).norm(2, dim=1).mean(dim=0)
        grad, = torch.autograd.grad(loss, beta)
        grad.detach_()

    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * data.sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta -= lr * (grad_priv + weight_decay * beta)
    return beta


@torch.no_grad()
def train(data, num_steps, eval_steps, lr, epsilon, delta, weight_decay):
    beta = torch.zeros(size=(1, data.beta_opt.size(1),), device=device)
    beta_avg = beta.clone()
    for global_step in range(0, num_steps):
        if global_step % eval_steps == 0:
            mdist = evaluate(data=data, beta=beta_avg)
            logging.warning(f"global_step: {global_step}, lr: {lr:.6f}, num_steps: {num_steps}, mdist: {mdist:.6f}")
        beta = train_one_step(
            data=data, beta=beta, lr=lr, epsilon=epsilon, delta=delta, weight_decay=weight_decay
        )
        beta_avg = beta_avg * global_step / (global_step + 1) + beta / (global_step + 1)
    mdist = evaluate(data=data, beta=beta_avg)
    logging.warning(f"final, lr: {lr:.6f}, num_steps: {num_steps}, mdist: {mdist:.6f}")
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


def main(img_dir=None, eval_steps=10000, weight_decay=0, epsilon=3, delta=1e-6, seeds=(42, 96, 10000)):
    dims = (10, 50, 100, 500,)
    num_steps_list = (10, 50, 100, 400, 700, 1000, 1300, 1600, 1900, 2200, 3000, 5000,)
    lrs = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10)

    # dims = (10,)
    # num_steps_list = (10, 50, 100, 500, 1000)
    # lrs = (1e-4, 1e-3, 1e-2, 1e-1, 1)

    losses_decay = []
    losses_const = []
    for dim in tqdm.tqdm(dims, desc="dims"):
        data_decay = make_data(mode='constant', d=dim)
        data_const = make_data(mode="linear", d=dim)

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
                    _, mse_decay = train(data_decay, **kwargs)
                    _, mse_const = train(data_const, **kwargs)
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
    # python decay_mean.py --img_dir "/mnt/disks/disk-2/dump/spectrum/toy2"
    fire.Fire(main)
