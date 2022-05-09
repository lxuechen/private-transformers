"""
Toy example of spectral decay.
"""
import logging
import math

import fire
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

n = 5000
d = 100
R = 1
C = 1
sum_sqrt = torch.sum(torch.arange(1, d + 1) ** -.5)  # sum_j 1 / sqrt(j)
M_decay = 1 / 4 * C ** 2 / sum_sqrt
M_const = 1 / 4 * C ** 2 / d

beta_opt = torch.full(fill_value=.5 / math.sqrt(d), size=(d,), device=device)  # 2-norm == 0.5; within radius-R ball.
mu_x = torch.zeros(size=(d,), device=device)
si_x_decay = torch.sqrt(M_decay * torch.arange(1, d + 1, device=device) ** -.5)  # standard deviation.
si_x_const = torch.full(fill_value=M_const, size=(d,), device=device)
sensitivity = 2 / n * C ** 2 * R

# Per-step epsilon and delta.
epsilon = 1
delta = 1 / n ** 1.1


def make_data(mode="decay"):
    if mode == "decay":
        si_x = si_x_decay
    elif mode == "const":
        si_x = si_x_const
    else:
        raise ValueError

    x = mu_x[None, :] + si_x[None, :] * torch.randn(size=(n, d), device=device)
    xt = x * torch.clamp_max(C / x.norm(2, dim=1, keepdim=True), max=1.)  # Almost no clipping happening here.
    yt = xt @ beta_opt  # no noise.
    return xt, yt


def train_one_step(x, y, beta, lr):
    residuals = (x @ beta - y)
    grad = (residuals[:, None] * x).mean(dim=0)
    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta -= lr * grad_priv
    beta = beta * torch.clamp_max(R / beta.norm(2), max=1.)  # projection of beta into the radius-R ball.
    return beta


def evaluate(x, y, beta):
    ypred = x @ beta
    mse = ((y - ypred) ** 2).mean(dim=0)
    dis = (beta - beta_opt).norm(2)
    return mse, dis


@torch.no_grad()
def train(x, y, num_steps, eval_steps, lr):
    beta = torch.zeros(size=(d,))
    for global_step in range(1, num_steps + 1):
        if global_step % eval_steps == 0:
            mse, dis = evaluate(x=x, y=y, beta=beta)
            logging.warning(f"global_step: {global_step}, mse: {mse:.6f}, iterate dist: {dis:.6f}")
        beta = train_one_step(x=x, y=y, beta=beta, lr=lr)


def main(num_steps=100, eval_steps=10, lr=1):
    x, y = make_data(mode="decay")
    train(x, y, num_steps=num_steps, eval_steps=eval_steps, lr=lr)

    x, y = make_data(mode="const")
    train(x, y, num_steps=num_steps, eval_steps=eval_steps, lr=lr)


if __name__ == "__main__":
    fire.Fire(main)
