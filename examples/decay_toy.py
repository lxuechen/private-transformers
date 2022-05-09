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
M = 1 / 4 * C ** 2 / sum_sqrt

beta_opt = torch.full(fill_value=.5 / math.sqrt(d), size=(d,), device=device)  # 2-norm == 0.5
mu_x = torch.zeros(size=(d,), device=device)
si_x = torch.sqrt(M * torch.arange(1, d + 1) ** -.5)  # standard deviation.
sensitivity = 2 / n * C ** 2 * R

# Per-step epsilon and delta.
epsilon = 1
delta = 1 / n ** 1.1


def make_data():
    x = mu_x[None, :] + si_x[None, :] * torch.randn(size=(n, d), device=device)
    xt = x * torch.clamp_max(C / x.norm(2, dim=1, keepdim=True), max=1.)  # Almost no clipping happening here.
    yt = xt @ beta_opt  # no noise.
    return xt, yt


def train_one_step(xt, yt, beta, lr):
    residuals = (xt @ beta - yt)
    grad = (residuals[:, None] * xt).mean(dim=0)
    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta -= lr * grad_priv
    beta = beta * torch.clamp_max(R / beta.norm(2), max=1.)  # projection of beta into the radius-R ball.
    return beta


def evaluate(x, y, beta):
    ypred = x @ beta
    mse = ((y - ypred) ** 2).mean(dim=0)
    return mse


@torch.no_grad()
def train(num_steps, eval_steps, lr):
    xt, yt = make_data()
    beta = torch.zeros(size=(d,))
    for global_step in range(1, num_steps + 1):
        beta = train_one_step(xt=xt, yt=yt, beta=beta, lr=lr)
        if global_step % eval_steps == 0:
            mse = evaluate(x=xt, y=yt, beta=beta)
            logging.warning(f"global_step: {global_step}, mse: {mse:.4f}")


def main(num_steps=1000, eval_steps=1, lr=1):
    train(num_steps=num_steps, eval_steps=eval_steps, lr=lr)


if __name__ == "__main__":
    fire.Fire(main)
