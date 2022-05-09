"""
Toy example of spectral decay.
"""
import logging
import math

import fire
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

n = 50000
d = 50
R = 1
C = 1
sum_sqrt = torch.sum(torch.arange(1, d + 1) ** -.5)  # sum_j 1 / sqrt(j)
G0 = 1 / 2 * C ** 2 / d

beta_opt = torch.full(fill_value=.5 / math.sqrt(d), size=(d,), device=device)  # 2-norm == 0.5; within radius-R ball.
mu_x = torch.zeros(size=(d,), device=device)
# TODO: decay has larger G0.
si_x_decay = math.sqrt(G0 * 4) * torch.sqrt(torch.arange(1, d + 1, device=device) ** -.5)  # standard deviation.
si_x_const = math.sqrt(G0) * torch.ones(size=(d,), device=device)
sensitivity = 2 / n * C ** 2 * R


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

    num_clipped = (x.norm(2, dim=1) > C).sum(dim=0)
    print(f'num_clipped: {num_clipped}')
    return xt, yt


def train_one_step(x, y, beta, lr, epsilon, delta, alpha):
    residuals = (x @ beta - y)
    grad = (residuals[:, None] * x).mean(dim=0)
    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta -= lr * (grad_priv + alpha * beta)
    beta = beta * torch.clamp_max(R / beta.norm(2), max=1.)  # projection of beta into the radius-R ball.
    return beta


def evaluate(x, y, beta):
    ypred = x @ beta
    mse = ((y - ypred) ** 2).mean(dim=0)
    dis = (beta - beta_opt).norm(2)
    return mse, dis


@torch.no_grad()
def train(x, y, num_steps, eval_steps, lr, epsilon, delta, alpha):
    beta = torch.zeros(size=(d,))
    beta_avg = beta.clone()
    for global_step in range(0, num_steps):
        if global_step % eval_steps == 0:
            mse, dis = evaluate(x=x, y=y, beta=beta_avg)
            logging.warning(f"global_step: {global_step}, mse: {mse:.6f}, iterate dist: {dis:.6f}")
        beta = train_one_step(x=x, y=y, beta=beta, lr=lr, epsilon=epsilon, delta=delta, alpha=alpha)
        beta_avg = beta_avg * global_step / (global_step + 1) + beta / (global_step + 1)
    mse, dis = evaluate(x=x, y=y, beta=beta_avg)
    logging.warning(f"final, mse: {mse:.6f}, iterate dist: {dis:.6f}")


def main(num_steps=1000, eval_steps=50, lr=5, epsilon=0.03, delta=1 / n ** 1.1):
    alpha = 1 / (10000 * lr)

    x, y = make_data(mode="decay")
    train(x, y, num_steps=num_steps, eval_steps=eval_steps, lr=lr, epsilon=epsilon, delta=delta, alpha=alpha)

    x, y = make_data(mode="const")
    train(x, y, num_steps=num_steps, eval_steps=eval_steps, lr=lr, epsilon=epsilon, delta=delta, alpha=alpha)

    deltap = delta * 10
    epsilonp = math.sqrt(2 * num_steps * math.log(1 / deltap)) * epsilon + num_steps * epsilon * (math.exp(epsilon) - 1)
    print(epsilonp, deltap)


if __name__ == "__main__":
    fire.Fire(main)
