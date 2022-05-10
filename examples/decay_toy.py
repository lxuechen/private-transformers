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

beta_opt = torch.full(fill_value=0.25 / math.sqrt(d), size=(d,), device=device)  # 2-norm == 0.25; within radius-R ball.
mu_x = torch.zeros(size=(d,), device=device)
si_x_decay = math.sqrt(G0) * torch.sqrt(torch.arange(1, d + 1, device=device) ** -.5)  # standard deviation.
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
    num_clipped = (x.norm(2, dim=1) > C).sum(dim=0)
    logging.warning(f"Create data in mode: {mode}; number of examples clipped: {num_clipped}")

    x = x * torch.clamp_max(C / x.norm(2, dim=1, keepdim=True), max=1.)  # Almost no clipping happening here.
    y = x @ beta_opt  # no noise.

    return x, y


def train_one_step(x, y, beta, lr, epsilon, delta, weight_decay):
    residuals = (x @ beta - y)
    grad = (residuals[:, None] * x).mean(dim=0)  # avg gradient.
    gaussian_mechanism_variance = 2. * math.log(1.25 / delta) * sensitivity ** 2. / epsilon ** 2.
    grad_priv = grad + torch.randn_like(grad) * math.sqrt(gaussian_mechanism_variance)
    beta -= lr * (grad_priv + weight_decay * beta)
    beta = beta * torch.clamp_max(.5 * R / beta.norm(2), max=1.)  # projection of beta into the radius-R ball.
    return beta


def evaluate(x, y, beta):
    ypred = x @ beta
    mse = ((y - ypred) ** 2).mean(dim=0)
    dis = (beta - beta_opt).norm(2)
    return mse, dis


@torch.no_grad()
def train(x, y, num_steps, eval_steps, lr, epsilon, delta, weight_decay):
    beta = torch.zeros(size=(d,))
    beta_avg = beta.clone()
    for global_step in range(0, num_steps):
        if global_step % eval_steps == 0:
            mse, dis = evaluate(x=x, y=y, beta=beta_avg)
            logging.warning(f"global_step: {global_step}, mse: {mse:.6f}, iterate dist: {dis:.6f}")
        beta = train_one_step(x=x, y=y, beta=beta, lr=lr, epsilon=epsilon, delta=delta, weight_decay=weight_decay)
        beta_avg = beta_avg * global_step / (global_step + 1) + beta / (global_step + 1)
    mse, dis = evaluate(x=x, y=y, beta=beta_avg)
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
    num_steps=1000, eval_steps=50, lr=1, weight_decay=1e-6,
    epsilon=3, delta=1 / n ** 1.1,
):
    per_step_epsilon, per_step_delta = make_per_step_privacy_spending(
        target_epsilon=epsilon, target_delta=delta, num_steps=num_steps,
    )
    logging.warning(f'per_step_epsilon: {per_step_epsilon}, per_step_delta: {per_step_delta}')

    x, y = make_data(mode="decay")
    train(
        x, y,
        num_steps=num_steps, eval_steps=eval_steps, lr=lr, weight_decay=weight_decay,
        epsilon=per_step_epsilon, delta=per_step_delta,
    )

    x, y = make_data(mode="const")
    train(
        x, y,
        num_steps=num_steps, eval_steps=eval_steps, lr=lr, weight_decay=weight_decay,
        epsilon=per_step_epsilon, delta=per_step_delta,
    )


if __name__ == "__main__":
    fire.Fire(main)
