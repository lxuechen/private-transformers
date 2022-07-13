import math
from typing import Callable, Optional, Union

import numpy as np

from . import rdp_accounting

ACCOUNTING_MODES = ('rdp', 'glw', 'all')
DEFAULT_ALPHAS = tuple(1 + x / 10.0 for x in range(1, 100)) + tuple(range(12, 64))  # RDP.


def get_sigma_from_rdp(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[Union[float, int]] = None,
    alphas=DEFAULT_ALPHAS,
    threshold=1e-3,
    sigma_hi_init=4,
    sigma_lo_init=0.1,
    steps=None,
    **kwargs,
) -> float:
    """Get noise multiplier σ for a given ε from RDP accounting."""
    if steps is None:
        if epochs is None:
            raise ValueError("Epochs and steps cannot both be None.")
        steps = math.ceil(epochs / sample_rate)

    def sigma_to_eps(sigma):
        eps, _ = eps_from_rdp(
            sigma=sigma,
            sample_rate=sample_rate,
            delta=target_delta,
            steps=steps,
            alphas=alphas,
        )
        return eps

    return _get_sigma_with_target_epsilon(
        sigma_hi_init=sigma_hi_init,
        sigma_lo_init=sigma_lo_init,
        sigma_to_eps=sigma_to_eps,
        target_epsilon=target_epsilon,
        threshold=threshold,
    )


def get_sigma_from_glw(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[Union[float, int]] = None,
    eps_error=0.05,
    threshold=1e-3,
    sigma_hi_init=4,
    sigma_lo_init=0.1,
    steps=None,
    **kwargs,
):
    """Get noise multiplier σ for a given ε from numerically composing tradeoff functions."""
    from prv_accountant import Accountant

    if steps is None:
        if epochs is None:
            raise ValueError("Epochs and steps cannot both be None")
        steps = math.ceil(epochs / sample_rate)

    def sigma_to_eps(sigma):
        accountant = Accountant(
            noise_multiplier=sigma,
            sampling_probability=sample_rate,
            delta=target_delta,
            max_compositions=steps,
            eps_error=eps_error,
        )
        eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=steps)
        return eps_upper  # Be conservative.

    return _get_sigma_with_target_epsilon(
        sigma_hi_init=sigma_hi_init,
        sigma_lo_init=sigma_lo_init,
        sigma_to_eps=sigma_to_eps,
        target_epsilon=target_epsilon,
        threshold=threshold,
    )


def _get_sigma_with_target_epsilon(
    sigma_hi_init: float,
    sigma_lo_init: float,
    sigma_to_eps: Callable,
    target_epsilon: float,
    threshold: float,
) -> float:
    """Core logic for binary searching σ given ε and δ."""
    if sigma_lo_init > sigma_hi_init:
        raise ValueError("`sigma_lo` should be smaller than `sigma_hi`.")

    # Find an appropriate region for binary search.
    sigma_hi = sigma_hi_init
    sigma_lo = sigma_lo_init

    # Ensure sigma_hi isn't too small.
    while True:
        eps = sigma_to_eps(sigma_hi)
        if eps < target_epsilon:
            break
        sigma_hi *= 2

    # Ensure sigma_lo isn't too large.
    while True:
        eps = sigma_to_eps(sigma_lo)
        if eps > target_epsilon:
            break
        sigma_lo /= 2

    # Binary search.
    while sigma_hi - sigma_lo > threshold:
        sigma = (sigma_hi + sigma_lo) / 2
        eps = sigma_to_eps(sigma)
        if eps < target_epsilon:
            sigma_hi = sigma
        else:
            sigma_lo = sigma

    # Conservative estimate.
    return sigma_hi


def eps_from_rdp(
    sample_rate,
    sigma,
    steps,
    delta,
    alphas=DEFAULT_ALPHAS,
    **_,
):
    """Compute RDP as usual, but convert to (ε, δ)-DP based on the result by Canonne, Kamath, Steinke."""
    rdp = rdp_accounting.compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=steps, orders=alphas)
    eps, alpha = rdp_accounting.get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)
    return eps, alpha


def eps_from_glw(
    sample_rate,
    sigma,
    steps,
    delta,
    eps_error=0.05,
    **_,
):
    if steps == 0:
        return np.nan, np.nan

    from prv_accountant import Accountant
    accountant = Accountant(
        noise_multiplier=sigma,
        sampling_probability=sample_rate,
        delta=delta,
        eps_error=eps_error,
        max_compositions=steps
    )
    eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=steps)
    return dict(eps_low=eps_low, eps_estimate=eps_estimate, eps_upper=eps_upper)
