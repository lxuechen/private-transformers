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
            sample_rate=sample_rate,
            sigma=sigma,
            steps=steps,
            alphas=alphas,
            delta=target_delta,
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
            eps_error=eps_error,
            max_compositions=steps,
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


def _compute_eps(orders, rdp, delta):
    """Compute epsilon given a list of RDP values and target delta.
    Args:
      orders: An array (or a scalar) of orders.
      rdp: A list (or a scalar) of RDP guarantees.
      delta: The target delta.
    Returns:
      Pair of (eps, optimal_order).
    Raises:
      ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):
        raise ValueError("Input lists must have the same length.")

    # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
    #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

    # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
    # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1: raise ValueError("Renyi divergence order must be >=1.")
        if r < 0: raise ValueError("Renyi divergence must be >=0.")

        if delta ** 2 + math.expm1(-r) >= 0:
            # In this case, we can simply bound via KL divergence:
            # delta <= sqrt(1-exp(-KL)).
            eps = 0  # No need to try further computation if we have eps = 0.
        elif a > 1.01:
            # This bound is not numerically stable as alpha->1.
            # Thus we have a min value of alpha.
            # The bound is also not useful for small alpha, so doesn't matter.
            eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
        else:
            # In this case we can't do anything. E.g., asking for delta = 0.
            eps = np.inf
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def eps_from_rdp(
    sample_rate,
    sigma,
    steps,
    delta,
    alphas=DEFAULT_ALPHAS,
    **_,
):
    """Compute RDP as usual, but the conversion to (ε, δ)-DP is based on result by Canonne, Kamath, Steinke.

    Code from
        https://github.com/tensorflow/privacy/blob/5f07198b66b3617b22609db983926e3ba97cd905/tensorflow_privacy/privacy/analysis/rdp_accountant.py#L237
    """
    rdp = rdp_accounting.compute_rdp(
        q=sample_rate, noise_multiplier=sigma, steps=steps, orders=alphas
    )
    # (ε, α)
    eps, alpha = _compute_eps(orders=alphas, rdp=rdp, delta=delta)
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
