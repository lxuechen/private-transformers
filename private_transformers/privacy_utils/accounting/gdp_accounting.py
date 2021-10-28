r"""This code applies the Dual and Central Limit
Theorem (CLT) to estimate privacy budget of an iterated subsampled
Gaussian Mechanism (by either uniform or Poisson subsampling).

This file is a direct copy of
https://github.com/woodyx218/privacy/blob/d06340e1cf4944faa065644efb5e95950fbaf487/tensorflow_privacy/privacy/analysis/gdp_accountant.py
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm


def compute_mu_uniform(epochs, noise_multi, sample_rate):
    """Compute mu from uniform subsampling."""
    T = epochs / sample_rate
    c = np.sqrt(T) * sample_rate
    return (
        np.sqrt(2) * c * np.sqrt(
            np.exp(noise_multi ** (-2)) * norm.cdf(1.5 / noise_multi) + 3 * norm.cdf(-0.5 / noise_multi) - 2
        )
    )


def compute_mu_poisson(epochs, noise_multi, sample_rate):
    """Compute mu from Poisson subsampling."""
    T = epochs / sample_rate
    return np.sqrt(np.exp(noise_multi ** (-2)) - 1) * np.sqrt(T) * sample_rate


def delta_eps_mu(eps, mu):
    """Compute dual between mu-GDP and (epsilon, delta)-DP."""
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(mu, delta, bracket=(0, 500)):
    """Compute epsilon from mu given delta via inverse dual."""

    def f(x):
        """Reversely solve dual by matching delta."""
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=bracket, method='brentq').root


def compute_eps_uniform(epochs, noise_multi, sample_rate, delta):
    """Compute epsilon given delta from inverse dual of uniform subsampling."""
    return eps_from_mu(compute_mu_uniform(epochs, noise_multi, sample_rate), delta)


def compute_eps_poisson(epochs, noise_multi, sample_rate, delta):
    """Compute epsilon given delta from inverse dual of Poisson subsampling."""
    return eps_from_mu(compute_mu_poisson(epochs, noise_multi, sample_rate), delta)


def get_noise_multiplier(
    sample_rate,
    epochs,
    target_epsilon,
    target_delta,
    sigma_min=0.01,
    sigma_max=10.0,
    threshold=1e-3,
):
    """Estimate the noise multiplier by binary search."""
    while sigma_max - sigma_min > threshold:
        sigma_mid = (sigma_min + sigma_max) / 2.
        epsilon = compute_eps_poisson(epochs, sigma_mid, sample_rate, target_delta)
        if epsilon > target_epsilon:
            sigma_min = sigma_mid
        else:
            sigma_max = sigma_mid
    return sigma_max
