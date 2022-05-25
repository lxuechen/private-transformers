# Shameless copy from https://github.com/google/spectral-density/blob/f0d3f1446bb1c200d9200cbdc67407e3f148ccba/jax/density.py#L120

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for converting Lanczos outputs to densities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np


def eigv_to_density(eig_vals, all_weights=None, grids=None,
                    grid_len=10000, sigma_squared=None, grid_expand=1e-2):
    """Compute the smoothed spectral density from a set of eigenvalues.

    Convolves the given eigenvalues with a Gaussian kernel, weighting the values
    by all_weights (or uniform weighting if all_weights is None). Example output
    can be seen in Figure 1 of https://arxiv.org/pdf/1901.10159.pdf. Visualizing
    the estimated density can be done by calling plt.plot(grids, density). There
    is likely not a best value of sigma_squared that works for all use cases,
    so it is recommended to try multiple values in the range [1e-5,1e-1].

    Args:
      eig_vals: Array of shape [num_draws, order]
      all_weights: Array of shape [num_draws, order], if None then weights will be
        taken to be uniform.
      grids: Array of shape [grid_len], the smoothed spectrum will be plotted
        in the interval [grids[0], grids[-1]]. If None then grids will be
        computed based on max and min eigenvalues and grid length.
      grid_len: Integer specifying number of grid cells to use, only used if
        grids is None
      sigma_squared: Scalar. Controls the smoothing of the spectrum estimate.
        If None, an appropriate value is inferred.
      grid_expand: Controls the window of values that grids spans.
        grids[0] = smallest eigenvalue - grid_expand.
        grids[-1] = largest_eigenvalue + grid_expand.

    Returns:
      density: Array of shape [grid_len], the estimated density, averaged over
        all draws.
      grids: Array of shape [grid_len]. The values the density is estimated on.
    """
    if all_weights is None:
        all_weights = np.ones(eig_vals.shape) * 1.0 / float(eig_vals.shape[1])
    num_draws = eig_vals.shape[0]

    lambda_max = np.nanmean(np.max(eig_vals, axis=1), axis=0) + grid_expand
    lambda_min = np.nanmean(np.min(eig_vals, axis=1), axis=0) - grid_expand

    if grids is None:
        assert grid_len is not None, 'grid_len is required if grids is None.'
        grids = np.linspace(lambda_min, lambda_max, num=grid_len)

    grid_len = grids.shape[0]
    if sigma_squared is None:
        sigma = 10 ** -5 * max(1, (lambda_max - lambda_min))
    else:
        sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    density_each_draw = np.zeros((num_draws, grid_len))
    for i in range(num_draws):

        if np.isnan(eig_vals[i, 0]):
            raise ValueError('tridaig has nan values.')
        else:
            for j in range(grid_len):
                x = grids[j]
                vals = _kernel(eig_vals[i, :], x, sigma)
                density_each_draw[i, j] = np.sum(vals * all_weights[i, :])
    density = np.nanmean(density_each_draw, axis=0)
    norm_fact = np.sum(density) * (grids[1] - grids[0])
    density = density / norm_fact
    return density, grids


def tridiag_to_eigv(tridiag_list):
    """Preprocess the tridiagonal matrices for density estimation.

    Args:
      tridiag_list: Array of shape [num_draws, order, order] List of the
        tridiagonal matrices computed from running num_draws independent runs
        of lanczos. The output of this function can be fed directly into
        eigv_to_density.

    Returns:
      eig_vals: Array of shape [num_draws, order]. The eigenvalues of the
        tridiagonal matricies.
      all_weights: Array of shape [num_draws, order]. The weights associated with
        each eigenvalue. These weights are to be used in the kernel density
        estimate.
    """
    # Calculating the node / weights from Jacobi matrices.
    num_draws = len(tridiag_list)
    num_lanczos = tridiag_list[0].shape[0]
    eig_vals = np.zeros((num_draws, num_lanczos))
    all_weights = np.zeros((num_draws, num_lanczos))
    for i in range(num_draws):
        nodes, evecs = np.linalg.eigh(tridiag_list[i])
        index = np.argsort(nodes)
        nodes = nodes[index]
        evecs = evecs[:, index]
        eig_vals[i, :] = nodes
        all_weights[i, :] = evecs[0] ** 2
    return eig_vals, all_weights


def tridiag_to_density(tridiag_list, sigma_squared=1e-5, grid_len=10000):
    """This function estimates the smoothed density from the output of lanczos.

    Args:
      tridiag_list: Array of shape [num_draws, order, order] List of the
        tridiagonal matrices computed from running num_draws independent runs
        of lanczos.
      sigma_squared: Controls the smoothing of the density.
      grid_len: Controls the granularity of the density.

    Returns:
      density: Array of size [grid_len]. The smoothed density estimate averaged
        over all num_draws.
      grids: Array of size [grid_len]. The values the density estimate is on.
    """
    eig_vals, all_weights = tridiag_to_eigv(tridiag_list)
    density, grids = eigv_to_density(eig_vals, all_weights,
                                     grid_len=grid_len,
                                     sigma_squared=sigma_squared)
    return density, grids


def _kernel(x, x0, variance):
    """Point estimate of the Gaussian kernel.

    This function computes the Gaussian kernel for
    C exp(-(x - x0) ^2 /(2 * variance)) where C is the appropriate normalization.
    variance should be a list of length 1. Either x0 or x should be a scalar. Only
    one of the x or x0 can be a numpy array.

    Args:
      x: Can be either scalar or array of shape [order]. Points to estimate
        the kernel on.
      x0: Scalar. Mean of the kernel.
      variance: Scalar. Variance of the kernel.

    Returns:
      point_estimate: A scalar corresponding to
        C exp(-(x - x0) ^2 /(2 * variance)).
    """
    coeff = 1.0 / np.sqrt(2 * math.pi * variance)
    val = -(x0 - x) ** 2
    val = val / (2.0 * variance)
    val = np.exp(val)
    point_estimate = coeff * val
    return point_estimate
