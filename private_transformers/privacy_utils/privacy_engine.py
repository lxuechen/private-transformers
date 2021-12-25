"""Code for a privacy engine that plays nicely with Hugging Face transformers.

Design mostly based on Opacus with the exception that `.step` and `virtual_step`
takes in per-example losses, which should not be called with `.backward()` by
the user.
"""

import collections
import logging
import math
import types
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from . import autograd_grad_sample
from . import misc
from . import transformers_support
from .accounting import gdp_accounting, rdp_accounting

DEFAULT_ALPHAS = tuple(1 + x / 10.0 for x in range(1, 100)) + tuple(range(12, 64))


class PrivacyEngine(object):
    """Differentially-private optimization engine that works gracefully with Hugging Face transformers.

    Supports ghost clipping as described in
        Li, X., Tramèr, F., Liang, P., & Hashimoto, T. (2021).
        Large Language Models Can Be Strong Differentially Private Learners.
        arXiv preprint arXiv:2110.05679.

    Implicitly assumes inputs are in batch first format.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        batch_size: int,
        sample_size: int,
        max_grad_norm: float,
        epochs: Optional[Union[int, float]] = None,
        noise_multiplier: Optional[float] = None,
        target_epsilon: Optional[float] = None,
        target_delta: Optional[float] = None,
        alphas: Sequence[float] = DEFAULT_ALPHAS,
        record_snr: bool = True,
        named_params: Optional[Sequence] = None,
        fp16: bool = False,
        numerical_stability_constant=1e-6,
        ghost_clipping: bool = False,
        # Accounting specifics.
        accounting_mode="rdp_cks",
        eps_error=0.05,
        **unused_kwargs,
    ):
        """Initialize the engine.

        Args:
            module: The PyTorch module for which per-sample gradient is required.
                Setting the `requires_grad` attribute of a parameter to False
                disables the per-sample gradient accumulation.
            batch_size: The expected lot size.
            sample_size: Size of dataset.
            max_grad_norm: The maximum 2-norm for gradient clipping.
            epochs: The number of epochs for training.
            noise_multiplier: The extra multiplier for DP-SGD noise.
            target_epsilon: The target privacy spending. Only used to estimate the `noise_multiplier` if it is not set.
            target_delta: The target failure probability. Defaults to 1 / (2 * sample_size) if not set.
            alphas: The RDP orders for (ε, δ)-DP conversion. Useless if not accounting in RDP.
            record_snr: Record and report the signal-to-noise ratio --
                ratio between norm of summed clipped gradient and norm of noise vector.
            named_params: Specifies which parameters need gradients;
                defaults to use parameters which require grad in module.
            fp16: Set this to True when training with mixed-precision.
            numerical_stability_constant: Small constant to avoid division by 0 when clipping.
            ghost_clipping: Set this to True to use memory efficient ghost clipping.
            accounting_mode: The method of accounting privacy. One of (`rdp`, `gdp`, `rdp_cks`, `glw`, `all`).
                Meanings of shorthands:
                    - rdp: The method in "Rényi Differential Privacy of the Sampled Gaussian Mechanism".
                        https://arxiv.org/abs/1908.10530
                    - rdp_cks: Account loss with RDP but perform conversion to approx-DP with a procedure defined in
                        "The Discrete Gaussian for Differential Privacy".
                        https://arxiv.org/abs/2004.00010
                        CKS are authors' last name's first letters.
                    - gdp: Account loss with Gaussian DP and its central limit theorem described in
                        "Deep Learning with Gaussian Differential Privacy".
                        WARNING: This method may underestimate privacy spending.
                    - glw: Account loss by numerically composing tradeoff functions in f-DP; defined in
                        "Numerical composition of differential privacy".
                        https://arxiv.org/abs/2106.02848
                        GLW are authors' last name's first letters.
                    - all: Report loss with all methods listed above.
            eps_error: Error threshold for upper and lower bound in the GLW accounting procedure.
        """
        misc.handle_unused_kwargs(unused_kwargs)
        del unused_kwargs

        super(PrivacyEngine, self).__init__()
        if accounting_mode not in ('rdp', 'gdp', 'rdp_cks', 'glw', 'all',):
            raise ValueError(f"Unknown accounting mode: {accounting_mode}")
        if epochs <= 0.0:
            raise ValueError(f"Number of training epochs cannot be non-positive, but found epochs={epochs}")

        # Privacy parameters.
        sample_rate = batch_size / sample_size
        if target_delta is None:
            target_delta = 1 / (2 * sample_size)
        if noise_multiplier is None:
            if target_epsilon is None or epochs is None:
                raise ValueError(
                    f"`target_epsilon` and `epochs` must be specified when `noise_multiplier` is `None`."
                )
            kwargs_for_get_sigma = dict(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_rate=sample_rate,
                epochs=epochs,
                alphas=alphas,
                eps_error=eps_error,
            )
            if accounting_mode == "rdp":
                noise_multiplier = get_sigma_from_rdp(**kwargs_for_get_sigma)
            elif accounting_mode == "rdp_cks":
                noise_multiplier = get_sigma_from_rdp_cks(**kwargs_for_get_sigma)
            elif accounting_mode == "glw":
                noise_multiplier = get_sigma_from_glw(**kwargs_for_get_sigma)
            else:
                noise_multiplier = get_sigma_from_gdp(**kwargs_for_get_sigma)

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.max_grad_norm = max_grad_norm

        self.epochs = epochs
        self.noise_multiplier = noise_multiplier
        self.effective_noise_multiplier = noise_multiplier / batch_size
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.alphas = alphas
        self.accounting_mode = accounting_mode
        self.record_snr = record_snr

        # Internals.
        self.steps = 0  # Tracks privacy spending.

        # Recording.
        self.max_clip = None
        self.min_clip = None
        self.med_clip = None
        self.signal = None
        self.noise = None
        self.snr = None
        self.noise_limit = None

        # Record parameters.
        self.module = module
        if named_params is None:
            self.named_params = tuple(
                (name, param) for (name, param) in module.named_parameters() if param.requires_grad
            )
        else:
            self.named_params = named_params
        self.num_params = sum(param.numel() for _, param in self.named_params)

        self._locked = False  # Lock the part where noisy gradients is created (in `self.step`) if True.
        self.fp16 = fp16
        self.numerical_stability_constant = numerical_stability_constant
        self.ghost_clipping = ghost_clipping
        if ghost_clipping:
            if fp16:
                # TODO: Make ghost clipping work with mixed-precision.
                raise NotImplementedError("Ghost clipping doesn't support mixed-precision.")
            # Prepare for first backward in ghost clipping.
            autograd_grad_sample.set_hooks_mode("ghost_norm")

        transformers_support.forward_swapper(module=module)

    def lock(self):
        self._locked = True

    def unlock(self):
        self._locked = False

    def attach(self, optimizer):
        autograd_grad_sample.add_hooks(model=self.module, batch_first=True, loss_reduction="sum", fp16=self.fp16)

        # Override zero grad.
        def dp_zero_grad(_self, *args, **kwargs):
            _self.privacy_engine.zero_grad()

        # Override step.
        def dp_step(_self, **kwargs):
            closure = kwargs.pop("closure", None)

            _self.privacy_engine.step(**kwargs)
            _self.original_step(closure=closure)
            _self.privacy_engine.unlock()  # Only enable creating new grads once parameters are updated.
            _self.privacy_engine.steps += 1

        def virtual_step(_self, **kwargs):
            _self.privacy_engine.virtual_step(**kwargs)

        def get_privacy_spent(_self, **kwargs):
            _self.privacy_engine.get_privacy_spent(**kwargs)

        def get_training_stats(_self, **kwargs):
            _self.privacy_engine.get_training_stats(**kwargs)

        optimizer.privacy_engine = self

        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)

        optimizer.original_zero_grad = optimizer.zero_grad
        optimizer.zero_grad = types.MethodType(dp_zero_grad, optimizer)

        optimizer.virtual_step = types.MethodType(virtual_step, optimizer)

        # Make getting info easier.
        optimizer.get_privacy_spent = types.MethodType(get_privacy_spent, optimizer)
        optimizer.get_training_stats = types.MethodType(get_training_stats, optimizer)

        self.module.privacy_engine = self

        # Just to be safe, we also override `zero_grad` for module.
        self.module.original_zero_grad = self.module.zero_grad
        self.module.zero_grad = types.MethodType(dp_zero_grad, self.module)

        # For easy detaching.
        self.optimizer = optimizer

    def detach(self):
        optimizer = self.optimizer
        optimizer.step = optimizer.original_step
        optimizer.zero_grad = optimizer.original_zero_grad
        delattr(optimizer, "privacy_engine")
        delattr(optimizer, "original_step")
        delattr(optimizer, "original_zero_grad")
        delattr(optimizer, "virtual_step")
        delattr(optimizer, "get_privacy_spent")
        delattr(optimizer, "get_training_stats")

        module = self.module
        autograd_grad_sample.remove_hooks(module)
        autograd_grad_sample.set_hooks_mode("default")  # This is super important when there are multiple attaches!
        module.zero_grad(skip_grad=True)
        module.zero_grad = module.original_zero_grad
        delattr(module, "original_zero_grad")

    # --- everything specific to ghost clipping ---
    def _ghost_step(self, loss: torch.Tensor):
        """Run double-backward on per-example loss, then sum up all gradients and noise it."""
        if self._locked:  # Skip this gradient creation step if already created gradient and haven't stepped.
            logging.warning("Attempted to step, but the engine is on lock.")
            return

        self._ghost_helper(loss)

        # Add noise and scale by inverse batch size.
        signals, noises = [], []
        for name, param in self.named_params:
            # This is only True when there are previous virtual steps.
            # The .grad contains the summed clipped gradients of this batch.
            # Summed clipped gradients of previous batches are in .summed_grad.
            # When there's no gradient accumulation, .summed_grad is not created.
            if hasattr(param, 'summed_grad'):
                param.grad += param.summed_grad
            if self.record_snr:
                signals.append(param.grad.reshape(-1).norm(2).cpu())

            if self.noise_multiplier > 0 and self.max_grad_norm > 0:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm,
                    size=param.size(),
                    device=param.device,
                    dtype=param.dtype,
                )
                param.grad += noise
                if self.record_snr:
                    noises.append(noise.reshape(-1).norm(2).cpu())
                del noise

            param.grad /= self.batch_size

        if self.record_snr and noises:
            self.signal, self.noise = tuple(torch.stack(lst).norm(2).item() for lst in (signals, noises))
            self.noise_limit = math.sqrt(self.num_params) * self.noise_multiplier * self.max_grad_norm
            self.snr = self.signal / self.noise
        else:
            self.snr = math.inf  # Undefined!

        self.lock()  # Make creating new gradients impossible, unless optimizer.step is called.

    @torch.no_grad()
    def _ghost_virtual_step(self, loss: torch.Tensor):
        """Run double-backward on per-example loss, then accumulate loss."""
        self._ghost_helper(loss)
        for name, param in self.named_params:
            if hasattr(param, 'summed_grad'):
                param.summed_grad += param.grad
            else:
                param.summed_grad = param.grad

            if hasattr(param, "grad"):
                del param.grad
            if hasattr(param, "norm_sample"):
                del param.norm_sample
            if hasattr(param, "grad_sample"):
                del param.grad_sample

    @torch.enable_grad()
    def _ghost_helper(self, loss: torch.Tensor):
        """Given per-example losses, do the double backward thing."""
        if loss.dim() != 1:
            raise ValueError(f"Expected `loss` to be a the per-example loss 1-D tensor.")

        first_loss = loss.sum()
        first_loss.backward(retain_graph=True)

        # Prepare for second backward.
        autograd_grad_sample.set_hooks_mode("ghost_grad")

        # The first backward might have accumulated things we don't need into `.grad`;
        #   remove it before the second pass to avoid accumulating garbage.
        for name, param in self.named_params:
            if hasattr(param, "grad"):
                del param.grad

        coef_sample = self.get_coef_sample()
        second_loss = (coef_sample * loss).sum(dim=0)
        second_loss.backward()

        # Prepare for first backward (in the next round).
        autograd_grad_sample.set_hooks_mode("ghost_norm")

    def get_norm_sample(self):
        """Get per-example norms."""
        norm_sample = torch.stack([param.norm_sample for name, param in self.named_params], dim=0).norm(2, dim=0)
        return norm_sample

    def get_coef_sample(self):
        """Get per-example gradient scaling factor for clipping."""
        norm_sample = self.get_norm_sample()
        return torch.clamp_max(self.max_grad_norm / (norm_sample + self.numerical_stability_constant), 1.)

    # ---------------------------------------------

    @torch.no_grad()
    def step(self, scale=1, **kwargs):
        """Step function.

        `loss` must be passed in as a keyword argument!
        """
        if self.ghost_clipping:
            self._ghost_step(loss=kwargs.pop("loss"))
        else:
            self._step(loss=kwargs.pop("loss"), scale=scale)

    @torch.no_grad()
    def virtual_step(self, scale=1, **kwargs):
        """Virtual step function when there's gradient accumulation.

        `loss` must be passed in as a keyword argument!
        """
        if self.ghost_clipping:
            self._ghost_virtual_step(loss=kwargs.pop("loss"))
        else:
            self._virtual_step(loss=kwargs.pop("loss"), scale=scale)

    def _step(self, loss, scale=1.):
        """Create noisy gradients.

        Should be ran right before you call `optimizer.step`.

        This function does 3 things:
            1) call `loss.backward()`
            2) clip the current `.grad_sample` and add that to `.summed_grad`
            3) noise the gradients
        In mixed-precision training (with amp), the last two steps require knowing the loss scaling factor.

        Args:
            loss: The per-example loss; a 1-D tensor.
            scale: The loss up-scaling factor in amp. In full precision, this arg isn't useful.
        """
        if self._locked:  # Skip this gradient creation step if already created gradient and haven't stepped.
            logging.warning("Attempted to step, but the engine is on lock.")
            return

        norm_sample, coef_sample = self._accumulate_summed_grad(loss=loss, scale=scale)
        # Collect stats for debugging.
        self.max_clip = coef_sample.max().item()
        self.min_clip = coef_sample.min().item()
        self.med_clip = coef_sample.median().item()

        # Add noise and scale by inverse batch size.
        signals, noises = [], []
        for name, param in self.named_params:
            if hasattr(param, 'summed_grad'):  # Ultra important to override `.grad`.
                param.grad = param.summed_grad.to(param.dtype)
            else:
                logging.fatal(
                    f"PrivacyEngine should not reach here; "
                    f"this means either "
                    f"1) there is parameter which requires gradient, but was not used in the computational graph, or "
                    f"2) the backward hook registry failed to find the corresponding module to register."
                )
            if self.record_snr:
                signals.append(param.grad.reshape(-1).norm(2).cpu())

            if self.noise_multiplier > 0 and self.max_grad_norm > 0:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm * scale,
                    size=param.size(),
                    device=param.device,
                    dtype=param.dtype,
                )
                if self.record_snr:
                    noises.append(noise.reshape(-1).norm(2).cpu())
                param.grad += noise
                del noise

            param.grad /= self.batch_size

        if self.record_snr and noises:
            self.signal, self.noise = tuple(torch.stack(lst).norm(2).item() for lst in (signals, noises))
            self.noise_limit = math.sqrt(self.num_params) * self.noise_multiplier * self.max_grad_norm
            self.snr = self.signal / self.noise
        else:
            self.snr = math.inf  # Undefined!

        self.lock()  # Make creating new gradients impossible, unless optimizer.step is called.

    def zero_grad(self, skip_grad=False):
        for name, param in self.named_params:
            if hasattr(param, "grad_sample"):
                del param.grad_sample
            if hasattr(param, "norm_sample"):
                del param.norm_sample
            if hasattr(param, "summed_grad"):
                del param.summed_grad
            if not skip_grad:
                if hasattr(param, "grad"):
                    del param.grad

    def _virtual_step(self, loss, scale=1.):
        self._accumulate_summed_grad(loss=loss, scale=scale)
        for name, param in self.named_params:
            # Del everything except `.summed_grad` to save memory!
            if hasattr(param, "grad_sample"):
                # This must be deleted due to how `privacy_utils::supported_layers_grad_samplers.py` works!
                #   When a parameter with `.grad_sample` is reused, the per-sample gradients are accumulated!
                del param.grad_sample
            if hasattr(param, "grad"):
                del param.grad

    @torch.no_grad()
    def _accumulate_summed_grad(self, loss, scale=1.):
        """Accumulate signal by summing clipped gradients."""
        if loss.dim() != 1:
            raise ValueError(f"Expected `loss` to be a the per-example loss 1-D tensor.")
        with torch.enable_grad():
            loss.sum(dim=0).backward()

        norm_sample = []
        for name, param in self.named_params:
            try:
                batch_size = param.grad_sample.size(0)
            except AttributeError as error:
                args = error.args
                extra_msg = f"\n *** {name} parameter has no grad_sample attribute ***"
                error.args = (args[0] + extra_msg, *args[1:])
                raise error
            norm = param.grad_sample.reshape(batch_size, -1).norm(2, dim=1)
            norm_sample.append(norm.cpu())

        # The stack operation here is prone to error, thus clarify where the error is.
        try:
            norm_sample = torch.stack(norm_sample, dim=0).norm(2, dim=0)
        except RuntimeError as runtime_error:
            args = runtime_error.args

            # Get the major shape.
            shapes = collections.defaultdict(int)
            for tensor in norm_sample:
                shapes[tensor.size()] += 1

            major_shape = None
            major_count = 0
            for shape, count in shapes.items():
                if count > major_count:
                    major_shape = shape
            del shape, count

            # Check which tensors don't have the major shape!
            extra_msg = f" \n*** Major shape: {major_shape}"
            for (name, param), tensor in zip(list(self.named_params), norm_sample):
                if tensor.size() != major_shape:
                    extra_msg += f", {name} wrong shape: {tensor.size()}"
            extra_msg += " ***"

            runtime_error.args = (args[0] + extra_msg, *args[1:])
            raise runtime_error

        coef_sample = torch.clamp_max(
            self.max_grad_norm * scale / (norm_sample + self.numerical_stability_constant), 1.
        )
        for name, param in self.named_params:
            if not hasattr(param, 'summed_grad'):
                param.summed_grad = 0.
            current_device = param.grad_sample.device
            param.summed_grad += torch.einsum("i,i...->...", coef_sample.to(current_device), param.grad_sample)
        return norm_sample, coef_sample

    def get_privacy_spent(self, steps=None, accounting_mode=None, lenient=False) -> Dict:
        if steps is None:
            steps = self.steps
        if accounting_mode is None:
            accounting_mode = self.accounting_mode

        privacy_results = {}

        kwargs = dict(
            sample_rate=self.sample_rate,
            steps=steps,
            delta=self.target_delta,
            sigma=self.noise_multiplier,
            alphas=self.alphas,
        )
        # The try-catch blocks are unusually sloppy... forgive me...
        if accounting_mode in ('all', 'rdp'):
            try:
                eps_rdp, alpha_rdp = _eps_from_rdp(**kwargs)
                privacy_results['eps_rdp_opacus'] = eps_rdp
                privacy_results['alpha_rdp_opacus'] = alpha_rdp
            except Exception as err:
                logging.fatal("RDP accounting failed! Double check privacy parameters.")
                if not lenient:
                    raise err

        if accounting_mode in ('all', 'gdp'):
            try:
                eps_gdp, mu_gdp = _eps_from_gdp(**kwargs)
                privacy_results['eps_gdp'] = eps_gdp
                privacy_results['mu_gdp'] = mu_gdp
            except Exception as err:
                logging.fatal("GDP accounting failed! Double check privacy parameters.")
                if not lenient:
                    raise err

        if accounting_mode in ('all', "rdp_cks"):
            try:
                eps_rdp, alpha_rdp = _eps_from_rdp_cks(**kwargs)
                privacy_results['eps_rdp'] = eps_rdp
                privacy_results['alpha_rdp'] = alpha_rdp
            except Exception as err:
                logging.fatal("RDP accounting with CKS conversion failed! "
                              "Double check privacy parameters.")
                if not lenient:
                    raise err

        if accounting_mode in ('all', "glw"):
            try:
                eps_glw = _eps_from_glw(**kwargs)
                privacy_results.update(eps_glw)
            except Exception as err:
                logging.fatal("Numerical composition of tradeoff functions failed! "
                              "Double check privacy parameters.")
                if not lenient:
                    raise err

        return privacy_results

    def get_training_stats(self):
        """Get the clipping, signal, and noise statistics."""
        return {
            "med_clip": self.med_clip,
            "max_clip": self.max_clip,
            "min_clip": self.min_clip,
            "snr": self.snr,
            "signal": self.signal,
            "noise": self.noise,
            "noise_limit": self.noise_limit,
        }

    def __repr__(self):
        return (
            f"PrivacyEngine(\n"
            f"  target_epsilon={self.target_epsilon}, \n"
            f"  target_delta={self.target_delta}, \n"
            f"  noise_multiplier={self.noise_multiplier}, \n"
            f"  effective_noise_multiplier={self.effective_noise_multiplier}, \n"
            f"  epochs={self.epochs}, \n"
            f"  max_grad_norm={self.max_grad_norm}, \n"
            f"  sample_rate={self.sample_rate}, \n"
            f"  batch_size={self.batch_size}, \n"
            f"  accounting_mode={self.accounting_mode}, \n"
            f"  ghost_clipping={self.ghost_clipping}\n"
            f")"
        )


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
    """Get noise multiplier σ for a given ε from Renyi-DP accounting.

    Notes:
        Setting `threshold` to an appropriate value is crucial for accurate search.
        The default is fine-grained enough for ε ∈ [0.1, 1e10].

    Args:
        target_epsilon: ε in (ε, δ)-DP.
        target_delta: δ in (ε, δ)-DP.
        sample_rate: Rate for Poisson subsampling, typically denoted as q.
        epochs: Number of passes through the dataset.
        alphas: Orders for Renyi-divergence.
        threshold: Threshold for binary search. Determines the granularity of
            the search result.
        sigma_hi_init: Starting point for the high end of binary search.
        sigma_lo_init: Starting point for the low end of binary search.
        steps: Number of updates; defaults to use `epochs` if not set.

    Returns:
        The noise multiplier σ for DP-SGD.
    """
    if steps is None:
        if epochs is None:
            raise ValueError("Epochs and steps cannot both be None")
        steps = math.ceil(epochs / sample_rate)  # Be conservative!

    def sigma_to_eps(sigma):
        eps, _ = _eps_from_rdp(
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


def get_sigma_from_rdp_cks(
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
    if steps is None:
        if epochs is None:
            raise ValueError("Epochs and steps cannot both be None")
        steps = math.ceil(epochs / sample_rate)

    def sigma_to_eps(sigma):
        eps, _ = _eps_from_rdp_cks(
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


def get_sigma_from_gdp(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[Union[float, int]] = None,
    threshold=1e-3,
    sigma_hi_init=4,
    sigma_lo_init=0.1,
    mode="poisson",
    steps=None,
    **kwargs,
) -> float:
    """Get noise multiplier σ for a given ε from f-DP accounting using the central limit theorem in Gaussian DP."""
    if steps is None:
        if epochs is None:
            raise ValueError("Epochs and steps cannot both be None")
        steps = math.ceil(epochs / sample_rate)

    def sigma_to_eps(sigma):
        eps, _ = _eps_from_gdp(
            sample_rate=sample_rate,
            sigma=sigma,
            steps=steps,
            delta=target_delta,
            mode=mode,
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


def _eps_from_rdp(
    sample_rate,
    sigma,
    steps,
    delta,
    alphas=DEFAULT_ALPHAS,
    **_,
):
    """Get the ε in (ε, δ)-DP from Renyi-DP accounting."""
    # This is based on Poisson sampling in https://arxiv.org/pdf/1908.10530.pdf
    rdp = rdp_accounting.compute_rdp(
        q=sample_rate, noise_multiplier=sigma, steps=steps, orders=alphas
    )
    # (ε, α)
    eps, alpha = rdp_accounting.get_privacy_spent(
        orders=alphas, rdp=rdp, delta=delta
    )
    return eps, alpha


def _compute_eps_cks(orders, rdp, delta):
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


def _eps_from_rdp_cks(
    sample_rate,
    sigma,
    steps,
    delta,
    alphas=DEFAULT_ALPHAS,
    **_,
):
    """Compute RDP as usual, but the conversion to (ε, δ)-DP is based on result by Canonne, Kamath, Steinke.

    # @formatter:off
    Code from https://github.com/tensorflow/privacy/blob/5f07198b66b3617b22609db983926e3ba97cd905/tensorflow_privacy/privacy/analysis/rdp_accountant.py#L237
    # @formatter:on
    """
    rdp = rdp_accounting.compute_rdp(
        q=sample_rate, noise_multiplier=sigma, steps=steps, orders=alphas
    )
    # (ε, α)
    eps, alpha = _compute_eps_cks(orders=alphas, rdp=rdp, delta=delta)
    return eps, alpha


def _eps_from_gdp(
    sample_rate,
    sigma,
    steps,
    delta,
    mode="poisson",
    **_,
):
    """Get the ε in (ε, δ)-DP from f-DP accounting."""
    epochs = steps * sample_rate
    if mode == "poisson":
        eps_fn = gdp_accounting.compute_eps_poisson
        mu_fn = gdp_accounting.compute_mu_poisson
    else:
        eps_fn = gdp_accounting.compute_eps_uniform
        mu_fn = gdp_accounting.compute_mu_uniform

    eps = eps_fn(
        epochs=epochs,
        noise_multi=sigma,
        delta=delta,
        sample_rate=sample_rate,
    )
    mu = mu_fn(
        epochs=epochs,
        noise_multi=sigma,
        sample_rate=sample_rate,
    )
    return eps, mu


def _eps_from_glw(
    sample_rate,
    sigma,
    steps,
    delta,
    eps_error=0.05,
    **_,
):
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
