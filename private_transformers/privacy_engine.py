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

import torch
from ml_swissknife import utils
from torch import nn

from . import autograd_grad_sample, transformers_support
from .accounting import accounting_manager
from .settings import AccountingMode, BackwardHookMode, ClippingMode, SUPPORTED_TRANSFORMERS


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
        alphas: Sequence[float] = accounting_manager.DEFAULT_ALPHAS,
        record_snr: bool = True,
        named_params: Optional[Sequence] = None,
        numerical_stability_constant=1e-6,
        clipping_mode=ClippingMode.default,
        accounting_mode="rdp",
        eps_error=0.05,
        skip_checks=False,
        **unused_kwargs,
    ):
        """Initialize the engine.

        Args:
            module: The PyTorch module for which per-sample gradient is required.
                Setting the `requires_grad` attribute of a parameter to False
                disables the per-sample gradient accumulation.
            batch_size: The expected size of Poisson-sampled batch, i.e., the lot size.
            sample_size: Size of dataset.
            max_grad_norm: The maximum 2-norm for gradient clipping.
            epochs: The number of epochs for training.
            noise_multiplier: The extra multiplier for DP-SGD noise.
            target_epsilon: The target privacy spending.
                Only used to estimate the `noise_multiplier` if it is not set.
            target_delta: The target failure probability.
                Defaults to sample_size ** -1.1 if not set.
            alphas: The RDP orders for (ε, δ)-DP conversion. Useless if not accounting in RDP.
            record_snr: Record and report the signal-to-noise ratio --
                ratio between norm of summed clipped gradient and norm of noise vector.
            named_params: Specifies which parameters need gradients;
                defaults to use parameters which require grad in module.
            numerical_stability_constant: Small constant to avoid division by 0 when clipping.
            clipping_mode: The clipping mode to use. One of 'default', 'ghost', 'per_layer', 'per_layer_percentile'.
            accounting_mode: The method of accounting privacy. One of (`rdp`, `glw`, `all`).
                Meanings of shorthands:
                    - rdp: Account loss with RDP but perform conversion to approx-DP with a procedure defined in
                        "The Discrete Gaussian for Differential Privacy". https://arxiv.org/abs/2004.00010
                    - glw: Account loss by numerically composing tradeoff functions in f-DP; defined in
                        "Numerical composition of differential privacy". https://arxiv.org/abs/2106.02848
                    - all: Report loss with all methods listed above.
            eps_error: Error threshold for upper and lower bound in the GLW accounting procedure.
            skip_checks: Skips the model type validation test if True.
        """
        utils.handle_unused_kwargs(unused_kwargs)
        del unused_kwargs
        super(PrivacyEngine, self).__init__()

        if clipping_mode not in ClippingMode.all():
            raise ValueError(f"Unknown clipping mode {clipping_mode}. Expected one of {ClippingMode.all()}.")
        if accounting_mode not in AccountingMode.all():
            raise ValueError(f"Unknown accounting mode: {accounting_mode}. Expected one of {AccountingMode.all()}.")
        if epochs <= 0.0:
            raise ValueError(f"Number of training epochs cannot be non-positive, but found epochs={epochs}")

        # Privacy parameters.
        sample_rate = batch_size / sample_size
        if target_delta is None:
            target_delta = sample_size ** -1.1
        if noise_multiplier is None:
            if target_epsilon is None or epochs is None:
                raise ValueError(
                    f"`target_epsilon` and `epochs` must be specified when `noise_multiplier` is `None`."
                )
            if accounting_mode in ("rdp", "all"):
                manager = accounting_manager.RDPManager(alphas=alphas)
            else:  # "glw"
                manager = accounting_manager.GLWManager(eps_error=eps_error)
            noise_multiplier = manager.compute_sigma(
                target_epsilon=target_epsilon, target_delta=target_delta, sample_rate=sample_rate, epochs=epochs,
            )

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
        self.eps_error = eps_error
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
        self.numerical_stability_constant = numerical_stability_constant
        self.clipping_mode = clipping_mode
        if clipping_mode == ClippingMode.ghost:
            autograd_grad_sample.set_hooks_mode(BackwardHookMode.ghost_norm)  # Prepare for first backward.
        else:
            autograd_grad_sample.set_hooks_mode(BackwardHookMode.default)  # Extra guard.

        if not isinstance(module, SUPPORTED_TRANSFORMERS) and not skip_checks:
            raise ValueError(
                f"Model type {type(module)} is not supported. Please file an issue if you want this model to be added.\n"
                f"Currently supported transformers are: {SUPPORTED_TRANSFORMERS}"
            )
        transformers_support.forward_swapper(module=module)  # Fix the position embeddings broadcast issue.

    def lock(self):
        """Run this after noisy clipped gradient is created to prevent tampering with it before parameter update."""
        self._locked = True

    def unlock(self):
        """Run this after parameter update to allow creation of noisy gradient for next step"""
        self._locked = False

    def attach(self, optimizer):
        # `loss_reduction="sum"` super important.
        autograd_grad_sample.add_hooks(model=self.module, loss_reduction="sum")

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
            return _self.privacy_engine.get_privacy_spent(**kwargs)

        def get_training_stats(_self, **kwargs):
            return _self.privacy_engine.get_training_stats(**kwargs)

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
        module.zero_grad(skip_grad=True)  # noqa
        module.zero_grad = module.original_zero_grad
        delattr(module, "original_zero_grad")

    @torch.no_grad()
    def step(
        self,
        loss: torch.Tensor,
        scale=1.,
        # Function that takes in named_params and does something.
        # This option was included to help with another spectrum analysis project.
        callback: Optional[Callable] = None,
    ):
        if loss.dim() != 1:
            raise ValueError(
                f"Expected `loss` to be the per-example loss 1-D tensor, but got a tensor with dims={loss.dim()}."
            )

        if self.clipping_mode == ClippingMode.ghost:
            if callback is not None:
                raise ValueError("Ghost clipping does not support `callback` in `optimizer.step`.")
            if scale != 1.:
                raise ValueError("Ghost clipping does not support mixed-precision training.")
            self._ghost_step(loss=loss)
        else:
            self._step(loss=loss, scale=scale, callback=callback)

    @torch.no_grad()
    def virtual_step(self, loss: torch.Tensor, scale=1.):
        """Virtual step function when there's gradient accumulation."""
        if self.clipping_mode == ClippingMode.ghost:
            self._ghost_virtual_step(loss=loss)
        else:
            self._virtual_step(loss=loss, scale=scale)

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

    def _create_noisy_clipped_gradient(self):
        """Create noisy clipped gradient for `optimizer.step`.

        Add noise and scale by inverse batch size.

        Notes:
            In ghost clipping, `summed_grad` stores previous micro-batches; `grad` stores current micro-batch.
            In default clipping, `summed_grad` stores summed clipped gradients for all micro-batches.
        """

        signals, noises = [], []
        for name, param in self.named_params:
            assert hasattr(param, 'summed_grad'), (
                f"Internal error: PrivacyEngine should not reach here; "
                f"this means either "
                f"1) there is parameter which requires gradient, but was not used in the computational graph, "
                f"or 2) the backward hook registry failed to find the corresponding module to register."
            )
            param.grad = param.summed_grad  # Ultra important to override `.grad`.

            if self.record_snr:
                signals.append(param.grad.reshape(-1).norm(2))

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
                    noises.append(noise.reshape(-1).norm(2))
                del noise

            param.grad /= self.batch_size

        if self.record_snr and len(noises) > 0:
            self.signal, self.noise = tuple(torch.stack(lst).norm(2).item() for lst in (signals, noises))
            self.noise_limit = math.sqrt(self.num_params) * self.noise_multiplier * self.max_grad_norm
            self.snr = self.signal / self.noise
        else:
            self.snr = math.inf  # Undefined!

        self.lock()  # Make creating new gradients impossible, unless optimizer.step is called.

    # --- ghost clipping ---
    def _ghost_step(self, loss: torch.Tensor):
        """Run double-backward on per-example loss, then sum up all gradients and noise it."""
        if self._locked:  # Skip this gradient creation step if already created gradient and haven't stepped.
            logging.warning("Attempted to step, but the engine is on lock.")
            return

        self._ghost_virtual_step(loss)
        self._create_noisy_clipped_gradient()

    @torch.no_grad()
    def _ghost_virtual_step(self, loss: torch.Tensor):
        """Backward twice to accumulate summed clipped gradients in `.summed_grad`.

        We accumulate gradients in `.summed_grad` for micro-batching.
        All of this copying actually creates a new 2x memory overhead.
        """
        self._double_backward(loss)

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
    def _double_backward(self, loss: torch.Tensor):
        """Given per-example losses, backward twice to accumulate summed clipped gradients in `.grad`."""
        first_loss = loss.sum()
        first_loss.backward(retain_graph=True)

        # Prepare for second backward.
        autograd_grad_sample.set_hooks_mode(BackwardHookMode.ghost_grad)

        # The first backward might have accumulated things we don't need into `.grad`;
        # remove it before the second pass to avoid accumulating garbage.
        for name, param in self.named_params:
            if hasattr(param, "grad"):
                del param.grad

        coef_sample = self.get_coef_sample()
        second_loss = (coef_sample * loss).sum(dim=0)
        second_loss.backward()

        # Prepare for first backward (in the next round).
        autograd_grad_sample.set_hooks_mode(BackwardHookMode.ghost_norm)

    def get_coef_sample(self) -> torch.Tensor:
        """Get per-example gradient scaling factor for clipping."""
        norm_sample = self.get_norm_sample()
        return torch.clamp_max(self.max_grad_norm / (norm_sample + self.numerical_stability_constant), 1.)

    def get_norm_sample(self) -> torch.Tensor:
        """Get per-example gradient norms."""
        norm_sample = torch.stack([param.norm_sample for name, param in self.named_params], dim=0).norm(2, dim=0)
        return norm_sample

    # --- default clipping ---
    def _step(
        self,
        loss,
        scale,
        callback,
    ):
        """Create noisy gradients.

        Should be run right before you call `optimizer.step`.

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

        if callback is not None:
            callback(self)
        self._create_noisy_clipped_gradient()

    def _virtual_step(self, loss, scale):
        self._accumulate_summed_grad(loss=loss, scale=scale)

    @torch.no_grad()
    def _accumulate_summed_grad(self, loss, scale):
        """Accumulate signal by summing clipped gradients.

        Removes `.grad_sample` and `.grad` for each variable that requires grad at the end.
        """
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
            norm_sample.append(norm)

        # The stack operation here is prone to error, thus clarify where the error is.
        try:
            norm_sample = torch.stack(norm_sample, dim=0).norm(2, dim=0)
        except RuntimeError as runtime_error:
            args = runtime_error.args

            # Get the major shape.
            shapes = collections.defaultdict(int)
            for tensor in norm_sample:
                shapes[tensor.size()] += 1

            # Get the shape that most tensors have.
            major_shape, major_count = max(shapes.items(), key=lambda x: x[1])

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

            # Aggressive memory saving -- delete everything except `.summed_grad` to save memory!
            if hasattr(param, "grad_sample"):
                # This must be deleted due to how `privacy_utils::supported_layers_grad_samplers.py` works!
                #   When a parameter with `.grad_sample` is reused, the per-sample gradients are accumulated!
                del param.grad_sample
            if hasattr(param, "grad"):
                del param.grad

        return norm_sample, coef_sample

    def get_privacy_spent(
        self,
        steps: Optional[int] = None,
        accounting_mode: Optional[str] = None,
        lenient=False
    ) -> Dict:
        if steps is None:
            steps = self.steps
        if accounting_mode is None:
            accounting_mode = self.accounting_mode

        privacy_results = {}  # Contains stats from all modes.
        if accounting_mode in (AccountingMode.all_, AccountingMode.rdp):
            try:
                manager = accounting_manager.RDPManager(alphas=self.alphas)
                privacy_results.update(
                    manager.compute_epsilon(
                        sigma=self.noise_multiplier,
                        sample_rate=self.sample_rate,
                        target_delta=self.target_delta,
                        steps=steps,
                    )
                )
            except Exception as err:
                logging.fatal("RDP accounting failed! Double check privacy parameters.")
                if not lenient:
                    raise err

        if accounting_mode in (AccountingMode.all_, AccountingMode.glw):
            try:
                manager = accounting_manager.GLWManager(eps_error=self.eps_error)
                privacy_results.update(
                    manager.compute_epsilon(
                        sigma=self.noise_multiplier,
                        sample_rate=self.sample_rate,
                        target_delta=self.target_delta,
                        steps=steps
                    )
                )
            except Exception as err:
                logging.fatal(
                    "Numerical composition of tradeoff functions failed! Double check privacy parameters."
                )
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
            f"  target_epsilon={self.target_epsilon:.6f}, \n"
            f"  target_delta={self.target_delta:.6f}, \n"
            f"  noise_multiplier={self.noise_multiplier:.6f}, \n"
            f"  effective_noise_multiplier={self.effective_noise_multiplier:.6f}, \n"
            f"  epochs={self.epochs}, \n"
            f"  max_grad_norm={self.max_grad_norm}, \n"
            f"  sample_rate={self.sample_rate}, \n"
            f"  batch_size={self.batch_size}, \n"
            f"  accounting_mode={self.accounting_mode}, \n"
            f"  clipping_mode={self.clipping_mode}\n"
            f")"
        )
