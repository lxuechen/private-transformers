"""
A large portion of this code is adapted from Opacus (https://github.com/pytorch/opacus),
which is licensed under Apache License 2.0.

We have modified it considerably to support ghost clipping.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .settings import BackwardHookMode
from .supported_layers_grad_samplers import _supported_layers_grad_samplers

# TODO: hooks mode should be settable based on the module.
_hooks_disabled: bool = False
_hooks_mode = BackwardHookMode.default


def set_hooks_mode(mode):
    if mode not in BackwardHookMode.all():
        raise ValueError(f"Unknown mode for hooks: {mode}; expected one of {BackwardHookMode.all()}.")

    global _hooks_mode
    _hooks_mode = mode  # Set mode.

    if _hooks_mode == BackwardHookMode.ghost_grad:  # Second backward pass of ghost clipping doesn't need hooks.
        disable_hooks()
    elif _hooks_mode == BackwardHookMode.ghost_norm:  # First backward pass of ghost clipping needs to accumulate norms.
        enable_hooks()


def get_hooks_mode():
    global _hooks_mode
    return _hooks_mode


def requires_grad(module: nn.Module, recurse: bool = False) -> bool:
    """
    Checks if any parameters in a specified module require gradients.

    Args:
        module: PyTorch module whose parameters are examined
        recurse: Flag specifying if the gradient requirement check should
            be applied recursively to sub-modules of the specified module

    Returns:
        Flag indicate if any parameters require gradients
    """
    return any(p.requires_grad for p in module.parameters(recurse))


def add_hooks(model: nn.Module, loss_reduction: str = "mean"):
    r"""
    Adds hooks to model to save activations and backprop values.
    The hooks will

    1. save activations into ``param.activations`` during forward pass.
    2. compute per-sample gradients and save them in ``param.grad_sample`` during backward pass.

    Args:
        model: Model to which hooks are added.
        loss_reduction: Indicates if the loss reduction (for aggregating the gradients) is a sum or a mean operation.
            Can take values ``sum`` or ``mean``.
    """
    if hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Trying to add hooks twice to the same model")

    enable_hooks()

    handles = []
    for name, layer in model.named_modules():
        if type(layer) in _supported_layers_grad_samplers:
            if requires_grad(layer, recurse=False):
                handles.append(layer.register_forward_hook(_capture_activations))

                def this_backward(this_layer, grad_input, grad_output):
                    return _capture_backprops(this_layer, grad_input, grad_output, loss_reduction)

                # Starting with 1.8.0, use `register_full_backward_hook`.
                handles.append(layer.register_backward_hook(this_backward))

    model.__dict__.setdefault("autograd_grad_sample_hooks", []).extend(handles)


def remove_hooks(model: nn.Module):
    """Removes hooks added by `add_hooks()`."""
    if not hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_grad_sample_hooks:
            handle.remove()
        del model.autograd_grad_sample_hooks


def disable_hooks():
    """Globally disables all hooks installed by this library."""
    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks():
    """Globally enables all hooks installed by this library."""
    global _hooks_disabled
    _hooks_disabled = False


def _capture_activations(layer: nn.Module, inputs: Tuple, outputs: Tuple):
    """Forward hook handler captures and saves activations."""
    if not requires_grad(layer) or not layer.training or _hooks_disabled:
        return

    if not hasattr(layer, "activations"):
        layer.activations = []

    # This improves on original Opacus and supports additional arguments on top of the (first) activation tensor.
    stored_inputs = tuple(input_i.detach() if torch.is_tensor(input_i) else input_i for input_i in inputs)
    layer.activations.append(stored_inputs)


def _capture_backprops(
    layer: nn.Module,
    inputs: Tuple[torch.Tensor],
    outputs: Tuple[torch.Tensor],
    loss_reduction: str
):
    """Backward hook handler captures grad_outputs."""
    # This improves on the original Opacus codebase and supports multiple outputs.
    backprops = tuple(output_i.detach() if torch.is_tensor(output_i) else output_i for output_i in outputs)
    _compute_grad_sample(layer, backprops, loss_reduction)


def _compute_grad_sample(layer: nn.Module, backprops: Tuple, loss_reduction: str):
    """Computes per-sample gradients with respect to the parameters."""
    if not requires_grad(layer) or not layer.training or _hooks_disabled:
        return

    if not hasattr(layer, "activations"):
        raise ValueError(f"No activations detected for {type(layer)}, run forward after add_hooks(model)")

    # Outside of the LSTM there is "batch_first" but not for the Linear inside the LSTM
    if isinstance(layer.activations, list):
        A = layer.activations.pop()
    else:
        A = layer.activations

    if not hasattr(layer, "max_batch_len"):
        assert torch.is_tensor(A[0]), f"Internal error: first input of the following layer isn't a Tensor. \n{layer}"
        layer.max_batch_len = _get_batch_size(layer, A[0])

    n = layer.max_batch_len
    if loss_reduction == "mean":
        B = tuple(B_i * n if torch.is_tensor(B_i) else B_i for B_i in backprops)
    elif loss_reduction == "sum":
        B = backprops
    else:
        raise ValueError(f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported")

    # compute grad sample for individual layers
    compute_layer_grad_sample = _supported_layers_grad_samplers.get(type(layer))
    compute_layer_grad_sample(layer, A, B)

    if (not isinstance(layer.activations, list) or len(layer.activations) == 0) and hasattr(layer, "max_batch_len"):
        del layer.max_batch_len


def _get_batch_size(layer: nn.Module, grad_sample: torch.Tensor) -> int:
    r"""
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over layer.activations + [grad_sample], where layer.activations is
    a list. If layer.activations is a not a list, then return grad_sample.shape[batch_dim].
    """

    batch_dim = 0
    max_batch_len = 0
    if isinstance(layer.activations, list):
        for out in layer.activations:
            assert torch.is_tensor(out[0]), (
                f"Internal error: first input of the following layer isn't a Tensor. \n{layer}"
            )
            if out[0].shape[batch_dim] > max_batch_len:
                max_batch_len = out[0].shape[batch_dim]

    max_batch_len = max(max_batch_len, grad_sample.shape[batch_dim])
    return max_batch_len
