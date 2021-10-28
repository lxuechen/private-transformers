"""
A large portion of this code is adapted from Opacus (https://github.com/pytorch/opacus),
which is licensed under Apache License 2.0.

We have modified it considerably to support ghost clipping.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .supported_layers_grad_samplers import _supported_layers_grad_samplers

# work-around for https://github.com/pytorch/pytorch/issues/25723
_hooks_disabled: bool = False
_fp16 = False
_hooks_mode = "default"


def set_hooks_mode(mode):
    if mode not in ("ghost_norm", "ghost_grad", "default"):
        raise ValueError(f"Unknown mode for hooks: {mode}; expected one of `ghost_norm`, `ghost_grad`, `default`.")

    global _hooks_mode
    _hooks_mode = mode

    if _hooks_mode == "ghost_grad":
        disable_hooks()
    elif _hooks_mode == "ghost_norm":
        enable_hooks()


def get_hooks_mode():
    global _hooks_mode
    return _hooks_mode


def enable_fp16():
    global _fp16
    _fp16 = True


def disable_fp16():
    global _fp16
    _fp16 = False


def fp16():
    return _fp16


def has_no_param(module: nn.Module) -> bool:
    """
    Checks if a module does not have any parameters.

    Args:
        module: The module on which this function is being evaluated.

    Returns:
        Flag indicating if the provided module does not have any
        parameters.
    """
    has_params = any(p is not None for p in module.parameters(recurse=False))
    return not has_params


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
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def get_layer_type(layer: nn.Module) -> str:
    """
    Returns the name of the type of the given layer.

    Args:
        layer: The module corresponding to the layer whose type
            is being queried.

    Returns:
        Name of the class of the layer
    """
    return layer.__class__.__name__


def add_hooks(
    model: nn.Module,
    loss_reduction: str = "mean",
    batch_first: bool = True,
    fp16=False,
):
    r"""
    Adds hooks to model to save activations and backprop values.
    The hooks will

    1. save activations into ``param.activations`` during forward pass.
    2. compute per-sample gradients and save them in ``param.grad_sample`` during backward pass.

    Args:
        model: Model to which hooks are added.
        loss_reduction: Indicates if the loss reduction (for aggregating the
            gradients) is a sum or a mean operation. Can take values ``sum`` or
            ``mean``.
        batch_first: Flag to indicate if the input tensor to the corresponding module
            has the first dimension represent the batch, for example of shape
            ``[batch_size, ..., ...]``. Set to True if batch appears in first
            dimension else set to False (``batch_first=False`` implies that the
            batch is always in the second dimension).
        fp16: Perform backward computation in fp16 for as much as possible if True.
    """
    if hasattr(model, "autograd_grad_sample_hooks"):
        raise ValueError("Trying to add hooks twice to the same model")

    enable_hooks()
    if fp16:
        enable_fp16()

    handles = []
    for name, layer in model.named_modules():
        if get_layer_type(layer) in _supported_layers_grad_samplers.keys():
            # Check if the layer has trainable parameters.
            is_trainable = False
            for p in layer.parameters(recurse=False):
                if p.requires_grad:
                    is_trainable = True
                    break

            if is_trainable:
                handles.append(layer.register_forward_hook(_capture_activations))

                def this_backward(this_layer, grad_input, grad_output):
                    return _capture_backprops(
                        this_layer, grad_input, grad_output, loss_reduction, batch_first
                    )

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


def is_supported(layer: nn.Module) -> bool:
    """Checks if the layer is supported by this library."""
    return get_layer_type(layer) in list(_supported_layers_grad_samplers.keys())


def _capture_activations(layer: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
    """Forward hook handler captures and saves activations."""
    layer_type = get_layer_type(layer)
    if (
        not requires_grad(layer)
        or layer_type not in _supported_layers_grad_samplers.keys()
        or not layer.training
    ):
        return

    if _hooks_disabled:
        return
    if get_layer_type(layer) not in _supported_layers_grad_samplers.keys():
        raise ValueError("Hook installed on unsupported layer")

    if not hasattr(layer, "activations"):
        layer.activations = []

    layer.activations.append(inputs[0].detach())


def _capture_backprops(
    layer: nn.Module,
    inputs: Tuple[torch.Tensor],
    outputs: Tuple[torch.Tensor],
    loss_reduction: str,
    batch_first: bool,
):
    """Backward hook handler captures grad_outputs."""
    if _hooks_disabled:
        return

    backprops = outputs[0].detach()
    _compute_grad_sample(layer, backprops, loss_reduction, batch_first)


def _compute_grad_sample(layer: nn.Module, backprops: torch.Tensor, loss_reduction: str, batch_first: bool):
    """Computes per-sample gradients with respect to the parameters."""
    layer_type = get_layer_type(layer)
    if (
        not requires_grad(layer)
        or layer_type not in _supported_layers_grad_samplers.keys()
        or not layer.training
    ):
        return

    if not hasattr(layer, "activations"):
        raise ValueError(
            f"No activations detected for {type(layer)},"
            " run forward after add_hooks(model)"
        )

    # Outside of the LSTM there is "batch_first" but not for the Linear inside the LSTM
    batch_dim = 0 if batch_first else 1
    if isinstance(layer.activations, list):
        A = layer.activations.pop()
    else:
        A = layer.activations

    if not hasattr(layer, "max_batch_len"):
        layer.max_batch_len = _get_batch_size(layer, A, batch_dim)

    n = layer.max_batch_len
    if loss_reduction == "mean":
        B = backprops * n
    elif loss_reduction == "sum":
        B = backprops
    else:
        raise ValueError(
            f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported"
        )

    # rearrange the blob dimensions
    if batch_dim != 0:
        A = A.permute([batch_dim] + [x for x in range(A.dim()) if x != batch_dim])
        B = B.permute([batch_dim] + [x for x in range(B.dim()) if x != batch_dim])
    # compute grad sample for  individual layers
    compute_layer_grad_sample = _supported_layers_grad_samplers.get(
        get_layer_type(layer)
    )

    compute_layer_grad_sample(layer, A, B)

    if (
        not isinstance(layer.activations, list) or len(layer.activations) == 0
    ) and hasattr(layer, "max_batch_len"):
        del layer.max_batch_len


def _get_batch_size(layer: nn.Module, grad_sample: torch.Tensor, batch_dim: int) -> int:
    r"""
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over layer.activations + [grad_sample], where layer.activations is
    a list. If layer.activations is a not a list, then return grad_sample.shape[batch_dim].
    """

    max_batch_len = 0
    if isinstance(layer.activations, list):
        for out in layer.activations:
            if out.shape[batch_dim] > max_batch_len:
                max_batch_len = out.shape[batch_dim]

    max_batch_len = max(max_batch_len, grad_sample.shape[batch_dim])
    return max_batch_len
