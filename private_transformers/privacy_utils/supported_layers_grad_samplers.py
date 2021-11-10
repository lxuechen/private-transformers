"""
This module is a collection of grad samplers - methods to calculate per sample gradients
for a layer given two tensors: 1) inputs, 2) grad_outputs.

Supports ghost clipping introduced in
Li, X., Tramèr, F., Liang, P., & Hashimoto, T. (2021).
Large Language Models Can Be Strong Differentially Private Learners. arXiv preprint arXiv:2110.05679.

A large portion of this code is adapted from Opacus (https://github.com/pytorch/opacus),
which is licensed under Apache License 2.0.
"""

import torch
from torch import nn
from torch.functional import F

from . import autograd_grad_sample


def sum_over_all_but_batch_and_last_n(tensor: torch.Tensor, n_dims: int) -> torch.Tensor:
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)


def _light_linear_weight_norm_sample(A, B) -> torch.Tensor:
    """Compute gradient sample norm for the weight matrix in a linear layer."""
    if A.dim() == 2:
        return _light_linear_weight_norm_sample_non_sequential(A, B)
    elif A.dim() == 3:
        return _light_linear_weight_norm_sample_sequential(A, B)
    else:
        raise ValueError(f"Unexpected input shape: {A.size()}, grad_output shape: {B.size()}")


def _light_linear_weight_norm_sample_sequential(A, B):
    """Lightweight norm computation in ghost clipping."""
    return torch.sqrt(
        (torch.bmm(A, A.transpose(-1, -2)) * torch.bmm(B, B.transpose(-1, -2))).sum(dim=(1, 2))
    )


def _light_linear_weight_norm_sample_non_sequential(A, B):
    """The Goodfellow trick, i.e., Frobenius norm equal to product of 2-norms."""
    return A.norm(2, dim=1) * B.norm(2, dim=1)


def _light_linear_bias_norm_sample(B):
    if B.dim() == 2:
        return B.norm(2, dim=1)
    elif B.dim() == 3:
        return B.sum(dim=1).norm(2, dim=1)
    else:
        raise ValueError(f"Unexpected grad_output shape: {B.size()}")


def _create_or_extend_grad_sample(param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int) -> None:
    """Creates a ``grad_sample`` attribute in the given parameter or accumulate the existing tensor."""
    if hasattr(param, "requires_grad") and not param.requires_grad:
        return

    assert grad_sample.shape[1:] == param.shape, f"grad_sample.size()={grad_sample.size()}, param.size()={param.size()}"

    # Warning: When a parameter with `grad_sample` is reused, the per-sample gradients are accumulated.
    if hasattr(param, "grad_sample"):
        param.grad_sample += grad_sample.detach()
    else:
        param.grad_sample = grad_sample.detach()


def _create_or_extend_norm_sample(param: torch.Tensor, norm_sample: torch.Tensor) -> None:
    """Creates a ``norm_sample`` attribute in the given parameter."""
    if not hasattr(param, "requires_grad") or not param.requires_grad:
        return

    if autograd_grad_sample.get_hooks_mode() == "ghost_norm":
        if hasattr(param, 'norm_sample'):
            raise ValueError("Ghost clipping does not support parameter sharing. "
                             "Parameter sharing may be due to default parameter sharing between lm_head and embedding."
                             "Please use a model without parameter sharing for ghost clipping.")
        param.norm_sample = norm_sample
    else:  # mode == "grad"; should not get here.
        raise ValueError("Internal error: Trying to extend `norm_sample` when `_hooks_mode='ghost_grad'`.")


def _compute_linear_grad_sample(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0) -> None:
    """Computes per sample gradients for `nn.Linear` layer.

    This function is written in an unusually bespoke way to avoid using `torch.einsum`.
    """
    if autograd_grad_sample.fp16():
        if B.dtype != torch.half:
            B = B.half()
        if A.dtype != torch.half:
            A = A.half()

    if autograd_grad_sample.get_hooks_mode() == "ghost_norm":
        _create_or_extend_norm_sample(layer.weight, _light_linear_weight_norm_sample(A, B))

        if layer.bias is not None:
            _create_or_extend_norm_sample(layer.bias, _light_linear_bias_norm_sample(B))
    else:
        if B.dim() == 3 and A.dim() == 3:
            grad_weight = torch.bmm(B.permute(0, 2, 1), A)
            grad_bias = B.sum(dim=1)
        elif B.dim() == 2 and A.dim() == 2:
            grad_weight = B[:, :, None] * A[:, None, :]
            grad_bias = B
        else:
            raise ValueError(
                f"Expected both grad_output and input to have dimension 2 or 3, "
                f"but found len(grad_output.dim())={len(B.dim())}, len(input.dim())={len(A.dim())}"
            )
        _create_or_extend_grad_sample(layer.weight, grad_weight, batch_dim)

        if layer.bias is not None:
            _create_or_extend_grad_sample(layer.bias, grad_bias, batch_dim)


def _compute_norm_grad_sample(
    layer: nn.LayerNorm,
    A: torch.Tensor,
    B: torch.Tensor,
    batch_dim: int = 0,
) -> None:
    """Computes per sample gradients for normalization layers."""
    if autograd_grad_sample.fp16():
        if A.dtype != torch.half:
            A = A.half()
        if B.dtype != torch.half:
            B = B.half()

    is_backward_ghost_norm = autograd_grad_sample.get_hooks_mode() == "ghost_norm"

    grad_sample = sum_over_all_but_batch_and_last_n(
        F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
        layer.weight.dim(),
    )
    if is_backward_ghost_norm:
        norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
        _create_or_extend_norm_sample(layer.weight, norm_sample)
    else:
        _create_or_extend_grad_sample(layer.weight, grad_sample, batch_dim)

    grad_sample = sum_over_all_but_batch_and_last_n(B, layer.bias.dim())
    if is_backward_ghost_norm:
        norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
        _create_or_extend_norm_sample(layer.bias, norm_sample)
    else:
        _create_or_extend_grad_sample(layer.bias, grad_sample, batch_dim)


def _compute_embedding_grad_sample(layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0) -> None:
    """Computes per sample gradients for `nn.Embedding` layer."""

    if autograd_grad_sample.fp16():
        if B.dtype != torch.half:
            B = B.half()

    if autograd_grad_sample.get_hooks_mode() == "ghost_norm":
        not_AAt: torch.Tensor = ~A[:, :, None].eq(A[:, None, :])
        # Clear the contribution to the norm of the gradient for the padding token.
        #   In vanilla backpropagation, this particular embedding doesn't contribute to the gradient anyway.
        #   For more see 1.10.0 doc: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        #       'the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”.'
        padding_idx = layer.padding_idx
        if padding_idx is not None:
            # The right way to think about the next line of code is that A_i[t, padding_idx] = 0 for all t in [T].
            #   So the entry gets cleared whenever one of A, A^t takes the padding idx.
            not_AAt.bitwise_or_((A[:, :, None] == padding_idx) | (A[:, None, :] == padding_idx))
        norm_sample = torch.sqrt((torch.bmm(B, B.transpose(-1, -2)).masked_fill(not_AAt, 0)).sum(dim=(1, 2)))
        _create_or_extend_norm_sample(layer.weight, norm_sample)
    else:
        A_dense = F.one_hot(A, num_classes=layer.weight.shape[0]).to(B)  # (batch_size, seq_len, vocab_dim,)
        grad_sample = torch.bmm(A_dense.permute(0, 2, 1), B)
        # `torch.nn.Embedding` layers don't accumulate gradient on the padding_idx position.
        #   We do the same for `grad_sample`.
        if layer.padding_idx is not None:
            # `grad_sample` has size (batch_size, num_vocab, embedding_dim).
            grad_sample[:, layer.padding_idx, :] = 0.
        _create_or_extend_grad_sample(layer.weight, grad_sample, batch_dim)


def _custom_compute_conv1d_grad_sample(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0):
    """Computes per sample gradients for `transformers.modeling_utils.Conv1D` layer."""

    if autograd_grad_sample.fp16():
        if B.dtype != torch.half:
            B = B.half()
        if A.dtype != torch.half:
            A = A.half()

    if autograd_grad_sample.get_hooks_mode() == "ghost_norm":
        _create_or_extend_norm_sample(layer.weight, _light_linear_weight_norm_sample(A, B))

        if layer.bias is not None:
            _create_or_extend_norm_sample(layer.bias, B.sum(dim=1).norm(2, dim=1))
    else:
        _create_or_extend_grad_sample(layer.weight, torch.bmm(A.permute(0, 2, 1), B), batch_dim)

        if layer.bias is not None:
            _create_or_extend_grad_sample(layer.bias, B.sum(dim=1), batch_dim)


_supported_layers_grad_samplers = {
    "Embedding": _compute_embedding_grad_sample,
    "Linear": _compute_linear_grad_sample,
    "LayerNorm": _compute_norm_grad_sample,
    "Conv1D": _custom_compute_conv1d_grad_sample,  # HuggingFace Open-AI GPT-2.
}
