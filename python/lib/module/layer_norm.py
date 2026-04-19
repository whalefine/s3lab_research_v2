"""RTL-friendly LayerNorm implementation.

Target API:
- torch.nn.LayerNorm

Upstream references:
- https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/modules/normalization.py
- https://raw.githubusercontent.com/pytorch/pytorch/main/torch/nn/functional.py

Notes:
- PyTorch's nn.LayerNorm.forward delegates to F.layer_norm, which dispatches to
  torch.layer_norm (ATen/C++ backend kernel).
- This module reimplements the forward path as explicit reduction steps:
  sum -> mean -> centered square -> variance -> rsqrt path -> affine.
- The goal is to keep the dataflow closer to a typical RTL decomposition.
"""

import numbers
from typing import Union, Tuple, List

import torch
import torch.nn as nn


ShapeLike = Union[int, List[int], Tuple[int, ...], torch.Size]


class LayerNorm(nn.Module):
    """Pure-Python LayerNorm with explicit reduction steps."""

    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]

    def __init__(
        self,
        normalized_shape: ShapeLike,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(input.ndim - len(self.normalized_shape), input.ndim))
        norm_elem_count = 1
        for size in self.normalized_shape:
            norm_elem_count *= size

        sum_x = input.sum(dim=dims, keepdim=True)
        mean = sum_x / norm_elem_count

        centered = input - mean
        squared = centered * centered
        sum_sq = squared.sum(dim=dims, keepdim=True)
        var = sum_sq / norm_elem_count

        inv_std = torch.rsqrt(var + self.eps)
        out = centered * inv_std
        if self.weight is not None:
            out = out * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, "
            "bias={use_bias}".format(**self.__dict__, use_bias=self.bias is not None)
        )


@torch.no_grad()
def _quick_parity_check() -> None:
    """Quick smoke test against torch.nn.LayerNorm."""
    torch.manual_seed(0)
    x = torch.randn(2, 7, 16)
    ref = nn.LayerNorm(16, eps=1e-6)
    impl = LayerNorm(16, eps=1e-6)
    impl.weight.copy_(ref.weight)
    if impl.bias is not None and ref.bias is not None:
        impl.bias.copy_(ref.bias)
    y_ref = ref(x)
    y_impl = impl(x)
    print("layer_norm parity max_abs_err =", (y_ref - y_impl).abs().max().item())


if __name__ == "__main__":
    _quick_parity_check()
