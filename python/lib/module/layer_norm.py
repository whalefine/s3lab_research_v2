"""RTL-friendly LayerNorm implementation.

Target API:
- torch.nn.LayerNorm

Forward decomposition (explicit reduction steps):
    sum → mean → centered → squared → variance → inv_std → affine

Hardware note:
- inv_std = (var + eps) ** (-0.5)  requires reciprocal-sqrt.
  In hardware this needs CORDIC or Newton-Raphson LUT; kept here as float
  reference. Marked hardware-unfriendly in the module summary.

Keeps nn.Module / nn.Parameter for checkpoint compatibility.
torch.rsqrt removed; replaced with ** (-0.5) via Python power operator.
"""

from __future__ import annotations

import numbers
from typing import Union, Tuple, List

import torch
import torch.nn as nn


ShapeLike = Union[int, List[int], Tuple[int, ...], torch.Size]


class LayerNorm(nn.Module):
    """Pure-Python LayerNorm with explicit reduction steps; no torch.rsqrt."""

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

        # Step 1: mean
        sum_x = input.sum(dim=dims, keepdim=True)
        mean = sum_x / norm_elem_count

        # Step 2: variance (explicit squared residuals)
        centered = input - mean
        squared = centered * centered
        sum_sq = squared.sum(dim=dims, keepdim=True)
        var = sum_sq / norm_elem_count

        # Step 3: inv_std via ** (-0.5)  [hardware-unfriendly: needs reciprocal-sqrt]
        inv_std = (var + self.eps) ** (-0.5)

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


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    x = torch.randn(2, 7, 16)
    ref = nn.LayerNorm(16, eps=1e-6)
    impl = LayerNorm(16, eps=1e-6)
    impl.weight.copy_(ref.weight)
    if impl.bias is not None and ref.bias is not None:
        impl.bias.copy_(ref.bias)
    err = (ref(x) - impl(x)).abs().max().item()
    print(f"layer_norm parity max_abs_err = {err}")
    assert err < 1e-5, f"mismatch: {err}"
    print("LayerNorm OK")


if __name__ == "__main__":
    _quick_parity_check()
