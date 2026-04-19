"""RTL-friendly Linear (affine) layer.

Target API:
- torch.nn.Linear

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
- Forward is mathematically y = x @ W^T + b, implemented in ATen as ``F.linear``.

Notes:
- PyTorch ``nn.Linear.forward`` delegates to ``F.linear``, which calls the
  fused ``linear`` ATen op (CPU/CUDA kernels).
- This module uses explicit multiply + reduce-sum + bias add steps so the
  forward dataflow is closer to a typical RTL implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init


class Linear(nn.Module):
    """Fully connected layer with explicit multiply-accumulate style forward."""

    __constants__ = ["in_features", "out_features"]

    in_features: int
    out_features: int
    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Expand the input and weights so the core affine op is expressed as
        # element-wise products followed by a reduction across in_features.
        input_expanded = input.unsqueeze(-2)
        weight_expanded = self.weight.view(*([1] * (input.ndim - 1)), self.out_features, self.in_features)
        products = input_expanded * weight_expanded
        out = products.sum(dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
        )


@torch.no_grad()
def _quick_parity_check() -> None:
    """Quick smoke test against torch.nn.Linear."""
    torch.manual_seed(0)
    for shape in [(4, 8), (2, 5, 16), (1, 3, 7, 32)]:
        in_f, out_f = shape[-1], 11
        ref = nn.Linear(in_f, out_f, bias=True)
        impl = Linear(in_f, out_f, bias=True)
        impl.weight.copy_(ref.weight)
        if impl.bias is not None and ref.bias is not None:
            impl.bias.copy_(ref.bias)
        x = torch.randn(shape)
        y_ref = ref(x)
        y_impl = impl(x)
        err = (y_ref - y_impl).abs().max().item()
        print(f"shape={shape} max_abs_err = {err}")
        assert err < 1e-6

    ref_nb = nn.Linear(8, 5, bias=False)
    impl_nb = Linear(8, 5, bias=False)
    impl_nb.weight.copy_(ref_nb.weight)
    x = torch.randn(2, 8)
    err = (ref_nb(x) - impl_nb(x)).abs().max().item()
    print("no-bias max_abs_err =", err)
    assert err < 1e-6


if __name__ == "__main__":
    _quick_parity_check()
