"""RTL-friendly Linear (affine) layer.

Target API:
- torch.nn.Linear

Forward: y = x @ W^T + b, expressed as explicit multiply-accumulate.

Changes from original:
- torch.cat replaced with pre-allocated output tensor + slice assignment
  (tensor indexing, no torch function call).
- All forward computation uses only Python operators and tensor methods.

Keeps nn.Module / nn.Parameter / nn.init for checkpoint compatibility.
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
        # Cap broadcast tensor size to avoid OOM; process output channels in chunks.
        self.max_product_elems = 32 * 1024 * 1024
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

    def _forward_chunk(self, input, weight, bias):
        """Explicit multiply then reduce-sum for a slice of output channels."""
        input_expanded = input.unsqueeze(-2)
        weight_expanded = weight.view(*([1] * (input.ndim - 1)), weight.shape[0], self.in_features)
        out = (input_expanded * weight_expanded).sum(dim=-1)
        if bias is not None:
            out = out + bias
        return out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.numel() == 0:
            return self._forward_chunk(input, self.weight, self.bias)

        chunk_size = max(1, self.max_product_elems // input.numel())
        chunk_size = min(self.out_features, chunk_size)

        if chunk_size >= self.out_features:
            return self._forward_chunk(input, self.weight, self.bias)

        # Pre-allocate output and fill in chunks — avoids torch.cat
        out_shape = input.shape[:-1] + (self.out_features,)
        result = input.new_empty(out_shape)
        for start in range(0, self.out_features, chunk_size):
            end = min(start + chunk_size, self.out_features)
            bias_chunk = None if self.bias is None else self.bias[start:end]
            result[..., start:end] = self._forward_chunk(input, self.weight[start:end], bias_chunk)
        return result

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, max_product_elems={}".format(
            self.in_features, self.out_features, self.bias is not None, self.max_product_elems
        )


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    for shape in [(4, 8), (2, 5, 16), (1, 3, 7, 32)]:
        in_f, out_f = shape[-1], 11
        ref = nn.Linear(in_f, out_f, bias=True)
        impl = Linear(in_f, out_f, bias=True)
        impl.weight.copy_(ref.weight)
        if impl.bias is not None and ref.bias is not None:
            impl.bias.copy_(ref.bias)
        x = torch.randn(shape)
        err = (ref(x) - impl(x)).abs().max().item()
        print(f"shape={shape} max_abs_err = {err}")
        assert err < 1e-6, f"mismatch: {err}"
    print("Linear OK")


if __name__ == "__main__":
    _quick_parity_check()
