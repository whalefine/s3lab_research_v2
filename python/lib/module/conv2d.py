"""RTL-friendly Conv2d implementation.

Target API:
- torch.nn.Conv2d

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py
- PyTorch Conv2d forward ultimately dispatches to ATen/C++ kernels.

Notes:
- This module reimplements the forward path with explicit padding, window
  extraction, element-wise multiply, and reduction.
- The implementation is intended to keep the dataflow closer to a typical RTL
  convolution block than fused backend kernels.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


Int2 = Union[int, Tuple[int, int]]


def _to_2tuple(value: Int2) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class Conv2d(nn.Module):
    """Explicit NCHW Conv2d with PyTorch-compatible core arguments."""

    __constants__ = [
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Int2,
        stride: Int2 = 1,
        padding: Int2 = 0,
        dilation: Int2 = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_2tuple(kernel_size)
        self.stride = _to_2tuple(stride)
        self.padding = _to_2tuple(padding)
        self.dilation = _to_2tuple(dilation)
        self.groups = groups

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        weight_shape = (
            out_channels,
            in_channels // groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        self.weight = nn.Parameter(torch.empty(weight_shape, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
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
        if input.ndim != 4:
            raise ValueError(f"Conv2d expects 4D NCHW input, got shape {tuple(input.shape)}")

        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation
        ker_h, ker_w = self.kernel_size

        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h))
        batch, _, in_h, in_w = input_padded.shape

        eff_ker_h = dil_h * (ker_h - 1) + 1
        eff_ker_w = dil_w * (ker_w - 1) + 1
        out_h = (in_h - eff_ker_h) // stride_h + 1
        out_w = (in_w - eff_ker_w) // stride_w + 1

        channels_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        output = input.new_zeros((batch, self.out_channels, out_h, out_w))

        for oy in range(out_h):
            for ox in range(out_w):
                y0 = oy * stride_h
                x0 = ox * stride_w
                patch = input_padded[
                    :,
                    :,
                    y0:y0 + eff_ker_h:dil_h,
                    x0:x0 + eff_ker_w:dil_w,
                ]
                for group_idx in range(self.groups):
                    in_start = group_idx * channels_per_group
                    in_end = in_start + channels_per_group
                    out_start = group_idx * out_per_group
                    out_end = out_start + out_per_group

                    patch_group = patch[:, in_start:in_end].unsqueeze(1)
                    weight_group = self.weight[out_start:out_end].unsqueeze(0)
                    products = patch_group * weight_group
                    output[:, out_start:out_end, oy, ox] = products.sum(dim=(2, 3, 4))

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

    def extra_repr(self) -> str:
        return (
            "in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, "
            "dilation={}, groups={}, bias={}".format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.bias is not None,
            )
        )


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    ref = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=True)
    impl = Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=True)
    impl.weight.copy_(ref.weight)
    if impl.bias is not None and ref.bias is not None:
        impl.bias.copy_(ref.bias)
    x = torch.randn(2, 3, 8, 8)
    err = (ref(x) - impl(x)).abs().max().item()
    print("conv2d parity max_abs_err =", err)
    assert err < 1e-6


if __name__ == "__main__":
    _quick_parity_check()
