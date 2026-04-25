"""RTL-friendly Conv2d implementation.

Target API:
- torch.nn.Conv2d

Changes from original:
- F.pad removed; replaced with _pad2d_zero() using tensor indexing and
  .new_zeros() (tensor method, no F.pad call).
- All forward computation uses only Python operators and tensor methods.

Keeps nn.Module / nn.Parameter / nn.init for checkpoint compatibility.

Hardware alignment (FPGA / fixed-point flow):
- Core cost is MAC: multiply input taps by weights and accumulate.
- Set explicit_mac_loops=True for scalar nested-loop accumulate that
  matches line-buffer + MAC tree scheduling.
- Quantization belongs at model boundary (to_fixed_point), not here.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.nn import init


Int2 = Union[int, Tuple[int, int]]


def _to_2tuple(value: Int2) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _pad2d_zero(x, pad_h: int, pad_w: int):
    """Zero-pad a 4-D NCHW tensor by pad_h rows and pad_w cols on each side.

    Uses .new_zeros() + slice assignment — no F.pad / torch function call.
    RTL mapping: zero-padding registers at boundary of line buffer.
    """
    if pad_h == 0 and pad_w == 0:
        return x
    B, C, H, W = x.shape
    out = x.new_zeros(B, C, H + 2 * pad_h, W + 2 * pad_w)
    out[:, :, pad_h: pad_h + H, pad_w: pad_w + W] = x
    return out


class Conv2d(nn.Module):
    """Explicit NCHW Conv2d with hand-crafted forward (no F.pad, no F.conv2d)."""

    __constants__ = [
        "in_channels", "out_channels", "kernel_size", "stride",
        "padding", "dilation", "groups", "explicit_mac_loops",
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
        *,
        explicit_mac_loops: bool = False,
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
        self.explicit_mac_loops = explicit_mac_loops

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        weight_shape = (out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
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

    def _forward_vectorized(self, input: torch.Tensor) -> torch.Tensor:
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation
        ker_h, ker_w = self.kernel_size

        padded = _pad2d_zero(input, pad_h, pad_w)
        batch, _, in_h, in_w = padded.shape

        eff_ker_h = dil_h * (ker_h - 1) + 1
        eff_ker_w = dil_w * (ker_w - 1) + 1
        out_h = (in_h - eff_ker_h) // stride_h + 1
        out_w = (in_w - eff_ker_w) // stride_w + 1

        channels_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        output = input.new_zeros(batch, self.out_channels, out_h, out_w)

        for oy in range(out_h):
            for ox in range(out_w):
                y0 = oy * stride_h
                x0 = ox * stride_w
                patch = padded[:, :, y0: y0 + eff_ker_h: dil_h, x0: x0 + eff_ker_w: dil_w]
                for g in range(self.groups):
                    in_s = g * channels_per_group
                    out_s = g * out_per_group
                    out_e = out_s + out_per_group
                    patch_g = patch[:, in_s: in_s + channels_per_group].unsqueeze(1)
                    weight_g = self.weight[out_s:out_e].unsqueeze(0)
                    output[:, out_s:out_e, oy, ox] = (patch_g * weight_g).sum(dim=(2, 3, 4))

        return output

    def _forward_explicit_mac(self, input: torch.Tensor) -> torch.Tensor:
        """Scalar MAC nest — one mul-add per inner step (RTL / golden-model style)."""
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation
        ker_h, ker_w = self.kernel_size

        padded = _pad2d_zero(input, pad_h, pad_w)
        batch, _, in_h, in_w = padded.shape

        eff_ker_h = dil_h * (ker_h - 1) + 1
        eff_ker_w = dil_w * (ker_w - 1) + 1
        out_h = (in_h - eff_ker_h) // stride_h + 1
        out_w = (in_w - eff_ker_w) // stride_w + 1

        channels_per_group = self.in_channels // self.groups
        out_per_group = self.out_channels // self.groups
        output = input.new_zeros(batch, self.out_channels, out_h, out_w)

        for b in range(batch):
            for oy in range(out_h):
                for ox in range(out_w):
                    y0 = oy * stride_h
                    x0 = ox * stride_w
                    for g in range(self.groups):
                        in_base = g * channels_per_group
                        out_base = g * out_per_group
                        for o_rel in range(out_per_group):
                            oc = out_base + o_rel
                            acc = padded.new_zeros(())
                            for ic_rel in range(channels_per_group):
                                ic = in_base + ic_rel
                                for kh in range(ker_h):
                                    for kw in range(ker_w):
                                        y_in = y0 + kh * dil_h
                                        x_in = x0 + kw * dil_w
                                        acc = acc + padded[b, ic, y_in, x_in] * self.weight[oc, ic_rel, kh, kw]
                            output[b, oc, oy, ox] = acc
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 4:
            raise ValueError(f"Conv2d expects 4D NCHW input, got shape {tuple(input.shape)}")
        if self.explicit_mac_loops:
            output = self._forward_explicit_mac(input)
        else:
            output = self._forward_vectorized(input)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output

    def extra_repr(self) -> str:
        return (
            "in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, "
            "dilation={}, groups={}, bias={}, explicit_mac_loops={}".format(
                self.in_channels, self.out_channels, self.kernel_size, self.stride,
                self.padding, self.dilation, self.groups,
                self.bias is not None, self.explicit_mac_loops,
            )
        )


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    ref = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=True)
    impl = Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=True)
    impl.weight.copy_(ref.weight)
    if impl.bias is not None and ref.bias is not None:
        impl.bias.copy_(ref.bias)
    x = torch.randn(2, 3, 8, 8)
    err = (ref(x) - impl(x)).abs().max().item()
    print(f"conv2d parity max_abs_err = {err}")
    assert err < 1e-6, f"mismatch: {err}"
    print("Conv2d OK")


if __name__ == "__main__":
    _quick_parity_check()
