"""RTL-friendly fixed-point helper functions.

Target APIs:
- torch.round
- torch.clamp

Notes:
- These helpers make fixed-point quantization dataflow explicit:
  scale -> round -> saturate -> descale.
- The functions are intended as reusable building blocks for modules that need
  hardware-oriented fixed-point simulation.
"""

from __future__ import annotations

import torch


def round_tensor(input: torch.Tensor) -> torch.Tensor:
    """Apply element-wise rounding with an explicit helper wrapper."""
    return torch.round(input)


def clamp_tensor(
    input: torch.Tensor,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> torch.Tensor:
    """Apply element-wise saturation bounds with an explicit helper wrapper."""
    return torch.clamp(input, min=min_value, max=max_value)


def to_fixed_point(input: torch.Tensor, int_bits: int = 8, frac_bits: int = 8) -> torch.Tensor:
    """Quantize a tensor to signed fixed-point with rounding and saturation."""
    scale = 2 ** frac_bits
    qmin = -(2 ** (int_bits + frac_bits - 1))
    qmax = (2 ** (int_bits + frac_bits - 1)) - 1
    scaled = input * scale
    rounded = round_tensor(scaled)
    saturated = clamp_tensor(rounded, qmin, qmax)
    return saturated / scale


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 5) * 3.0
    scale = 2 ** 8
    qmin = -(2 ** (8 + 8 - 1))
    qmax = (2 ** (8 + 8 - 1)) - 1

    ref = torch.clamp(torch.round(x * scale), qmin, qmax) / scale
    impl = to_fixed_point(x, 8, 8)
    err = (ref - impl).abs().max().item()
    print("fixed_point parity max_abs_err =", err)
    assert err < 1e-6


if __name__ == "__main__":
    _quick_parity_check()
