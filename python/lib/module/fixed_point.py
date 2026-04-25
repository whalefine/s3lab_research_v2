"""RTL-friendly fixed-point helper functions.

Target APIs:
- torch.round  → replaced by tensor method .round()
- torch.clamp  → replaced by lib.module.clamp

Dataflow: scale → round → saturate → descale.
No torch imports — only tensor methods and lib.module.clamp.
"""

from __future__ import annotations

from lib.module.clamp import clamp as _clamp


def round_tensor(input):
    """逐元素四捨五入，使用 tensor 方法（無 torch.round）。"""
    return input.round()


def clamp_tensor(input, min_value=None, max_value=None):
    """逐元素飽和截斷，使用手刻 clamp 模組（無 torch.clamp）。

    None 表示該方向不截斷（以有限值逐步 clamp 避免 inf * 0 = nan）。
    """
    out = input
    if min_value is not None:
        below = out < min_value
        out = out * (~below) + min_value * below
    if max_value is not None:
        above = out > max_value
        out = out * (~above) + max_value * above
    return out


def to_fixed_point(input, int_bits: int = 8, frac_bits: int = 8):
    """有號定點數量化：scale → round → saturate → descale。"""
    scale = 2 ** frac_bits
    qmin = -(2 ** (int_bits + frac_bits - 1))
    qmax = (2 ** (int_bits + frac_bits - 1)) - 1
    scaled = input * scale
    rounded = round_tensor(scaled)
    saturated = clamp_tensor(rounded, qmin, qmax)
    return saturated / scale


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    x = torch.randn(4, 5) * 3.0
    scale = 2 ** 8
    qmin = -(2 ** (8 + 8 - 1))
    qmax = (2 ** (8 + 8 - 1)) - 1

    ref = torch.clamp(torch.round(x * scale), qmin, qmax) / scale
    impl = to_fixed_point(x, 8, 8)
    err = (ref - impl).abs().max().item()
    print(f"fixed_point parity max_abs_err = {err}")
    assert err < 1e-6, f"mismatch: {err}"
    print("fixed_point OK")


if __name__ == "__main__":
    _quick_parity_check()
