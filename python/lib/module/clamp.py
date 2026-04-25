"""Hardware-friendly clamp (saturation).

RTL mapping:
    Step 1 — compare:   above = (x > max_val)   → 1-bit comparator output
                        below = (x < min_val)   → 1-bit comparator output
    Step 2 — mux:       if above  → max_val
                        elif below → min_val
                        else       → x

No torch.clamp / torch.where — only Python comparison and arithmetic operators.
"""

from __future__ import annotations

from lib.module.module_base import Module


def clamp(x, min_val: float, max_val: float):
    """逐元素飽和截斷：x < min → min，x > max → max，中間保持 x。"""
    above = x > max_val          # BoolTensor: 1 where x > max_val
    below = x < min_val          # BoolTensor: 1 where x < min_val
    in_range = ~above & ~below   # BoolTensor: 1 where min_val <= x <= max_val
    return x * in_range + max_val * above + min_val * below


class Clamp(Module):
    """``torch.clamp`` 相容模組，無可學習參數。"""

    def __init__(self, min_val: float, max_val: float) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return clamp(x, self.min_val, self.max_val)

    def __repr__(self) -> str:
        return f"Clamp(min={self.min_val}, max={self.max_val})"


def _quick_parity_check() -> None:
    import torch
    with torch.no_grad():
        x = torch.tensor([-10.0, -4.0, -2.0, 0.0, 2.0, 4.0, 10.0])
        y_impl = clamp(x, -4.0, 4.0)
        y_ref = torch.clamp(x, -4.0, 4.0)
        err = (y_ref - y_impl).abs().max().item()
        print(f"clamp max_abs_err = {err}")
        assert err < 1e-6, f"mismatch: {y_impl}"
        print("Clamp module OK")


if __name__ == "__main__":
    _quick_parity_check()
