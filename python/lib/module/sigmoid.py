"""Hardware-friendly piecewise-linear sigmoid approximation.

Target API (kept):
- torch.nn.Sigmoid
- torch.nn.functional.sigmoid / torch.sigmoid

Approximation:
    σ(x) = clamp(x, -4, 4) * (1/8) + 0.5

RTL mapping:
    Step 1 — saturate:  clamp(x, -4, 4)  → saturation register
    Step 2 — scale:     >> 3              → arithmetic right shift by 3
    Step 3 — offset:    + 0.5             → adder with constant

Boundary values:
    x <= -4  →  -4 * 0.125 + 0.5 = 0
    x  =  0  →   0 * 0.125 + 0.5 = 0.5
    x >= +4  →  +4 * 0.125 + 0.5 = 1

No torch / torch.nn imports — only Python operators and lib.module primitives.
"""

from __future__ import annotations

from lib.module.clamp import clamp
from lib.module.module_base import Module


def sigmoid(x):
    """逐元素分段線性 sigmoid：飽和截斷 → shift-add。"""
    x_sat = clamp(x, -4.0, 4.0)
    return x_sat * 0.125 + 0.5


class Sigmoid(Module):
    """``nn.Sigmoid`` 相容模組，呼叫手刻 ``sigmoid`` 函式。"""

    def forward(self, x):
        return sigmoid(x)


def _quick_parity_check() -> None:
    import torch
    with torch.no_grad():
        x = torch.tensor([-10.0, -4.0, -2.0, 0.0, 2.0, 4.0, 10.0], dtype=torch.float32)
        y_impl = sigmoid(x)
        y_expected = torch.tensor([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0], dtype=torch.float32)
        print("sigmoid piecewise output =", y_impl.tolist())
        assert torch.allclose(y_impl, y_expected, atol=1e-6, rtol=0), f"mismatch: {y_impl}"

        m_impl = Sigmoid()
        assert torch.allclose(m_impl(x), y_expected, atol=1e-6, rtol=0)
        print("Sigmoid module OK")


if __name__ == "__main__":
    _quick_parity_check()
