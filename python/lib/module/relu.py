"""RTL-friendly ReLU activation.

Target API:
- torch.nn.functional.relu
- torch.nn.ReLU

RTL mapping:
    out = x * (x > 0)
    Step 1 — compare: (x > 0)  → 1-bit comparator
    Step 2 — mux:     x * mask  → pass-through or zero

No torch / torch.nn imports — only Python operators and module_base.
"""

from __future__ import annotations

from lib.module.module_base import Module


def relu(input, inplace: bool = False):
    """逐元素 ReLU：負值歸零，以比較器 mask 實作。"""
    mask = input > 0      # BoolTensor: 1 where x > 0
    out = input * mask    # type promotion: float * bool → float
    if inplace:
        input.copy_(out)
        return input
    return out


class ReLU(Module):
    """``nn.ReLU`` 相容模組，呼叫手刻 ``relu`` 函式。"""

    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace

    def forward(self, input):
        return relu(input, inplace=self.inplace)

    def __repr__(self) -> str:
        return f"ReLU(inplace={self.inplace})"


def _quick_parity_check() -> None:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    y_ref = F.relu(x)
    y_impl = relu(x)
    assert torch.equal(y_ref, y_impl), f"mismatch: max_err={(y_ref-y_impl).abs().max().item()}"
    print("relu parity: exact match")

    a, b = x.clone(), x.clone()
    F.relu(a, inplace=True)
    relu(b, inplace=True)
    assert torch.equal(a, b)
    print("relu inplace parity: exact match")


if __name__ == "__main__":
    _quick_parity_check()
