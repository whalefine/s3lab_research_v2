"""RTL-friendly ReLU6 activation.

Target API:
- torch.nn.functional.relu6
- torch.nn.ReLU6

RTL mapping:
    relu6(x) = clamp(x, 0, 6)   → saturation register with bounds [0, 6]

Delegates to lib.module.clamp for the explicit comparator-mux dataflow.
No torch / torch.nn imports.
"""

from __future__ import annotations

from lib.module.clamp import clamp
from lib.module.module_base import Module


def relu6(input, inplace: bool = False):
    """逐元素 ReLU6：以手刻 clamp 限制輸出在 [0, 6]。"""
    out = clamp(input, 0.0, 6.0)
    if inplace:
        input.copy_(out)
        return input
    return out


class ReLU6(Module):
    """``nn.ReLU6`` 相容模組，呼叫手刻 ``relu6`` 函式。"""

    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace

    def forward(self, input):
        return relu6(input, inplace=self.inplace)

    def __repr__(self) -> str:
        return f"ReLU6(inplace={self.inplace})"


def _quick_parity_check() -> None:
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    x = torch.randn(3, 4) * 3.0
    y_ref = F.relu6(x)
    y_impl = relu6(x)
    assert torch.equal(y_ref, y_impl), f"mismatch: max_err={(y_ref-y_impl).abs().max().item()}"
    print("relu6 parity: exact match")

    a, b = x.clone(), x.clone()
    F.relu6(a, inplace=True)
    relu6(b, inplace=True)
    assert torch.equal(a, b)
    print("relu6 inplace parity: exact match")


if __name__ == "__main__":
    _quick_parity_check()
