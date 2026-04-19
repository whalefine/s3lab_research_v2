"""Reference-compatible ReLU6 activation.

Target API:
- torch.nn.functional.relu6

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py
  (``relu6`` delegates to ATen kernels)

Notes:
- Math: ``relu6(x) = min(max(x, 0), 6)``.
- This implementation uses explicit element-wise max/min steps so the dataflow
  maps more directly to RTL than a generic clamp kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def relu6(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """逐元素 ReLU6：將輸出限制在 ``[0, 6]``，與 ``F.relu6`` 語意對齊。"""
    zero = torch.zeros_like(input)
    six = torch.full_like(input, 6)
    out = torch.minimum(torch.maximum(input, zero), six)
    if inplace:
        input.copy_(out)
        return input
    return out


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4) * 3.0
    y_ref = F.relu6(x)
    y_impl = relu6(x)
    assert torch.equal(y_ref, y_impl)
    a, b = x.clone(), x.clone()
    y_a = F.relu6(a, inplace=True)
    y_b = relu6(b, inplace=True)
    assert torch.equal(y_a, y_b) and torch.equal(a, b)
    print("relu6 parity: max_abs_err = 0.0 (exact match on random tensor)")


if __name__ == "__main__":
    _quick_parity_check()
