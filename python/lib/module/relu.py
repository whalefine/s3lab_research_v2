"""Reference-compatible ReLU activation.

Target API:
- torch.nn.functional.relu

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py
  (``relu`` delegates to ATen ``relu`` / ``relu_`` kernels)

Notes:
- Math: ``relu(x) = max(x, 0)``.
- This implementation uses an explicit element-wise maximum so the dataflow maps
  more directly to RTL than a generic clamp kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """逐元素 ReLU：``max(x, 0)``，與 ``F.relu`` 語意對齊。"""
    zero = torch.zeros_like(input)
    out = torch.maximum(input, zero)
    if inplace:
        input.copy_(out)
        return input
    return out


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    y_ref = F.relu(x)
    y_impl = relu(x)
    assert torch.equal(y_ref, y_impl)
    a, b = x.clone(), x.clone()
    y_a = F.relu(a, inplace=True)
    y_b = relu(b, inplace=True)
    assert torch.equal(y_a, y_b) and torch.equal(a, b)
    print("relu parity: max_abs_err = 0.0 (exact match on random tensor)")


if __name__ == "__main__":
    _quick_parity_check()
