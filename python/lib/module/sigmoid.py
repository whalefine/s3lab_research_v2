"""Reference-compatible Sigmoid activation.

Target API:
- torch.nn.Sigmoid
- torch.nn.functional.sigmoid / torch.sigmoid

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py
- Forward delegates to ATen ``sigmoid`` kernel.

Notes:
- Math: ``σ(x) = 1 / (1 + exp(-x))``.
- Naive ``1 / (1 + exp(-x))`` 在 |x| 很大時易溢位；此處用 ``z = exp(-|x|)`` 的分支式
  寫法（與常見數值庫相同），對齊 ``torch.sigmoid`` 的穩定性與前向結果。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sigmoid(input: torch.Tensor) -> torch.Tensor:
    """逐元素 sigmoid，與 ``torch.sigmoid`` / ``F.sigmoid`` 語意對齊。"""
    z = torch.exp(-torch.abs(input))
    return torch.where(input >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


class Sigmoid(nn.Module):
    """``nn.Sigmoid`` 相容模組（無參數）。"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return sigmoid(input)


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 5) * 5.0
    y_ref = torch.sigmoid(x)
    y_impl = sigmoid(x)
    err = (y_ref - y_impl).abs().max().item()
    print("sigmoid functional max_abs_err =", err)
    assert err < 1e-6
    m_ref = nn.Sigmoid()
    m_impl = Sigmoid()
    err_m = (m_ref(x) - m_impl(x)).abs().max().item()
    print("Sigmoid module max_abs_err =", err_m)
    assert err_m < 1e-6
    # 極端值
    t = torch.tensor([-80.0, 80.0, 0.0], dtype=torch.float32)
    assert torch.allclose(torch.sigmoid(t), sigmoid(t), atol=1e-6, rtol=0)


if __name__ == "__main__":
    _quick_parity_check()
