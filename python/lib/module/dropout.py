"""Reference-compatible Dropout implementation.

Target API:
- torch.nn.Dropout
- torch.nn.functional.dropout

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/dropout.py
- https://github.com/pytorch/pytorch/blob/main/torch/nn/functional.py

Notes:
- PyTorch dropout dispatches to backend kernels for mask generation and scaling.
- This module keeps the dataflow explicit: sample mask -> zero dropped elements ->
  scale survivors by ``1 / (1 - p)`` during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> torch.Tensor:
    """Apply dropout with explicit mask and scaling."""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, got {p}")
    if not training or p == 0.0:
        return input if inplace else input.clone()
    if p == 1.0:
        out = torch.zeros_like(input)
        if inplace:
            input.copy_(out)
            return input
        return out

    keep_prob = 1.0 - p
    mask = (torch.rand_like(input) < keep_prob).to(input.dtype)
    out = input * mask
    out = out / keep_prob
    if inplace:
        input.copy_(out)
        return input
    return out


class Dropout(nn.Module):
    """``nn.Dropout`` compatible module."""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return dropout(input, p=self.p, training=self.training, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    ref = nn.Dropout(p=0.25)
    impl = Dropout(p=0.25)

    ref.eval()
    impl.eval()
    err_eval = (ref(x) - impl(x)).abs().max().item()
    print("dropout eval max_abs_err =", err_eval)
    assert err_eval == 0.0

    ref.train()
    impl.train()
    torch.manual_seed(123)
    y_ref = ref(x)
    torch.manual_seed(123)
    y_impl = impl(x)
    err_train = (y_ref - y_impl).abs().max().item()
    print("dropout train max_abs_err =", err_train)
    assert err_train < 1e-6


if __name__ == "__main__":
    _quick_parity_check()
