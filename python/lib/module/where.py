"""RTL-friendly where (multiplexer) implementation.

Target API:
- torch.where(condition, input, other)

RTL mapping:
    c   = condition  (1-bit select line)
    out = input * c + other * (1 - c)   → 2-to-1 mux

This replaces the original mask-copy approach with an explicit linear
combination that maps directly to a hardware multiplexer.
No torch / torch.nn imports — only Python operators.
"""

from __future__ import annotations


def where(condition, input, other):
    """逐元素 2-to-1 多工器：condition=1 選 input，condition=0 選 other。"""
    c = condition * 1.0    # BoolTensor → FloatTensor (0.0 or 1.0)
    return input * c + other * (1.0 - c)


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    cond = torch.randn(3, 4) > 0
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    ref = torch.where(cond, a, b)
    impl = where(cond, a, b)
    err = (ref - impl).abs().max().item()
    print(f"where max_abs_err = {err}")
    assert err < 1e-6, f"mismatch: {err}"
    print("where OK")


if __name__ == "__main__":
    _quick_parity_check()
