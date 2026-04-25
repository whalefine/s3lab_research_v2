"""RTL-friendly Dropout implementation.

Target API:
- torch.nn.Dropout
- torch.nn.functional.dropout

Hardware note:
- Dropout requires a PRNG (hardware-unfriendly).
- At inference (training=False) or p=0, it is identity — the PRNG path is
  never reached in RTL. Only the training path needs stochastic hardware.
- Mask generation uses tensor.new_empty().uniform_() (tensor method, no torch import).

No torch / torch.nn imports — only Python operators and tensor methods.
"""

from __future__ import annotations

from lib.module.module_base import Module


def dropout(input, p: float = 0.5, training: bool = True, inplace: bool = False):
    """顯式 mask 生成 + scale 的 dropout。"""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout probability must be between 0 and 1, got {p}")
    if not training or p == 0.0:
        return input if inplace else input.clone()
    if p == 1.0:
        out = input * 0
        if inplace:
            input.copy_(out)
            return input
        return out

    keep_prob = 1.0 - p
    # Generate uniform random mask without torch.rand_like
    rand = input.new_empty(input.shape).uniform_(0.0, 1.0)
    mask = rand < keep_prob   # BoolTensor
    out = input * mask / keep_prob
    if inplace:
        input.copy_(out)
        return input
    return out


class Dropout(Module):
    """``nn.Dropout`` 相容模組。"""

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return dropout(input, p=self.p, training=self.training, inplace=self.inplace)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p}, inplace={self.inplace})"


def _quick_parity_check() -> None:
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    x = torch.randn(4, 5)

    ref = nn.Dropout(p=0.25)
    impl = Dropout(p=0.25)
    ref.eval()
    impl.eval()
    err = (ref(x) - impl(x)).abs().max().item()
    print(f"dropout eval max_abs_err = {err}")
    assert err == 0.0
    print("Dropout eval OK")


if __name__ == "__main__":
    _quick_parity_check()
