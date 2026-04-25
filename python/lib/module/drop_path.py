"""RTL-friendly DropPath (stochastic depth) implementation.

Target API:
- timm.models.layers.DropPath
- timm.models.layers.drop_path

Hardware note:
- DropPath requires per-sample PRNG (hardware-unfriendly, same as Dropout).
- At inference (training=False) or drop_prob=0, it is always identity —
  the PRNG path is never reached in the RTL inference flow.
- Mask generation uses tensor.new_empty().uniform_() (tensor method, no torch import).

No torch / torch.nn imports — only Python operators and tensor methods.
"""

from __future__ import annotations

from lib.module.module_base import Module


def drop_path(input, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """逐樣本 stochastic depth：推論時為 identity。"""
    if drop_prob <= 0.0 or not training:
        return input
    keep_prob = 1.0 - drop_prob
    # Per-sample mask shape: (N, 1, 1, ...) to broadcast over spatial dims
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    rand = input.new_empty(shape).uniform_(0.0, 1.0)
    mask = rand < keep_prob    # BoolTensor, shape (N, 1, ...)
    if scale_by_keep and keep_prob > 0.0:
        mask = mask * (1.0 / keep_prob)
    return input * mask


class DropPath(Module):
    """``timm`` DropPath 相容模組，推論時為 identity。"""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, input):
        return drop_path(input, drop_prob=self.drop_prob, training=self.training,
                         scale_by_keep=self.scale_by_keep)

    def __repr__(self) -> str:
        return f"DropPath(drop_prob={self.drop_prob}, scale_by_keep={self.scale_by_keep})"


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    x = torch.randn(4, 5, 7)

    impl_eval = DropPath(drop_prob=0.3)
    impl_eval.eval()
    y = impl_eval(x)
    err = (y - x).abs().max().item()
    print(f"drop_path eval max_abs_err = {err}")
    assert err == 0.0, "eval must be identity"

    impl_zero = DropPath(drop_prob=0.0)
    impl_zero.train()
    y = impl_zero(x)
    err = (y - x).abs().max().item()
    print(f"drop_path zero-prob train max_abs_err = {err}")
    assert err == 0.0
    print("DropPath OK")


if __name__ == "__main__":
    _quick_parity_check()
