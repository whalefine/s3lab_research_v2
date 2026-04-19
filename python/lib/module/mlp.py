"""Reference-compatible MLP block.

Target API:
- timm.models.layers.Mlp

Upstream references:
- https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py

Notes:
- This module mirrors the common transformer MLP structure:
  fc1 -> activation -> dropout -> fc2 -> dropout.
- It is intended to replace higher-level library MLP blocks with a local module
  composed from simpler building blocks in ``python/lib/module``.
"""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from .dropout import Dropout
from .linear import Linear


class Mlp(nn.Module):
    """Transformer-style feed-forward block built from local modules."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.act = act_layer()
        self.drop1 = Dropout(drop)
        self.fc2 = Linear(hidden_features, out_features, bias=bias, **factory_kwargs)
        self.drop2 = Dropout(drop)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.fc1(input)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def extra_repr(self) -> str:
        return (
            "in_features={}, hidden_features={}, out_features={}".format(
                self.in_features,
                self.hidden_features,
                self.out_features,
            )
        )


@torch.no_grad()
def _quick_parity_check() -> None:
    try:
        from timm.models.layers import Mlp as TimmMlp
    except ImportError as exc:
        raise RuntimeError("timm is required to run the MLP parity check") from exc

    torch.manual_seed(0)
    ref = TimmMlp(in_features=8, hidden_features=16, out_features=8, act_layer=nn.ReLU, drop=0.0)
    impl = Mlp(in_features=8, hidden_features=16, out_features=8, act_layer=nn.ReLU, drop=0.0)

    impl.fc1.weight.copy_(ref.fc1.weight)
    impl.fc2.weight.copy_(ref.fc2.weight)
    if impl.fc1.bias is not None and ref.fc1.bias is not None:
        impl.fc1.bias.copy_(ref.fc1.bias)
    if impl.fc2.bias is not None and ref.fc2.bias is not None:
        impl.fc2.bias.copy_(ref.fc2.bias)

    x = torch.randn(2, 5, 8)
    y_ref = ref(x)
    y_impl = impl(x)
    err = (y_ref - y_impl).abs().max().item()
    print("mlp parity max_abs_err =", err)
    assert err < 1e-6


if __name__ == "__main__":
    _quick_parity_check()
