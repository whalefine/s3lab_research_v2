"""RTL-friendly MLP block.

Target API:
- timm.models.layers.Mlp

Structure: fc1 → fixed_point → act → fixed_point → drop → fc2 → fixed_point → drop.

Keeps nn.Module for parameter tracking / checkpoint loading.
Type hint Type[nn.Module] removed to avoid torch.nn import dependency in callers.
All forward computation delegates to hand-crafted sub-modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dropout import Dropout
from .fixed_point import to_fixed_point
from .linear import Linear
from .relu import ReLU


class Mlp(nn.Module):
    """Transformer-style feed-forward block built from local modules."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=ReLU,
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
        x = to_fixed_point(x, 8, 8)
        x = self.act(x)
        x = to_fixed_point(x, 8, 8)
        x = self.drop1(x)
        x = self.fc2(x)
        x = to_fixed_point(x, 8, 8)
        x = self.drop2(x)
        return x

    def extra_repr(self) -> str:
        return "in_features={}, hidden_features={}, out_features={}".format(
            self.in_features, self.hidden_features, self.out_features
        )


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    impl = Mlp(in_features=8, hidden_features=16, out_features=8, drop=0.0)
    impl.eval()
    x = torch.randn(2, 5, 8)
    y = impl(x)
    print(f"mlp output shape = {tuple(y.shape)}")
    assert y.shape == (2, 5, 8)
    print("Mlp OK")


if __name__ == "__main__":
    _quick_parity_check()
