"""Minimal base class for parameter-free RTL-friendly modules.

Replaces torch.nn.Module for modules that carry no learnable parameters.
All computation uses plain Python operators so the dataflow maps 1-to-1
to RTL primitives (comparators, multipliers, adders).
"""

from __future__ import annotations


class Module:
    """Callable base for stateless RTL modules — no torch.nn dependency."""

    training: bool = False  # class-level default; .train()/.eval() create instance attribute

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
