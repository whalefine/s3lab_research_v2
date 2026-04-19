"""Reference-compatible where implementation.

Target API:
- torch.where

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/_refs/__init__.py
- Runtime dispatch is handled by ATen/C++ kernels.

Notes:
- This module keeps the selection dataflow explicit by broadcasting inputs and
  then copying values according to the boolean condition mask.
"""

from __future__ import annotations

from typing import Union

import torch


ScalarLike = Union[int, float, bool]


def where(condition: torch.Tensor, input: Union[torch.Tensor, ScalarLike], other: Union[torch.Tensor, ScalarLike]) -> torch.Tensor:
    """Broadcast inputs and select element values using an explicit mask copy."""
    input_tensor = input if isinstance(input, torch.Tensor) else torch.as_tensor(input, device=condition.device)
    other_tensor = other if isinstance(other, torch.Tensor) else torch.as_tensor(other, device=condition.device)

    cond_b, input_b, other_b = torch.broadcast_tensors(condition.bool(), input_tensor, other_tensor)
    output = other_b.clone()
    output[cond_b] = input_b[cond_b]
    return output


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    cond = torch.randn(3, 4) > 0
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    ref = torch.where(cond, a, b)
    impl = where(cond, a, b)
    err = (ref - impl).abs().max().item()
    print("where max_abs_err =", err)
    assert err < 1e-6


if __name__ == "__main__":
    _quick_parity_check()
