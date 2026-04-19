"""RTL-friendly sort implementation.

Target API:
- torch.sort

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/_refs/__init__.py
- Runtime dispatch is handled by ATen/C++ kernels.

Notes:
- This module uses an explicit selection-sort style loop on the target dimension.
- It is intended for small tensor sizes where hardware-readable dataflow matters
  more than software performance.
"""

from __future__ import annotations

from collections import namedtuple

import torch


SortResult = namedtuple("SortResult", ["values", "indices"])


def sort(input: torch.Tensor, dim: int = -1, descending: bool = False, stable: bool = False) -> SortResult:
    """Sort elements along a dimension with explicit iterative selection."""
    if stable:
        raise NotImplementedError("stable=True is not implemented in this reference module")

    dim = dim if dim >= 0 else input.ndim + dim
    if dim < 0 or dim >= input.ndim:
        raise IndexError("dimension out of range")

    moved = input.movedim(dim, -1)
    last_dim = moved.shape[-1]
    flat = moved.reshape(-1, last_dim)
    indices = torch.arange(last_dim, device=input.device, dtype=torch.long).expand(flat.shape[0], -1).clone()
    values = flat.clone()

    for pos in range(last_dim):
        tail = values[:, pos:]
        if descending:
            best_rel = tail.argmax(dim=-1)
        else:
            best_rel = tail.argmin(dim=-1)
        best = best_rel + pos
        row_ids = torch.arange(values.shape[0], device=input.device)

        cur_values = values[row_ids, pos].clone()
        cur_indices = indices[row_ids, pos].clone()

        values[row_ids, pos] = values[row_ids, best]
        indices[row_ids, pos] = indices[row_ids, best]
        values[row_ids, best] = cur_values
        indices[row_ids, best] = cur_indices

    out_shape = moved.shape
    values = values.reshape(out_shape).movedim(-1, dim)
    indices = indices.reshape(out_shape).movedim(-1, dim)
    return SortResult(values=values, indices=indices)


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    ref = torch.sort(x, dim=1)
    impl = sort(x, dim=1)
    err_v = (ref.values - impl.values).abs().max().item()
    err_i = (ref.indices - impl.indices).abs().max().item()
    print("sort values max_abs_err =", err_v)
    print("sort indices max_abs_err =", err_i)
    assert err_v < 1e-6
    assert err_i == 0


if __name__ == "__main__":
    _quick_parity_check()
