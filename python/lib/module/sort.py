"""RTL-friendly sort implementation.

Target API:
- torch.sort

Changes from original:
- torch.arange replaced with cumsum trick on .new_ones() (tensor method).
  RTL mapping: index counter register incremented each step.
- All other operations use tensor methods (.movedim, .reshape, .clone,
  .argmax, .argmin) — no torch.xxx function calls.

Hardware note:
- Sorting is implemented as iterative selection sort; maps to
  a comparator network in RTL. Manageable but non-trivial.
"""

from __future__ import annotations

from collections import namedtuple


SortResult = namedtuple("SortResult", ["values", "indices"])


def sort(input, dim: int = -1, descending: bool = False, stable: bool = False) -> SortResult:
    """Sort elements along a dimension with explicit iterative selection."""
    if stable:
        raise NotImplementedError("stable=True is not implemented in this reference module")

    dim = dim if dim >= 0 else input.ndim + dim
    if dim < 0 or dim >= input.ndim:
        raise IndexError("dimension out of range")

    moved = input.movedim(dim, -1)
    last_dim = moved.shape[-1]
    flat = moved.reshape(-1, last_dim)
    N = flat.shape[0]

    # Index range [0, 1, ..., last_dim-1] via cumsum — no torch.arange
    idx_base = flat.new_ones(1, last_dim).long().cumsum(dim=-1) - 1
    indices = idx_base.expand(N, -1).clone()
    values = flat.clone()

    # Row ids [0, 1, ..., N-1] via cumsum — no torch.arange
    row_ids = values.new_ones(N).long().cumsum(dim=0) - 1

    for pos in range(last_dim):
        tail = values[:, pos:]
        best_rel = tail.argmax(dim=-1) if descending else tail.argmin(dim=-1)
        best = best_rel + pos

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


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    x = torch.randn(4, 6)
    ref = torch.sort(x, dim=1)
    impl = sort(x, dim=1)
    err_v = (ref.values - impl.values).abs().max().item()
    err_i = (ref.indices - impl.indices).abs().max().item()
    print(f"sort values max_abs_err = {err_v}")
    print(f"sort indices max_abs_err = {err_i}")
    assert err_v < 1e-6 and err_i == 0
    print("sort OK")


if __name__ == "__main__":
    _quick_parity_check()
