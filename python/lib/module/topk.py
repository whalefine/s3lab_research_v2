"""RTL-friendly top-k implementation.

Target API:
- torch.topk

Changes from original:
- torch.iinfo  → element_size() * 8 to derive integer bit-width (tensor method).
- torch.full_like  → .new_full() (tensor method).
- torch.stack  → pre-allocated tensor + slice assignment (no torch function call).
- torch.empty_like → .new_empty().long() (tensor methods).
- torch.argsort    → lib.module.sort (hand-crafted sort module).
- All other operations use tensor methods (.movedim, .reshape, .max, .min,
  .scatter_, .clone, .gather) — no torch.xxx function calls.

Hardware note:
- Iterative compare-select maps to a comparator tree; manageable in RTL.
- scatter_ (fill-with-sentinel) maps to a conditional write with a mask.
"""

from __future__ import annotations

from collections import namedtuple

from lib.module.sort import sort as _sort


TopkResult = namedtuple("TopkResult", ["values", "indices"])


def topk(input, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> TopkResult:
    """Select the top-k elements with explicit iterative compare-select steps."""
    if input.ndim == 0:
        raise ValueError("topk expects at least 1D input")

    dim = dim if dim >= 0 else input.ndim + dim
    if dim < 0 or dim >= input.ndim:
        raise IndexError("dimension out of range")

    size_along_dim = input.shape[dim]
    if k < 0 or k > size_along_dim:
        raise ValueError(f"k must be in range [0, {size_along_dim}], got {k}")

    moved = input.movedim(dim, -1)
    flat = moved.reshape(-1, size_along_dim)
    N = flat.shape[0]
    work = flat.clone()

    # Sentinel fill value — avoids torch.iinfo
    if input.dtype.is_floating_point:
        fill_value = float("-inf") if largest else float("inf")
    else:
        bits = input.element_size() * 8          # tensor method: bytes → bits
        fill_value = -(2 ** (bits - 1)) if largest else (2 ** (bits - 1) - 1)

    selected_values: list = []
    selected_indices: list = []
    for _ in range(k):
        if largest:
            values, indices = work.max(dim=-1)
        else:
            values, indices = work.min(dim=-1)
        selected_values.append(values)
        selected_indices.append(indices)
        # Mark selected position as sentinel — .new_full() instead of torch.full_like
        idx_unsq = indices.unsqueeze(-1)
        sentinel = work.new_full(idx_unsq.shape, fill_value)
        work.scatter_(-1, idx_unsq, sentinel)

    # Assemble results via pre-allocated tensor + slice assignment — avoids torch.stack
    if k > 0:
        values_out = flat.new_empty(N, k)
        indices_out = flat.new_empty(N, k).long()
        for i, (v, idx) in enumerate(zip(selected_values, selected_indices)):
            values_out[:, i] = v
            indices_out[:, i] = idx
    else:
        values_out = flat[:, :0]
        indices_out = flat[:, :0].long()

    if sorted and k > 1:
        # Use hand-crafted sort module instead of torch.argsort
        sorter = _sort(values_out, dim=-1, descending=largest).indices
        values_out = values_out.gather(-1, sorter)
        indices_out = indices_out.gather(-1, sorter)

    out_shape = moved.shape[:-1] + (k,)
    values_out = values_out.reshape(out_shape).movedim(-1, dim)
    indices_out = indices_out.reshape(out_shape).movedim(-1, dim)
    return TopkResult(values=values_out, indices=indices_out)


def _quick_parity_check() -> None:
    import torch
    torch.manual_seed(0)
    x = torch.randn(3, 7)
    ref = torch.topk(x, 3, dim=1)
    impl = topk(x, 3, dim=1)
    err_v = (ref.values - impl.values).abs().max().item()
    err_i = (ref.indices - impl.indices).abs().max().item()
    print(f"topk values max_abs_err = {err_v}")
    print(f"topk indices max_abs_err = {err_i}")
    assert err_v < 1e-6 and err_i == 0
    print("topk OK")


if __name__ == "__main__":
    _quick_parity_check()
