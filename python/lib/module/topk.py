"""RTL-friendly top-k implementation.

Target API:
- torch.topk

Upstream references:
- https://github.com/pytorch/pytorch/blob/main/torch/_refs/__init__.py
- Runtime dispatch is handled by ATen/C++ kernels.

Notes:
- This implementation uses iterative selection on the target dimension.
- The dataflow is closer to a repeated compare-select block than the fused
- backend implementation.
"""

from __future__ import annotations

from collections import namedtuple

import torch


TopkResult = namedtuple("TopkResult", ["values", "indices"])


def topk(
    input: torch.Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> TopkResult:
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
    work = flat.clone()

    if input.dtype.is_floating_point:
        fill_value = float("-inf") if largest else float("inf")
    else:
        info = torch.iinfo(input.dtype)
        fill_value = info.min if largest else info.max

    selected_values = []
    selected_indices = []
    for _ in range(k):
        if largest:
            values, indices = work.max(dim=-1)
        else:
            values, indices = work.min(dim=-1)
        selected_values.append(values)
        selected_indices.append(indices)
        work.scatter_(-1, indices.unsqueeze(-1), torch.full_like(indices.unsqueeze(-1), fill_value, dtype=work.dtype))

    values_out = torch.stack(selected_values, dim=-1) if k > 0 else flat[:, :0]
    indices_out = torch.stack(selected_indices, dim=-1) if k > 0 else torch.empty_like(flat[:, :0], dtype=torch.long)

    if sorted and k > 1:
        sorter = torch.argsort(values_out, dim=-1, descending=largest)
        values_out = values_out.gather(-1, sorter)
        indices_out = indices_out.gather(-1, sorter)

    out_shape = moved.shape[:-1] + (k,)
    values_out = values_out.reshape(out_shape).movedim(-1, dim)
    indices_out = indices_out.reshape(out_shape).movedim(-1, dim)
    return TopkResult(values=values_out, indices=indices_out)


@torch.no_grad()
def _quick_parity_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 7)
    ref = torch.topk(x, 3, dim=1)
    impl = topk(x, 3, dim=1)
    err_v = (ref.values - impl.values).abs().max().item()
    err_i = (ref.indices - impl.indices).abs().max().item()
    print("topk values max_abs_err =", err_v)
    print("topk indices max_abs_err =", err_i)
    assert err_v < 1e-6
    assert err_i == 0


if __name__ == "__main__":
    _quick_parity_check()
