"""從 SGLATrack checkpoint（.pth / .pth.tar）讀取參數，畫「元素級」數值直方圖（每個 state_dict key 一張圖）。

預設只處理與 ViT Block 內部一致、與 weight_minmax TSV 對齊的鍵名：
  backbone.blocks.<i>.norm1.(weight|bias)
  backbone.blocks.<i>.norm2.(weight|bias)
  backbone.blocks.<i>.attn.(qkv|proj).(weight|bias)
  backbone.blocks.<i>.mlp.fc[12].(weight|bias)

可用 `--scope backbone_blocks` 把同一目錄下 checkpoint 裡所有 `backbone.blocks.*` 的浮點參數都畫出來
（例如某版多出其他子模組權重也會一併輸出）。

用法::

    cd python
    PYTHONPATH=. python tracking/plot_checkpoint_param_histograms.py \\
        --checkpoint output/checkpoints/train/sglatrack/vit_coco_uav123_care_relu6/sglatrack_ep0050.pth.tar \\
        --out-dir output/analysis/ckpt_hist_care_relu6_ep50

需要 matplotlib。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import torch

prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

# 與使用者截圖 / weight TSV 一致的 Block 內部參數
_BLOCK_INTERNAL_RE = re.compile(
    r"^backbone\.blocks\.\d+\."
    r"(norm1\.(weight|bias)|norm2\.(weight|bias)|"
    r"attn\.(qkv|proj)\.(weight|bias)|mlp\.fc[12]\.(weight|bias))$"
)


def _load_state_dict_net(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        state = checkpoint["net"]
    elif isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected dict with key 'net' or a flat tensor state_dict."
        )
    return state


def _flatten_key(key: str) -> str:
    return key.replace(".", "_").replace("/", "_")


def _natural_block_sort_key(name: str) -> Tuple:
    m = re.match(r"^backbone\.blocks\.(\d+)\.(.+)$", name)
    if m:
        return (0, int(m.group(1)), m.group(2))
    return (1, 0, name)


def _select_keys(
    state: Dict[str, torch.Tensor],
    scope: str,
    extra_regex: List[str],
) -> List[str]:
    extra_res = [re.compile(p) for p in extra_regex if p.strip()]
    keys: List[str] = []

    for k, t in state.items():
        if not torch.is_tensor(t) or not t.dtype.is_floating_point or t.numel() == 0:
            continue

        if scope == "block_internals":
            base = bool(_BLOCK_INTERNAL_RE.match(k))
        elif scope == "backbone_blocks":
            base = k.startswith("backbone.blocks.")
        elif scope == "all_float":
            base = True
        else:
            raise ValueError(f"Unknown scope: {scope}")

        extra = any(er.match(k) for er in extra_res) if extra_res else False
        if base or extra:
            keys.append(k)

    return sorted(set(keys), key=_natural_block_sort_key)


def _sample_flat(t: torch.Tensor, max_samples: int, seed: int) -> torch.Tensor:
    """回傳 1D float32 CPU tensor（可能為抽樣）。"""
    x = t.detach().float().cpu().contiguous().view(-1)
    n = x.numel()
    if max_samples > 0 and n > max_samples:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        perm = torch.randperm(n, generator=g, device="cpu")[:max_samples]
        x = x[perm]
    return x


def _plot_one(
    key: str,
    values: torch.Tensor,
    numel_total: int,
    out_path: str,
    bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmin = float(values.min().item())
    vmax = float(values.max().item())
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    n_plot = int(values.numel())

    fig, ax = plt.subplots(figsize=(7, 4))
    if vmin == vmax or bins < 1:
        ax.bar([vmin], [float(n_plot)], width=0.1 if vmin == vmax else (vmax - vmin) * 0.1, color="#3498db", edgecolor="white")
    else:
        counts = torch.histc(values, bins=bins, min=vmin, max=vmax).float()
        edges = torch.linspace(vmin, vmax, bins + 1)
        left = edges[:-1]
        right = edges[1:]
        centers = (left + right) * 0.5
        bar_w = float((right - left).mean()) * 0.98
        ax.bar(
            centers.tolist(),
            counts.tolist(),
            width=bar_w,
            align="center",
            color="#3498db",
            edgecolor="white",
            alpha=0.88,
        )
    sub = f"n_total={numel_total}  n_hist={n_plot}" + (
        "  (subsampled)" if n_plot < numel_total else ""
    )
    ax.set_xlabel("parameter value (tensor element; may be subsampled for histogram)")
    ax.set_ylabel("count of elements (per bin)")
    ax.set_title(
        f"{key}\n{sub}  min={vmin:.4g}  max={vmax:.4g}  mean={mean:.4g}  std={std:.4g}",
        fontsize=9,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pth.tar / .pth (uses checkpoint['net'] if present)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Directory for one PNG per key (default: output/analysis/ckpt_hist_<ckpt_stem>/)",
    )
    p.add_argument(
        "--scope",
        type=str,
        choices=("block_internals", "backbone_blocks", "all_float"),
        default="block_internals",
        help="block_internals: norm1/2, attn qkv/proj, mlp fc1/fc2 only; "
        "backbone_blocks: all float tensors under backbone.blocks.*; "
        "all_float: entire state_dict float tensors",
    )
    p.add_argument(
        "--extra-regex",
        action="append",
        default=[],
        help="Extra key regex (repeatable), e.g. '^backbone\\\\.norm\\\\.weight$'. "
        "Always OR-ed: matching keys are plotted in addition to --scope.",
    )
    p.add_argument("--bins", type=int, default=120, help="Histogram bins per figure")
    p.add_argument(
        "--max-samples",
        type=int,
        default=2_000_000,
        help="Max elements per tensor for histogram (0 = use all); large tensors subsampled",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling")
    p.add_argument(
        "--ext",
        type=str,
        default="png",
        choices=("png", "pdf"),
        help="Output image extension",
    )
    args = p.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    state = _load_state_dict_net(ckpt)
    keys = _select_keys(state, args.scope, args.extra_regex)
    if not keys:
        raise SystemExit("No keys selected. Try --scope backbone_blocks or --extra-regex.")

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        stem = os.path.splitext(os.path.basename(ckpt))[0]
        if stem.endswith(".pth"):
            stem = stem[:-4]
        out_dir = os.path.join(prj_path, "output", "analysis", f"ckpt_hist_{stem}")

    os.makedirs(out_dir, exist_ok=True)
    max_s = args.max_samples

    for i, key in enumerate(keys):
        t = state[key]
        numel = int(t.numel())
        values = _sample_flat(t, max_s, args.seed + i)
        fname = _flatten_key(key) + f".{args.ext}"
        out_path = os.path.join(out_dir, fname)
        _plot_one(key, values, numel, out_path, args.bins)
        if (i + 1) % 20 == 0 or i + 1 == len(keys):
            print(f"  {i + 1}/{len(keys)} … {fname}")

    print(f"Done. Wrote {len(keys)} figures under {out_dir}")


if __name__ == "__main__":
    main()
