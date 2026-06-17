"""Backbone-only checker for RTL outputs vs Python golden .npy files.

It compares only files related to backbone pipeline, including:
- pre/post-embed backbone inputs
- merge/pos-drop
- adaptive selector outputs
- backbone block intermediates
- backbone final outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


BACKBONE_PREFIXES = (
    "template_after_patch_embed_out.npy",
    "search_after_patch_embed_out.npy",
    "template_pos_embed.npy",
    "search_pos_embed.npy",
    "template_after_pos_add_out.npy",
    "search_after_pos_add_out.npy",
    "template_post_embed_input.npy",
    "search_post_embed_input.npy",
    "merged_tokens.npy",
    "after_pos_drop_out.npy",
    "adaptive_",
    "backbone_blocks_",
    "backbone_after_",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backbone-only Python/RTL checker.")
    parser.add_argument("--golden-dir", required=True)
    parser.add_argument("--rtl-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_backbone_file(name: str) -> bool:
    for p in BACKBONE_PREFIXES:
        if name.startswith(p):
            return True
    return False


def _golden_files(golden_dir: Path, manifest: Dict) -> List[str]:
    files = manifest.get("all_npy_files")
    if isinstance(files, list) and files:
        names = [f for f in files if f.endswith(".npy")]
    else:
        names = [p.name for p in golden_dir.glob("*.npy")]
    return sorted([n for n in names if _is_backbone_file(n)])


def main() -> None:
    args = _parse_args()
    golden_dir = Path(args.golden_dir).resolve()
    rtl_dir = Path(args.rtl_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else golden_dir / "golden_manifest.json"
    manifest = _load_manifest(manifest_path)
    names = _golden_files(golden_dir, manifest)

    missing, shape_bad, dtype_bad, value_bad = [], [], [], []
    value_ok = 0

    for n in names:
        gp, rp = golden_dir / n, rtl_dir / n
        if not rp.exists():
            missing.append(n)
            continue

        g = np.load(gp)
        r = np.load(rp)

        if g.shape != r.shape:
            shape_bad.append((n, list(g.shape), list(r.shape)))
            continue

        if g.dtype != r.dtype:
            dtype_bad.append((n, str(g.dtype), str(r.dtype)))

        d = np.abs(g.astype(np.float64, copy=False) - r.astype(np.float64, copy=False))
        max_abs = float(d.max()) if d.size else 0.0
        if np.allclose(g, r, atol=args.atol, rtol=args.rtol):
            value_ok += 1
        else:
            value_bad.append((n, max_abs))

    value_bad.sort(key=lambda x: x[1], reverse=True)

    print("=== Backbone Check Summary ===")
    print(f"golden_dir: {golden_dir}")
    print(f"rtl_dir: {rtl_dir}")
    print(f"manifest: {manifest_path}")
    print(f"total_backbone_files: {len(names)}")
    print(f"missing: {len(missing)}")
    print(f"shape_mismatch: {len(shape_bad)}")
    print(f"dtype_mismatch: {len(dtype_bad)}")
    print(f"value_pass: {value_ok}")
    print(f"value_fail: {len(value_bad)}")

    if missing:
        print("\n[Missing files] (first 20)")
        for x in missing[:20]:
            print(x)

    if shape_bad:
        print("\n[Shape mismatch] (first 20)")
        for n, gs, rs in shape_bad[:20]:
            print(f"{n}: golden={gs}, rtl={rs}")

    if dtype_bad:
        print("\n[Dtype mismatch] (first 20)")
        for n, gd, rd in dtype_bad[:20]:
            print(f"{n}: golden={gd}, rtl={rd}")

    if value_bad:
        print("\n[Value mismatch top 30]")
        for n, d in value_bad[:30]:
            print(f"{n}: max_abs_diff={d:.10g}")


if __name__ == "__main__":
    main()
