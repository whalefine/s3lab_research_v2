"""Compare RTL outputs against Python golden .npy files.

Usage example:
    python tracking/python_check.py \
      --golden-dir output/golden/vit_care_relu6_fixed_golden \
      --rtl-dir output/rtl_out/case1 \
      --atol 0.0 --rtol 0.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DiffResult:
    filename: str
    max_abs_diff: float
    mean_abs_diff: float
    allclose: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare golden vs RTL .npy outputs.")
    parser.add_argument("--golden-dir", required=True, help="Directory with golden .npy files.")
    parser.add_argument("--rtl-dir", required=True, help="Directory with RTL output .npy files.")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults to <golden-dir>/golden_manifest.json.",
    )
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for allclose.")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for allclose.")
    parser.add_argument(
        "--strict-dtype",
        action="store_true",
        help="Treat dtype mismatch as failure.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional output path for machine-readable report JSON.",
    )
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> Dict:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _golden_file_list(golden_dir: Path, manifest: Dict) -> List[str]:
    files = manifest.get("all_npy_files")
    if isinstance(files, list) and files:
        return sorted([f for f in files if f.endswith(".npy")])
    return sorted([p.name for p in golden_dir.glob("*.npy")])


def _compare_arrays(
    g: np.ndarray,
    r: np.ndarray,
    atol: float,
    rtol: float,
) -> Tuple[bool, float, float]:
    g64 = g.astype(np.float64, copy=False)
    r64 = r.astype(np.float64, copy=False)
    diff = np.abs(g64 - r64)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    ok = bool(np.allclose(g64, r64, atol=atol, rtol=rtol))
    return ok, max_abs, mean_abs


def main() -> None:
    args = _parse_args()
    golden_dir = Path(args.golden_dir).resolve()
    rtl_dir = Path(args.rtl_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else golden_dir / "golden_manifest.json"

    manifest = _load_manifest(manifest_path)
    golden_files = _golden_file_list(golden_dir, manifest)

    missing: List[str] = []
    shape_mismatch: List[Dict] = []
    dtype_mismatch: List[Dict] = []
    value_fail: List[DiffResult] = []
    value_pass = 0

    for name in golden_files:
        g_path = golden_dir / name
        r_path = rtl_dir / name

        if not r_path.exists():
            missing.append(name)
            continue

        g_arr = np.load(g_path)
        r_arr = np.load(r_path)

        if g_arr.shape != r_arr.shape:
            shape_mismatch.append(
                {"filename": name, "golden_shape": list(g_arr.shape), "rtl_shape": list(r_arr.shape)}
            )
            continue

        if g_arr.dtype != r_arr.dtype:
            dtype_mismatch.append(
                {"filename": name, "golden_dtype": str(g_arr.dtype), "rtl_dtype": str(r_arr.dtype)}
            )
            if args.strict_dtype:
                continue

        ok, max_abs, mean_abs = _compare_arrays(g_arr, r_arr, atol=args.atol, rtol=args.rtol)
        if ok:
            value_pass += 1
        else:
            value_fail.append(
                DiffResult(filename=name, max_abs_diff=max_abs, mean_abs_diff=mean_abs, allclose=False)
            )

    value_fail_sorted = sorted(value_fail, key=lambda x: x.max_abs_diff, reverse=True)
    summary = {
        "golden_dir": str(golden_dir),
        "rtl_dir": str(rtl_dir),
        "manifest": str(manifest_path),
        "total_golden_files": len(golden_files),
        "missing_files": len(missing),
        "shape_mismatch": len(shape_mismatch),
        "dtype_mismatch": len(dtype_mismatch),
        "value_pass": value_pass,
        "value_fail": len(value_fail_sorted),
        "atol": args.atol,
        "rtol": args.rtol,
        "strict_dtype": args.strict_dtype,
    }

    print("=== Python Check Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if missing:
        print("\n[Missing files] (first 20)")
        for x in missing[:20]:
            print(x)

    if shape_mismatch:
        print("\n[Shape mismatch] (first 20)")
        for x in shape_mismatch[:20]:
            print(x)

    if dtype_mismatch:
        print("\n[Dtype mismatch] (first 20)")
        for x in dtype_mismatch[:20]:
            print(x)

    if value_fail_sorted:
        print("\n[Value mismatch top 30]")
        for item in value_fail_sorted[:30]:
            print(
                f"{item.filename}: max_abs_diff={item.max_abs_diff:.10g}, "
                f"mean_abs_diff={item.mean_abs_diff:.10g}"
            )

    if args.report_json:
        report = {
            "summary": summary,
            "missing": missing,
            "shape_mismatch": shape_mismatch,
            "dtype_mismatch": dtype_mismatch,
            "value_fail": [
                {
                    "filename": x.filename,
                    "max_abs_diff": x.max_abs_diff,
                    "mean_abs_diff": x.mean_abs_diff,
                }
                for x in value_fail_sorted
            ],
        }
        out_path = Path(args.report_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
