"""
彙整多個 dataset 的 AUC / Precision（與 calculate_metrics 相同之 eval_data）與 profile_model 的 GPU FPS，
輸出對齊之文字表格（stdout；可選寫入檔案）。

使用前須已對清單內各 dataset 跑過 test.py（與建議之 calculate_metrics 以產生 eval_data）。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import _init_paths  # noqa: F401
import numpy as np
import torch

from lib.test.analysis.plot_results import check_and_load_precomputed_results, get_auc_curve, get_prec_curve
from lib.test.evaluation import get_dataset, trackerlist

DISPLAY_NAMES = {
    "dtb70": "DTB70",
    "uavdt": "UAVDT",
    "uav123": "UAV123",
    "uavtrack112": "UAVTrack112",
    "uavtrack": "UAVTrack112_L",
    "uav123_10fps": "UAV123_10fps",
}


def _display(ds: str) -> str:
    return DISPLAY_NAMES.get(ds.lower().strip(), ds)


def read_dataset_list(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def collect_auc_prec(tracker: str, param: str, dataset_name: str, run_id: int | None) -> tuple[float, float]:
    trackers = trackerlist(
        name=tracker,
        parameter_name=param,
        dataset_name=dataset_name,
        run_ids=run_id,
        display_name=f"{tracker}_{param}",
    )
    dataset = get_dataset(dataset_name)
    eval_data = check_and_load_precomputed_results(trackers, dataset, dataset_name)
    valid_sequence = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    ave_overlap = torch.tensor(eval_data["ave_success_rate_plot_overlap"])
    ave_center = torch.tensor(eval_data["ave_success_rate_plot_center"])
    _, auc = get_auc_curve(ave_overlap, valid_sequence)
    _, prec = get_prec_curve(ave_center, valid_sequence)
    return float(auc[0].item()), float(prec[0].item())


def run_profile_fps(script: str, config: str, python_root: Path) -> float | None:
    cmd = [sys.executable, str(python_root / "tracking" / "profile_model.py"), "--script", script, "--config", config]
    proc = subprocess.run(cmd, cwd=str(python_root), capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"FPS is\s+([0-9.]+)\s*fps", text, re.I)
    if not m:
        print(text[-2000:] if len(text) > 2000 else text, file=sys.stderr)
        return None
    return float(m.group(1))


def format_text_table_simple(
    row_label: str,
    datasets: list[str],
    auc_vals: list[float],
    prec_vals: list[float],
    avg_auc: float,
    avg_p: float,
    fps_gpu: float | None,
    fps_cpu: str,
) -> str:
    """較穩定的單層標題表格（每欄一個欄位名）。"""
    headers: list[str] = ["Model"]
    for ds in datasets:
        d = _display(ds)
        headers.append(f"{d} AUC(%)")
        headers.append(f"{d} P(%)")
    headers.extend(["Avg. AUC(%)", "Avg. P(%)", "FPS GPU", "FPS CPU"])

    gpu_str = "—" if fps_gpu is None else f"{fps_gpu:.2f}"
    row: list[str] = [row_label]
    for i in range(len(datasets)):
        row.append(f"{auc_vals[i]:.2f}")
        row.append(f"{prec_vals[i]:.2f}")
    row.extend([f"{avg_auc:.2f}", f"{avg_p:.2f}", gpu_str, fps_cpu])

    rows = [headers, row]
    ncols = len(headers)
    widths = [max(len(str(rows[i][c])) for i in range(2)) for c in range(ncols)]

    def fmt_line(cells: list[str]) -> str:
        return " | ".join(str(cells[i]).ljust(widths[i]) for i in range(ncols))

    sep = "-+-".join("-" * w for w in widths)
    out = [
        sep,
        fmt_line(rows[0]),
        sep,
        fmt_line(rows[1]),
        sep,
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 結果彙整（文字表格）")
    parser.add_argument("--tracker", type=str, required=True)
    parser.add_argument("--param", type=str, required=True)
    parser.add_argument("--script", type=str, default=None, help="profile_model 之 --script，預設與 tracker 相同")
    parser.add_argument("--dataset_list", type=str, required=True)
    parser.add_argument("--runid", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="可選：將表格寫入此文字檔（仍會印出至 stdout）")
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument("--fps_gpu", type=float, default=None)
    parser.add_argument("--fps_cpu", type=str, default="—")
    parser.add_argument("--json", type=str, default=None, help="可選：另存 JSON 路徑")
    args = parser.parse_args()

    python_root = Path(__file__).resolve().parent.parent
    script = args.script or args.tracker
    datasets = read_dataset_list(args.dataset_list)
    if not datasets:
        raise SystemExit("清單中沒有任何 dataset")

    auc_vals: list[float] = []
    prec_vals: list[float] = []
    for ds in datasets:
        a, p = collect_auc_prec(args.tracker, args.param, ds, args.runid)
        auc_vals.append(a)
        prec_vals.append(p)

    avg_auc = float(np.mean(auc_vals))
    avg_p = float(np.mean(prec_vals))

    fps_gpu = args.fps_gpu
    if not args.skip_profile and fps_gpu is None:
        print("執行 profile_model.py 取得 GPU FPS …", file=sys.stderr)
        fps_gpu = run_profile_fps(script, args.param, python_root)
        if fps_gpu is None:
            print("警告：無法從 profile_model 解析 FPS，GPU 欄位將顯示為 —", file=sys.stderr)

    row_label = f"{args.tracker}_{args.param}"
    table = format_text_table_simple(
        row_label,
        datasets,
        auc_vals,
        prec_vals,
        avg_auc,
        avg_p,
        fps_gpu,
        args.fps_cpu,
    )
    print(table)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print(f"（已寫入 {args.out}）", file=sys.stderr)

    if args.json:
        summary = {
            "tracker": args.tracker,
            "param": args.param,
            "datasets": {ds: {"auc": auc_vals[i], "precision": prec_vals[i]} for i, ds in enumerate(datasets)},
            "avg_auc": avg_auc,
            "avg_precision": avg_p,
            "fps_gpu": fps_gpu,
            "fps_cpu_display": args.fps_cpu,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"（已寫入 JSON {args.json}）", file=sys.stderr)


if __name__ == "__main__":
    main()
