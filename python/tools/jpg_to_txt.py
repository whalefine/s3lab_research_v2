#!/usr/bin/env python3
"""將 JPG/PNG 等影像逐像素轉成每行 24 字元（R、G、B 各 8 bit 二進位）的純文字檔。

需要 Pillow 或 OpenCV 其中之一：
  pip install pillow
  或
  pip install opencv-python
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_rgb(path: Path):
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore[misc, assignment]

    if Image is not None:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        raw = im.tobytes()
        return w, h, raw

    try:
        import cv2
    except ImportError as e:
        raise SystemExit(
            "找不到影像讀取套件。請安裝其一：\n"
            "  pip install pillow\n"
            "  或\n"
            "  pip install opencv-python"
        ) from e

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"無法讀取影像: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    raw = rgb.tobytes()
    return w, h, raw


def image_bytes_to_rgb24_lines(w: int, h: int, raw: bytes) -> list[str]:
    """raw 為 row-major 的 RGBRGB...，每像素 3 bytes。"""
    expected = w * h * 3
    if len(raw) != expected:
        raise ValueError(f"像素資料長度不符：需要 {expected} bytes，實際 {len(raw)}")
    lines: list[str] = []
    for i in range(0, len(raw), 3):
        r, g, b = raw[i], raw[i + 1], raw[i + 2]
        lines.append(f"{r:08b}{g:08b}{b:08b}")
    return lines


def main() -> None:
    repo_python = Path(__file__).resolve().parents[1]
    default_in = (
        repo_python
        / "data"
        / "uav123"
        / "UAV123"
        / "data_seq"
        / "UAV123"
        / "boat2"
        / "000015.jpg"
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=default_in,
        help="輸入影像路徑（預設為 boat2/000015.jpg）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="輸出 .txt；未指定時為輸入檔同目錄、副檔名改為 .txt",
    )
    args = p.parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"找不到輸入檔: {inp}")

    out = args.output
    if out is None:
        out = inp.with_suffix(".txt")
    else:
        out = out.resolve()

    w, h, raw = _load_rgb(inp)
    lines = image_bytes_to_rgb24_lines(w, h, raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    print(f"影像 {w}x{h}，共 {len(lines)} 像素 -> {out}")


if __name__ == "__main__":
    main()
