"""Pure-numpy SGLATrack backbone + shared-trunk head forward pass from pre-computed post-embed .npy.

使用方式：
    python tracking/run_backbone_numpy_shared_trunk.py \\
      --golden-dir output/golden/<dump_golden_intermediate_out_dir> \\
      --weight-dir output/exported_npy/<export_checkpoint_npy_out_dir> \\
      --output-dir output/golden/<numpy_out_dir>

設計原則：
- 純 numpy，不建立任何 PyTorch model 或呼叫 .forward()
- template_after_pos_add_out.npy / search_after_pos_add_out.npy 以 np.load 從
  --golden-dir 載入，作為 backbone 的真正輸入
- 所有 weight 從 --weight-dir 的 exported npy 載入（float32）
- backbone 每個 op 後的 activation 做 to_fixed_point(8, 8)（np.trunc 向零截斷量化）
- shared-trunk head 對齊 `head_shared_trunk_dump.py` 的節點命名：
  shared_conv1/2（folded BN）+ tail_ctr/size/offset（1x1 conv），ctr/size 做 sigmoid_clamped
- 輸出 npy 命名盡量與 dump_golden_intermediate.py / head_shared_trunk_dump.py 一致，
  方便與 RTL 逐層比對
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Model constants (vit_care_relu6_dim32 + shared-trunk head)
# ---------------------------------------------------------------------------
EMBED_DIM = 32
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
SCALE = HEAD_DIM**-0.5
S = SCALE**0.5
LENS_Z = 64
LENS_X = 256
N_TOKENS = LENS_Z + LENS_X
START_LAYER = 5
LN_EPS = 1e-6
FEAT_SZ = 16
FEAT_LEN = FEAT_SZ * FEAT_SZ
SEARCH_SIZE = 256


# ---------------------------------------------------------------------------
# Numpy primitives
# ---------------------------------------------------------------------------

def to_fixed_point(x: np.ndarray, int_bits: int, frac_bits: int) -> np.ndarray:
    """有號定點數量化：scale → trunc（向零截斷）→ saturate → descale。"""
    scale = 2**frac_bits
    qmin = -(2 ** (int_bits + frac_bits - 1))
    qmax = (2 ** (int_bits + frac_bits - 1)) - 1
    scaled = x.astype(np.float64) * scale
    truncated = np.trunc(scaled)
    saturated = np.clip(truncated, qmin, qmax)
    return (saturated / scale).astype(np.float32)


def fp(x: np.ndarray) -> np.ndarray:
    """Shorthand: to_fixed_point(x, 8, 8)."""
    return to_fixed_point(x, 8, 8)


def layer_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float = LN_EPS,
    inv_sqrt_iter: int = 2,
) -> np.ndarray:
    """硬體友善 LayerNorm（Newton-Raphson inv_sqrt）。"""
    N = x.shape[-1]
    rcp_n = float(round(1.0 / N * 65536)) / 65536.0
    mean = fp(x.sum(axis=-1, keepdims=True) * rcp_n)
    centered = fp(x - mean)
    var = fp((centered.astype(np.float64) ** 2).sum(axis=-1, keepdims=True) * rcp_n)
    inv_std = fp(_inv_sqrt_nr(var + eps, num_iter=inv_sqrt_iter))
    return fp(weight * fp(centered * inv_std) + bias)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Q8.8 fixed-point integer MAC linear（整數 MAC + int64 累加器）。"""
    _SCALE = 1 << 8
    x_int = np.trunc(x.astype(np.float64) * _SCALE).astype(np.int32)
    w_int = np.trunc(weight.astype(np.float64) * _SCALE).astype(np.int32)

    *batch, in_dim = x_int.shape
    out_dim = w_int.shape[0]
    x64 = x_int.reshape(-1, in_dim).astype(np.int64)
    w64 = w_int.astype(np.int64)
    acc_q16 = x64 @ w64.T
    acc_q88 = acc_q16 >> 8
    if bias is not None:
        bias_int = np.trunc(bias.astype(np.float64) * _SCALE).astype(np.int64)
        acc_q88 = acc_q88 + bias_int
    return (acc_q88.reshape(*batch, out_dim).astype(np.float64) / _SCALE).astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(np.float32)


def relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 6.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Sigmoid LUT（模組載入時建立一次）
# ---------------------------------------------------------------------------
_SIGMOID_LUT_N = 64
_SIGMOID_LUT = 1.0 / (1.0 + np.exp(-np.linspace(-8.0, 8.0, _SIGMOID_LUT_N + 1).astype(np.float64)))
_SIGMOID_LUT_INT = np.round(_SIGMOID_LUT * 256).astype(np.int32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """LUT + 線性插值 sigmoid（對齊 RTL 硬體語意）。"""
    x_int = np.round(np.clip(x.astype(np.float64), -8.0, 8.0) * 256.0).astype(np.int32)
    shifted = x_int + 2048
    idx = np.clip(shifted >> 6, 0, _SIGMOID_LUT_N - 1).astype(np.int32)
    frac6 = (shifted & 0x3F).astype(np.int32)
    lo_int = _SIGMOID_LUT_INT[idx]
    hi_int = _SIGMOID_LUT_INT[idx + 1]
    delta = (hi_int - lo_int).astype(np.int32)
    result = (lo_int * 64 + delta * frac6) >> 6
    return (result.astype(np.float64) / 256.0).astype(np.float32)


def sigmoid_clamped(x: np.ndarray) -> np.ndarray:
    """head 的 _sigmoid：LUT sigmoid → Q8.8 → Q8.8 邊界 clamp。"""
    _LB = 1.0 / 256
    _UB = 255.0 / 256
    return np.clip(fp(sigmoid(x)), _LB, _UB).astype(np.float32)


def conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    padding: int = 1,
) -> np.ndarray:
    """Q8.8 fixed-point integer MAC conv2d。"""
    _SCALE = 1 << 8
    N, C_in, H, W = x.shape
    C_out, c_w, kH, kW = weight.shape
    if c_w != C_in:
        raise ValueError(f"conv2d Cin mismatch: x has {C_in}, weight has {c_w}")

    x_int = np.round(x.astype(np.float64) * _SCALE).astype(np.int32)
    w_int = np.round(weight.astype(np.float64) * _SCALE).astype(np.int32)

    if padding:
        x_int = np.pad(
            x_int,
            [(0, 0), (0, 0), (padding, padding), (padding, padding)],
            mode="constant",
            constant_values=0,
        )
    H_out = x_int.shape[2] - kH + 1
    W_out = x_int.shape[3] - kW + 1
    if H_out < 1 or W_out < 1:
        raise ValueError(f"conv2d output size invalid: padded {x_int.shape[2:]}, kernel ({kH},{kW})")

    acc = np.zeros((N, C_out, H_out, W_out), dtype=np.int64)
    for n in range(N):
        for oc in range(C_out):
            w_oc = w_int[oc].astype(np.int64)
            for oh in range(H_out):
                for ow in range(W_out):
                    patch = x_int[n, :, oh : oh + kH, ow : ow + kW].astype(np.int64)
                    acc[n, oc, oh, ow] = np.sum(patch * w_oc)

    acc_q88 = acc >> 8
    if bias is not None:
        bias_int = np.round(bias.astype(np.float64) * _SCALE).astype(np.int64)
        acc_q88 += bias_int[np.newaxis, :, np.newaxis, np.newaxis]

    return (acc_q88.astype(np.float64) / _SCALE).astype(np.float32)


def hann1d(sz: int, centered: bool = True) -> np.ndarray:
    if centered:
        return (0.5 * (1 - np.cos((2 * math.pi / (sz + 1)) * np.arange(1, sz + 1)))).astype(np.float32)
    w = 0.5 * (1 + np.cos((2 * math.pi / (sz + 2)) * np.arange(0, sz // 2 + 1)))
    return np.concatenate([w, w[1 : sz - sz // 2][::-1]]).astype(np.float32)


def hann2d(sz_h: int, sz_w: int, centered: bool = True) -> np.ndarray:
    h = hann1d(sz_h, centered).reshape(-1, 1)
    w = hann1d(sz_w, centered).reshape(1, -1)
    return (h * w).reshape(1, 1, sz_h, sz_w)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_out_dir: Optional[Path] = None
_bi_act_dir: Optional[Path] = None
_bi_wgt_dir: Optional[Path] = None


def write_bi(arr: np.ndarray, base: Path, int_bits: int = 8, frac_bits: int = 8) -> None:
    flat = arr.flatten()
    scale = 1 << frac_bits
    total_w = int_bits + frac_bits
    min_int = -(1 << (int_bits - 1)) * scale
    max_int = (1 << (int_bits - 1)) * scale - 1
    base_str = str(base)
    with open(base_str + ".txt", "w") as f_dec, open(base_str + "_bi.txt", "w") as f_bin:
        for num in flat:
            fixed = int(num * scale)
            if fixed < min_int:
                fixed = min_int
            elif fixed > max_int:
                fixed = max_int
            f_dec.write(f"{fixed / scale:.{frac_bits}f}\n")
            twos = fixed & ((1 << total_w) - 1)
            f_bin.write(format(twos, f"0{total_w}b") + "\n")


def write_wbi(arr: np.ndarray, name: str, int_bits: int = 8, frac_bits: int = 8) -> None:
    if _bi_wgt_dir is not None:
        write_bi(arr, _bi_wgt_dir / name, int_bits, frac_bits)


def save_npy(filename: str, arr: np.ndarray) -> None:
    if _out_dir is not None:
        np.save(_out_dir / filename, arr)
    if _bi_act_dir is not None:
        stem = filename[:-4] if filename.endswith(".npy") else filename
        write_bi(arr, _bi_act_dir / stem)


def load_w(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Attention (CARE ReLU6 fixed-point)
# ---------------------------------------------------------------------------

def _inv_sqrt_nr(v: np.ndarray, num_iter: int = 2) -> np.ndarray:
    v64 = v.astype(np.float64)
    y = 1.0 / np.sqrt(v64 + 1e-30)
    for _ in range(num_iter):
        y = y * (1.5 - 0.5 * v64 * y * y)
    return y.astype(np.float32)


def _recip_nr(x: np.ndarray, num_iter: int = 1) -> np.ndarray:
    x64 = x.astype(np.float64)
    y = 1.0 / x64
    for _ in range(num_iter):
        y = y * (2.0 - x64 * y)
    return y.astype(np.float32)


def attention_forward(x: np.ndarray, block_idx: int, wp: Path) -> np.ndarray:
    B, N, C = x.shape
    H, d = NUM_HEADS, HEAD_DIM
    pf = f"backbone_blocks_{block_idx}"
    lp = wp / "linearParam"

    qkv_w = load_w(lp / f"{pf}_attn_qkv_weight.npy")
    write_wbi(qkv_w, f"{pf}_attn_qkv_weight")
    qkv_b = load_w(lp / f"{pf}_attn_qkv_bias.npy")
    write_wbi(qkv_b, f"{pf}_attn_qkv_bias")
    proj_w = load_w(lp / f"{pf}_attn_proj_weight.npy")
    write_wbi(proj_w, f"{pf}_attn_proj_weight")
    proj_b = load_w(lp / f"{pf}_attn_proj_bias.npy")
    write_wbi(proj_b, f"{pf}_attn_proj_bias")

    qkv = linear(x, qkv_w, qkv_b).reshape(B, N, 3, H, d).transpose(2, 0, 3, 1, 4)
    q, k, v = fp(qkv[0]), fp(qkv[1]), fp(qkv[2])
    save_npy(f"{pf}_attn_after_qkv_q.npy", q)
    save_npy(f"{pf}_attn_after_qkv_k.npy", k)
    save_npy(f"{pf}_attn_after_qkv_v.npy", v)

    q = fp(relu6(fp(q * S)))
    k = fp(relu6(fp(k * S)))
    v = fp(v)

    rcp_n = float(round(1.0 / N * 65536)) / 65536.0
    k_mean = fp(k.sum(axis=-2, keepdims=True) * rcp_n)
    qk_mean = fp(q @ k_mean.transpose(0, 1, 3, 2))
    qk_mean_eps = fp(np.maximum(qk_mean, 1.0 / 256))
    z_recip = fp(_recip_nr(qk_mean_eps, num_iter=1))
    kv = fp((k.transpose(0, 1, 3, 2) @ v) * rcp_n)
    attn_out = fp(fp(q @ kv) * z_recip)

    attn_out = fp(attn_out.transpose(0, 2, 1, 3).reshape(B, N, C))
    attn_out = fp(linear(attn_out, proj_w, proj_b))
    save_npy(f"{pf}_after_attn_attn_out.npy", attn_out)
    return attn_out


def block_forward(x: np.ndarray, block_idx: int, wp: Path) -> np.ndarray:
    pf = f"backbone_blocks_{block_idx}"
    lp = wp / "linearParam"
    lap = wp / "layerParam"

    norm1_w = load_w(lap / f"{pf}_norm1_weight.npy")
    write_wbi(norm1_w, f"{pf}_norm1_weight")
    norm1_b = load_w(lap / f"{pf}_norm1_bias.npy")
    write_wbi(norm1_b, f"{pf}_norm1_bias")
    norm2_w = load_w(lap / f"{pf}_norm2_weight.npy")
    write_wbi(norm2_w, f"{pf}_norm2_weight")
    norm2_b = load_w(lap / f"{pf}_norm2_bias.npy")
    write_wbi(norm2_b, f"{pf}_norm2_bias")
    fc1_w = load_w(lp / f"{pf}_mlp_fc1_weight.npy")
    write_wbi(fc1_w, f"{pf}_mlp_fc1_weight")
    fc1_b = load_w(lp / f"{pf}_mlp_fc1_bias.npy")
    write_wbi(fc1_b, f"{pf}_mlp_fc1_bias")
    fc2_w = load_w(lp / f"{pf}_mlp_fc2_weight.npy")
    write_wbi(fc2_w, f"{pf}_mlp_fc2_weight")
    fc2_b = load_w(lp / f"{pf}_mlp_fc2_bias.npy")
    write_wbi(fc2_b, f"{pf}_mlp_fc2_bias")

    x_norm1 = fp(layer_norm(x, norm1_w, norm1_b))
    save_npy(f"{pf}_after_norm1_out.npy", x_norm1)
    attn_out = attention_forward(x_norm1, block_idx, wp)
    x = fp(x + attn_out)
    save_npy(f"{pf}_after_residual_add1_out.npy", x)

    x_norm2 = fp(layer_norm(x, norm2_w, norm2_b))
    save_npy(f"{pf}_after_norm2_out.npy", x_norm2)
    mlp_out = fp(linear(fp(relu(fp(linear(x_norm2, fc1_w, fc1_b)))), fc2_w, fc2_b))
    save_npy(f"{pf}_mlp_after_mlp_out.npy", mlp_out)
    x = fp(x + mlp_out)
    save_npy(f"{pf}_after_block_out.npy", x)
    return x


# ---------------------------------------------------------------------------
# Shared-trunk head（shared_conv1/2 + tail_ctr/size/offset）
# ---------------------------------------------------------------------------

def head_shared_trunk(opt_feat: np.ndarray, wp: Path):
    """對齊 head_shared_trunk_dump.CenterPredictorSharedTrunkDump.get_score_map 的命名。"""
    fb = wp / "foldedBN"
    cp = wp / "convParam"

    # shared_conv1（Conv+BN folded）+ ReLU
    w1 = load_w(fb / "box_head_shared_conv1_folded_weight.npy")
    b1 = load_w(fb / "box_head_shared_conv1_folded_bias.npy")
    write_wbi(w1, "box_head_shared_conv1_folded_weight")
    write_wbi(b1, "box_head_shared_conv1_folded_bias")
    x1 = fp(relu(conv2d(opt_feat, w1, b1, padding=1)))
    save_npy("box_head_shared_after_conv1_out.npy", x1)

    # shared_conv2（Conv+BN folded）+ ReLU
    w2 = load_w(fb / "box_head_shared_conv2_folded_weight.npy")
    b2 = load_w(fb / "box_head_shared_conv2_folded_bias.npy")
    write_wbi(w2, "box_head_shared_conv2_folded_weight")
    write_wbi(b2, "box_head_shared_conv2_folded_bias")
    x2 = fp(relu(conv2d(x1, w2, b2, padding=1)))
    save_npy("box_head_shared_after_conv2_out.npy", x2)

    # tail 1x1 conv（無 BN）
    w_ctr = load_w(cp / "box_head_tail_ctr_weight.npy")
    b_ctr = load_w(cp / "box_head_tail_ctr_bias.npy")
    write_wbi(w_ctr, "box_head_tail_ctr_weight")
    write_wbi(b_ctr, "box_head_tail_ctr_bias")
    raw_ctr = fp(conv2d(x2, w_ctr, b_ctr, padding=0))
    save_npy("box_head_tail_ctr_after_conv_out.npy", raw_ctr)

    w_size = load_w(cp / "box_head_tail_size_weight.npy")
    b_size = load_w(cp / "box_head_tail_size_bias.npy")
    write_wbi(w_size, "box_head_tail_size_weight")
    write_wbi(b_size, "box_head_tail_size_bias")
    raw_size = fp(conv2d(x2, w_size, b_size, padding=0))
    save_npy("box_head_tail_size_after_conv_out.npy", raw_size)

    w_off = load_w(cp / "box_head_tail_offset_weight.npy")
    b_off = load_w(cp / "box_head_tail_offset_bias.npy")
    write_wbi(w_off, "box_head_tail_offset_weight")
    write_wbi(b_off, "box_head_tail_offset_bias")
    raw_off = fp(conv2d(x2, w_off, b_off, padding=0))
    save_npy("box_head_tail_offset_after_conv_out.npy", raw_off)

    score_map_ctr = sigmoid_clamped(raw_ctr)
    score_map_size = sigmoid_clamped(raw_size)
    score_map_offset = raw_off
    save_npy("box_head_tail_ctr_after_sigmoid_out.npy", score_map_ctr)
    save_npy("box_head_tail_size_after_sigmoid_out.npy", score_map_size)
    save_npy("box_head_tail_offset_final_out.npy", score_map_offset)

    return score_map_ctr, score_map_size, score_map_offset


def cal_bbox(score_map_ctr: np.ndarray, size_map: np.ndarray, offset_map: np.ndarray) -> np.ndarray:
    flat = score_map_ctr.reshape(1, -1)
    idx = int(np.argmax(flat, axis=1)[0])
    idx_y, idx_x = idx // FEAT_SZ, idx % FEAT_SZ

    size = size_map.reshape(1, 2, -1)[:, :, idx]
    offset = offset_map.reshape(1, 2, -1)[:, :, idx]

    offset_x_q88 = int(round(float(offset[0, 0]) * 256))
    offset_y_q88 = int(round(float(offset[0, 1]) * 256))
    cx = ((int(idx_x) * 256 + offset_x_q88) >> 4) / 256.0
    cy = ((int(idx_y) * 256 + offset_y_q88) >> 4) / 256.0
    w = float(size[0, 0])
    h = float(size[0, 1])
    return np.array([[cx, cy, w, h]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Tracker post-processing helpers
# ---------------------------------------------------------------------------

def map_box_back(state_xywh: list, pred_box_cxcywh: list, resize_factor: float) -> list:
    cx_prev = state_xywh[0] + 0.5 * state_xywh[2]
    cy_prev = state_xywh[1] + 0.5 * state_xywh[3]
    cx, cy, w, h = pred_box_cxcywh
    half_side = 0.5 * SEARCH_SIZE / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def clip_box_numpy(box: list, H: int, W: int, margin: int = 10) -> list:
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    return [x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pure-numpy SGLATrack backbone + shared-trunk head.")
    p.add_argument("--golden-dir", required=True, help="含 template_after_pos_add_out.npy / search_after_pos_add_out.npy 的目錄")
    p.add_argument(
        "--weight-dir",
        default="output/exported_npy/vit_coco_uav123_care_relu6_ep0050_all",
        help="exported weight npy 根目錄（來自 export_checkpoint_npy.py）",
    )
    p.add_argument("--output-dir", required=True, help="計算結果 npy 輸出目錄")
    return p.parse_args()


def main() -> None:
    global _out_dir, _bi_act_dir, _bi_wgt_dir
    args = parse_args()

    gd = Path(args.golden_dir)
    wp = Path(args.weight_dir)
    _out_dir = Path(args.output_dir)
    _out_dir.mkdir(parents=True, exist_ok=True)
    _bi_act_dir = _out_dir / "Activation"
    _bi_wgt_dir = _out_dir / "Weight"
    _bi_act_dir.mkdir(parents=True, exist_ok=True)
    _bi_wgt_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {}
    manifest_path = gd / "golden_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    # Step 1: load post-embed input
    z = fp(np.load(gd / "template_after_pos_add_out.npy").astype(np.float32))
    x = fp(np.load(gd / "search_after_pos_add_out.npy").astype(np.float32))
    save_npy("template_post_embed_input.npy", z)
    save_npy("search_post_embed_input.npy", x)

    # Step 2: combine_tokens + pos_drop
    merged = fp(np.concatenate([z, x], axis=1))
    save_npy("merged_tokens.npy", merged)
    x = fp(merged)
    save_npy("after_pos_drop_out.npy", x)

    # Step 3: blocks 0..START_LAYER
    for i in range(START_LAYER + 1):
        x = block_forward(x, i, wp)

    # Step 4: adaptive selector（優先用 golden）
    golden_topk_path = gd / "adaptive_sorted_topk_indices.npy"
    golden_pro_path = gd / "adaptive_pro.npy"

    if golden_topk_path.exists():
        sorted_topk = np.load(golden_topk_path)
        selected_block = int(sorted_topk.flat[0])
        pro = np.load(golden_pro_path).astype(np.float32) if golden_pro_path.exists() else None
        print(f"[adaptive] loaded from golden: selected_block={selected_block}")
    else:
        lp = wp / "linearParam"
        mlp1_w = load_w(lp / "backbone_MLP_fc1_weight.npy")
        write_wbi(mlp1_w, "backbone_MLP_fc1_weight")
        mlp1_b = load_w(lp / "backbone_MLP_fc1_bias.npy")
        write_wbi(mlp1_b, "backbone_MLP_fc1_bias")
        mlp2_w = load_w(lp / "backbone_MLP_fc2_weight.npy")
        write_wbi(mlp2_w, "backbone_MLP_fc2_weight")
        mlp2_b = load_w(lp / "backbone_MLP_fc2_bias.npy")
        write_wbi(mlp2_b, "backbone_MLP_fc2_bias")
        mlp_in = x[:, :, 0]
        h_mlp = relu(linear(mlp_in, mlp1_w, mlp1_b))
        pro = sigmoid(linear(h_mlp, mlp2_w, mlp2_b))
        selected_idx = int(np.argmax(pro, axis=1)[0])
        selected_block = selected_idx + START_LAYER + 1
        print(f"[adaptive] computed from numpy (no golden): selected_block={selected_block}")

    save_npy("adaptive_pro.npy", pro if pro is not None else np.zeros((1, 6), dtype=np.float32))
    save_npy("adaptive_sorted_topk_indices.npy", np.array([[selected_block]], dtype=np.int64))
    save_npy("adaptive_selected_layer_index.npy", np.array([selected_block], dtype=np.int64))

    # Step 5: selected block
    x = block_forward(x, selected_block, wp)

    # Step 6: final norm
    save_npy("backbone_after_recover_tokens_out.npy", fp(x))
    lap = wp / "layerParam"
    norm_w = load_w(lap / "backbone_norm_weight.npy")
    write_wbi(norm_w, "backbone_norm_weight")
    norm_b = load_w(lap / "backbone_norm_bias.npy")
    write_wbi(norm_b, "backbone_norm_bias")
    backbone_out = fp(layer_norm(x, norm_w, norm_b))
    save_npy("backbone_after_norm_backbone_out.npy", backbone_out)

    # Step 7: forward_head reshape → opt_feat [1, C, 16, 16]
    enc_opt = backbone_out[:, -FEAT_LEN:]
    opt = enc_opt[:, :, :, np.newaxis]
    opt = opt.transpose(0, 3, 2, 1)
    opt_feat = opt.reshape(-1, EMBED_DIM, FEAT_SZ, FEAT_SZ)

    # Step 8: shared-trunk head
    save_npy("box_head_head_input.npy", opt_feat)
    score_map_ctr, size_map, offset_map = head_shared_trunk(opt_feat, wp)
    save_npy("box_head_after_forward_head_score_map.npy", score_map_ctr)
    save_npy("box_head_after_forward_head_size_map.npy", size_map)
    save_npy("box_head_after_forward_head_offset_map.npy", offset_map)

    # Step 9: cal_bbox → pred_boxes
    bbox = cal_bbox(score_map_ctr, size_map, offset_map)
    bbox_q = fp(bbox)
    save_npy("box_head_after_cal_bbox_bbox.npy", bbox_q)
    pred_boxes = fp(bbox_q.reshape(1, 1, 4))
    save_npy("box_head_after_forward_head_pred_boxes.npy", pred_boxes)

    # Step 10: tracker post-processing（若有 manifest）
    if manifest:
        x_resize_factor = float(manifest.get("search_crop_resize_factor", 1.0))
        state = manifest.get("init_bbox_xywh", [0.0, 0.0, 1.0, 1.0])
        search_size = int(manifest.get("search_size", SEARCH_SIZE))
        frame2_path = manifest.get("frame2", "")
        H, W = 1080, 1920
        if frame2_path and os.path.exists(frame2_path):
            import cv2

            img = cv2.imread(frame2_path)
            if img is not None:
                H, W = img.shape[:2]

        window = hann2d(FEAT_SZ, FEAT_SZ, centered=True)
        response = fp(window * score_map_ctr)
        save_npy("tracker_after_output_window_response.npy", response)

        bbox_after = fp(cal_bbox(response, size_map, offset_map))
        save_npy("tracker_after_cal_bbox_bbox.npy", bbox_after)

        pred_box = (bbox_after[0] * search_size / x_resize_factor).tolist()
        mapped = map_box_back(state, pred_box, x_resize_factor)
        save_npy("tracker_after_map_box_back_bbox.npy", fp(np.array(mapped, dtype=np.float32)))

        final_bbox = clip_box_numpy(mapped, H, W, margin=10)
        save_npy("tracker_after_final_bbox_bbox.npy", fp(np.array(final_bbox, dtype=np.float32)))
        x1, y1, bw, bh = final_bbox
        print(f"[tracker] 最終 bbox（frame2 像素 xywh）: x1={x1:.4f}, y1={y1:.4f}, w={bw:.4f}, h={bh:.4f}")
    else:
        print("[WARNING] golden_manifest.json not found; skipping tracker post-processing.")
        print("[tracker] 最終 bbox: 未計算（無 manifest，已略過後處理）")

    print(f"Output dir     : {_out_dir}")
    print(f"  Activation/  : {_bi_act_dir}")
    print(f"  Weight/      : {_bi_wgt_dir}")
    print(f"Selected block : {selected_block}  (pro={pro.tolist() if pro is not None else None})")
    print(f"pred_boxes     : {pred_boxes.tolist()}")
    print(f"score_map range: [{float(score_map_ctr.min()):.4f}, {float(score_map_ctr.max()):.4f}]")


if __name__ == "__main__":
    main()

