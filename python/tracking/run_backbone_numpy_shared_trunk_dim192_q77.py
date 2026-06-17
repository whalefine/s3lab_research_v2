"""Pure-numpy SGLATrack backbone + shared-trunk head（dim192、Q7.7）。

使用方式：
    python tracking/run_backbone_numpy_shared_trunk_dim192_q77.py \\
      --golden-dir output/golden/vit_care_relu6_fixed_trunk_dim192_q77_golden \\
      --weight-dir output/exported_npy/vit_care_relu6_dim192_shared_trunk_q77_post_embed \\
      --output-dir output/golden/vit_care_relu6_numpy_trunk_dim192_q77_out

設計原則：
- 純 numpy，不建立 PyTorch model
- 輸入：``dump_golden_intermediate.py`` 產生的 post-embed golden（Q7.7 dump yaml）
- 權重：``export_checkpoint_npy.py`` 匯出的 float32 npy（同一 shared-trunk checkpoint）
- 定點：``to_fixed_point(7, 7)``，與 ``vit_CARE_relu6_dim192_fixed_q77_dump.py`` /
  ``head_shared_trunk_dump.py``（``FIXED_INT_BITS=7``）一致
- backbone attention 走 **浮點 CARE + 逐步 Q7.7**（非 dim32 的 Q8.8 整數 care_attention 路徑）
- head shared conv 中間節點 **不** 對 conv1/2 輸出做 Q7.7（對齊 dump 存檔語意）
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
# Model constants (vit_care_relu6_dim192 + shared-trunk head)
# ---------------------------------------------------------------------------
INT_BITS = 7
FRAC_BITS = 7
FIX_SCALE = 1 << FRAC_BITS

EMBED_DIM = 192
NUM_HEADS = 3
HEAD_DIM = EMBED_DIM // NUM_HEADS
SCALE = HEAD_DIM**-0.5
LENS_Z = 64
LENS_X = 256
N_TOKENS = LENS_Z + LENS_X
START_LAYER = 5
LN_EPS = 1e-6
FEAT_SZ = 16
FEAT_LEN = FEAT_SZ * FEAT_SZ
SEARCH_SIZE = 256
QK_MEAN_EPS_MIN = 1.0 / 128.0  # 對齊 AttentionDump clamp（Q7.7 最小正數）


# ---------------------------------------------------------------------------
# Numpy primitives
# ---------------------------------------------------------------------------

def to_fixed_point(x: np.ndarray, int_bits: int = INT_BITS, frac_bits: int = FRAC_BITS) -> np.ndarray:
    """有號定點數量化：scale → round → saturate → descale。"""
    scale = 2**frac_bits
    qmin = -(2 ** (int_bits + frac_bits - 1))
    qmax = (2 ** (int_bits + frac_bits - 1)) - 1
    scaled = x.astype(np.float64) * scale
    rounded = np.round(scaled)
    saturated = np.clip(rounded, qmin, qmax)
    return (saturated / scale).astype(np.float32)


def fp(x: np.ndarray) -> np.ndarray:
    return to_fixed_point(x, INT_BITS, FRAC_BITS)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = LN_EPS) -> np.ndarray:
    """浮點 LayerNorm（對齊 lib.module.LayerNorm）。"""
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=-1, keepdims=True)
    inv_std = (var + eps) ** (-0.5)
    y = centered * inv_std
    return (y * weight + bias).astype(np.float32)


def linear(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """浮點 linear（對齊 nn.Linear / lib.module.Linear）。"""
    w = weight.astype(np.float64)
    x64 = x.astype(np.float64)
    out = x64 @ w.T
    if bias is not None:
        out = out + bias.astype(np.float64)
    return out.astype(np.float32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(np.float32)


def relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 6.0).astype(np.float32)


def conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    padding: int = 1,
) -> np.ndarray:
    """浮點 conv2d（folded BN 權重：bias 已含在 folded_bias）。"""
    N, C_in, H, W = x.shape
    C_out, c_w, kH, kW = weight.shape
    if c_w != C_in:
        raise ValueError(f"conv2d Cin mismatch: x has {C_in}, weight has {c_w}")

    if padding:
        x = np.pad(
            x,
            [(0, 0), (0, 0), (padding, padding), (padding, padding)],
            mode="constant",
            constant_values=0.0,
        )
    H_out = x.shape[2] - kH + 1
    W_out = x.shape[3] - kW + 1
    if H_out < 1 or W_out < 1:
        raise ValueError(f"conv2d output size invalid: padded {x.shape[2:]}, kernel ({kH},{kW})")

    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    w = weight.astype(np.float64)
    x64 = x.astype(np.float64)
    for n in range(N):
        for oc in range(C_out):
            w_oc = w[oc]
            for oh in range(H_out):
                for ow in range(W_out):
                    patch = x64[n, :, oh : oh + kH, ow : ow + kW]
                    out[n, oc, oh, ow] = np.sum(patch * w_oc)
    if bias is not None:
        out = out + bias.astype(np.float32).reshape(1, C_out, 1, 1)
    return out.astype(np.float32)


def sigmoid_head(x: np.ndarray) -> np.ndarray:
    """對齊 head_shared_trunk_dump._sigmoid_q：sigmoid → clamp → Q7.7。"""
    s = 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    y = np.clip(s, 1e-4, 1.0 - 1e-4).astype(np.float32)
    return fp(y)


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


def write_bi(arr: np.ndarray, base: Path, int_bits: int = INT_BITS, frac_bits: int = FRAC_BITS) -> None:
    flat = arr.flatten()
    scale = 1 << frac_bits
    total_w = int_bits + frac_bits
    min_int = -(1 << (int_bits - 1)) * scale
    max_int = (1 << (int_bits - 1)) * scale - 1
    base_str = str(base)
    with open(base_str + "_bi.txt", "w") as f_bin:
        for num in flat:
            fixed = int(round(float(num) * scale))
            if fixed < min_int:
                fixed = min_int
            elif fixed > max_int:
                fixed = max_int
            twos = fixed & ((1 << total_w) - 1)
            f_bin.write(format(twos, f"0{total_w}b") + "\n")


def write_wbi(arr: np.ndarray, name: str, int_bits: int = INT_BITS, frac_bits: int = FRAC_BITS) -> None:
    if _bi_wgt_dir is not None:
        write_bi(arr, _bi_wgt_dir / name, int_bits, frac_bits)


def save_npy(filename: str, arr: np.ndarray) -> None:
    if _bi_act_dir is not None:
        stem = filename[:-4] if filename.endswith(".npy") else filename
        write_bi(arr, _bi_act_dir / stem, INT_BITS, FRAC_BITS)


def load_w(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# CARE attention（浮點路徑 + Q7.7，對齊 vit_CARE_relu6_dim192_fixed_q77_dump.AttentionDump）
# ---------------------------------------------------------------------------

def attention_forward(x: np.ndarray, block_idx: int, wp: Path) -> np.ndarray:
    B, N, C = x.shape
    if B != 1:
        raise ValueError(f"attention_forward expects B=1 (got {B})")
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

    qkv = linear(x, qkv_w, qkv_b)
    qkv = qkv.reshape(B, N, 3, H, d).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k, v = fp(q), fp(k), fp(v)
    save_npy(f"{pf}_attn_after_qkv_q.npy", q)
    save_npy(f"{pf}_attn_after_qkv_k.npy", k)
    save_npy(f"{pf}_attn_after_qkv_v.npy", v)

    s = SCALE**0.5
    qs = fp(q * s)
    ks = fp(k * s)
    q = fp(relu6(qs))
    k = fp(relu6(ks))
    v = fp(v)

    k_mean = fp(k.mean(axis=2, keepdims=True))
    qk_mean = fp(np.matmul(q, np.swapaxes(k_mean, -2, -1)))
    qk_mean_eps = fp(np.maximum(qk_mean, QK_MEAN_EPS_MIN))
    z = fp(1.0 / qk_mean_eps)

    kv = np.matmul(np.swapaxes(k, -2, -1), v) / float(N)
    kv = fp(kv)
    attn = np.matmul(q, kv) * z
    attn = fp(attn)
    attn = attn.transpose(0, 2, 1, 3).reshape(B, N, C)
    attn = fp(attn)

    out = fp(linear(attn, proj_w, proj_b))
    save_npy(f"{pf}_after_attn_attn_out.npy", out)
    return out


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
# Shared-trunk head
# ---------------------------------------------------------------------------

def head_shared_trunk(opt_feat: np.ndarray, wp: Path):
    """對齊 head_shared_trunk_dump.CenterPredictorSharedTrunkDump.get_score_map。"""
    fb = wp / "foldedBN"
    cp = wp / "convParam"

    w1 = load_w(fb / "box_head_shared_conv1_folded_weight.npy")
    b1 = load_w(fb / "box_head_shared_conv1_folded_bias.npy")
    write_wbi(w1, "box_head_shared_conv1_folded_weight")
    write_wbi(b1, "box_head_shared_conv1_folded_bias")
    x1 = relu(conv2d(opt_feat, w1, b1, padding=1))
    save_npy("box_head_shared_after_conv1_out.npy", x1)

    w2 = load_w(fb / "box_head_shared_conv2_folded_weight.npy")
    b2 = load_w(fb / "box_head_shared_conv2_folded_bias.npy")
    write_wbi(w2, "box_head_shared_conv2_folded_weight")
    write_wbi(b2, "box_head_shared_conv2_folded_bias")
    x2 = relu(conv2d(x1, w2, b2, padding=1))
    save_npy("box_head_shared_after_conv2_out.npy", x2)

    w_ctr = load_w(cp / "box_head_tail_ctr_weight.npy")
    b_ctr = load_w(cp / "box_head_tail_ctr_bias.npy")
    write_wbi(w_ctr, "box_head_tail_ctr_weight")
    write_wbi(b_ctr, "box_head_tail_ctr_bias")
    raw_ctr = conv2d(x2, w_ctr, b_ctr, padding=0)
    q_ctr = fp(raw_ctr)
    save_npy("box_head_tail_ctr_after_conv_out.npy", q_ctr)

    w_size = load_w(cp / "box_head_tail_size_weight.npy")
    b_size = load_w(cp / "box_head_tail_size_bias.npy")
    write_wbi(w_size, "box_head_tail_size_weight")
    write_wbi(b_size, "box_head_tail_size_bias")
    raw_size = conv2d(x2, w_size, b_size, padding=0)
    q_size = fp(raw_size)
    save_npy("box_head_tail_size_after_conv_out.npy", q_size)

    w_off = load_w(cp / "box_head_tail_offset_weight.npy")
    b_off = load_w(cp / "box_head_tail_offset_bias.npy")
    write_wbi(w_off, "box_head_tail_offset_weight")
    write_wbi(b_off, "box_head_tail_offset_bias")
    raw_off = conv2d(x2, w_off, b_off, padding=0)
    q_off = fp(raw_off)
    save_npy("box_head_tail_offset_after_conv_out.npy", q_off)

    score_map_ctr = sigmoid_head(q_ctr)
    score_map_size = sigmoid_head(q_size)
    score_map_offset = q_off
    save_npy("box_head_tail_ctr_after_sigmoid_out.npy", score_map_ctr)
    save_npy("box_head_tail_size_after_sigmoid_out.npy", score_map_size)
    save_npy("box_head_tail_offset_final_out.npy", score_map_offset)

    return score_map_ctr, score_map_size, score_map_offset


def cal_bbox(score_map_ctr: np.ndarray, size_map: np.ndarray, offset_map: np.ndarray) -> np.ndarray:
    """對齊 head_shared_trunk_dump.cal_bbox + Q7.7。"""
    flat = score_map_ctr.reshape(1, -1)
    idx = int(np.argmax(flat, axis=1)[0])
    idx_y, idx_x = idx // FEAT_SZ, idx % FEAT_SZ

    size = size_map.reshape(1, 2, -1)[:, :, idx]
    offset = offset_map.reshape(1, 2, -1)[:, :, idx]

    cx = (float(idx_x) + float(offset[0, 0])) / FEAT_SZ
    cy = (float(idx_y) + float(offset[0, 1])) / FEAT_SZ
    w = float(size[0, 0])
    h = float(size[0, 1])
    return fp(np.array([[cx, cy, w, h]], dtype=np.float32))


# ---------------------------------------------------------------------------
# Tracker post-processing
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
    p = argparse.ArgumentParser(
        description="Pure-numpy SGLATrack dim192 Q7.7 backbone + shared-trunk head."
    )
    p.add_argument(
        "--golden-dir",
        required=True,
        help="dump_golden_intermediate 輸出目錄（含 template/search_after_pos_add_out.npy）",
    )
    p.add_argument(
        "--weight-dir",
        default="output/exported_npy/vit_care_relu6_dim192_shared_trunk_q77_post_embed",
        help="export_checkpoint_npy 輸出根目錄",
    )
    p.add_argument("--output-dir", required=True, help="Activation/*_bi.txt 與 Weight/*_bi.txt 輸出目錄")
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

    z = fp(np.load(gd / "template_after_pos_add_out.npy").astype(np.float32))
    x = fp(np.load(gd / "search_after_pos_add_out.npy").astype(np.float32))
    save_npy("template_post_embed_input.npy", z)
    save_npy("search_post_embed_input.npy", x)

    merged = fp(np.concatenate([z, x], axis=1))
    save_npy("merged_tokens.npy", merged)
    x = fp(merged)
    save_npy("after_pos_drop_out.npy", x)

    for i in range(START_LAYER + 1):
        x = block_forward(x, i, wp)

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
        pro = 1.0 / (1.0 + np.exp(-linear(h_mlp, mlp2_w, mlp2_b).astype(np.float64)))
        pro = pro.astype(np.float32)
        selected_idx = int(np.argmax(pro, axis=1)[0])
        selected_block = selected_idx + START_LAYER + 1
        print(f"[adaptive] computed from numpy (no golden): selected_block={selected_block}")

    save_npy("adaptive_pro.npy", pro if pro is not None else np.zeros((1, 6), dtype=np.float32))
    save_npy("adaptive_sorted_topk_indices.npy", np.array([[selected_block]], dtype=np.int64))
    save_npy("adaptive_selected_layer_index.npy", np.array([selected_block], dtype=np.int64))

    x = block_forward(x, selected_block, wp)

    save_npy("backbone_after_recover_tokens_out.npy", fp(x))
    lap = wp / "layerParam"
    norm_w = load_w(lap / "backbone_norm_weight.npy")
    write_wbi(norm_w, "backbone_norm_weight")
    norm_b = load_w(lap / "backbone_norm_bias.npy")
    write_wbi(norm_b, "backbone_norm_bias")
    backbone_out = fp(layer_norm(x, norm_w, norm_b))
    save_npy("backbone_after_norm_backbone_out.npy", backbone_out)

    enc_opt = backbone_out[:, -FEAT_LEN:]
    opt = enc_opt[:, :, :, np.newaxis]
    opt = opt.transpose(0, 3, 2, 1)
    opt_feat = opt.reshape(-1, EMBED_DIM, FEAT_SZ, FEAT_SZ)

    save_npy("box_head_head_input.npy", opt_feat)
    score_map_ctr, size_map, offset_map = head_shared_trunk(opt_feat, wp)
    save_npy("box_head_after_forward_head_score_map.npy", score_map_ctr)
    save_npy("box_head_after_forward_head_size_map.npy", size_map)
    save_npy("box_head_after_forward_head_offset_map.npy", offset_map)

    bbox = cal_bbox(score_map_ctr, size_map, offset_map)
    save_npy("box_head_after_cal_bbox_bbox.npy", bbox)
    pred_boxes = fp(bbox.reshape(1, 1, 4))
    save_npy("box_head_after_forward_head_pred_boxes.npy", pred_boxes)

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
        init = manifest.get("init_bbox_xywh")
        if init is not None:
            print(
                f"[tracker] init_bbox (frame1 init, not frame2 target): "
                f"x1={init[0]:.4f}, y1={init[1]:.4f}, w={init[2]:.4f}, h={init[3]:.4f}"
            )
        print(f"[tracker] final bbox (frame2 px xywh): x1={x1:.4f}, y1={y1:.4f}, w={bw:.4f}, h={bh:.4f}")
    else:
        print("[WARNING] golden_manifest.json not found; skipping tracker post-processing.")

    print(f"Output dir     : {_out_dir}")
    print(f"  Activation/  : {_bi_act_dir}")
    print(f"  Weight/      : {_bi_wgt_dir}")
    print(f"Q format       : Q{INT_BITS}.{FRAC_BITS}")
    print(f"Selected block : {selected_block}  (pro={pro.tolist() if pro is not None else None})")
    print(f"pred_boxes     : {pred_boxes.tolist()}")
    print(f"score_map range: [{float(score_map_ctr.min()):.4f}, {float(score_map_ctr.max()):.4f}]")


if __name__ == "__main__":
    main()
