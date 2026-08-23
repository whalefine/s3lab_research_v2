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
- 定點：``to_fixed_point(7, 7)``（INT_BITS=7、FRAC_BITS=7，14-bit signed）
- **全整數 MAC 路徑**（對齊 ``run_backbone_numpy_shared_trunk.py`` 的 Q8.8 做法，改為 Q7.7）：
  linear / conv2d / layer_norm / attention 一律「先量化成整數 → 整數 MAC → 移位回 Q7.7」，
  權重以 ``round(w * 128)`` 量化（等同 ROM 的 Q7.7 權重），可與 RTL 整數 MAC 對拍。
- **捨入契約**：所有 ``>> FRAC_BITS`` 的回 Q7.7（linear / conv / layer_norm / attention 各步）與
  ``cal_bbox`` 的 ``>> 4`` 皆採 **round-to-nearest**（``(acc + 2^(shift-1)) >> shift``）；
  僅 reciprocal 的 ``_trunc_q_slice`` 維持截斷（對齊 recip_nr.v 語意）。RTL 對應移位須一致。
- inv_sqrt / reciprocal / sigmoid 改為 LUT 種子 + Newton 迭代 / LUT 內插的整數版本。
  注意：若 RTL 使用對應 LUT ROM，其內容需與本腳本產生的表一致（見 §20 ROM 再產生）。
- head shared conv1/2 輸出比照 dim32 加 ``fp()`` 截斷成 Q7.7。
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


# ---------------------------------------------------------------------------
# Q7.7 fixed-point integer helpers（對齊 run_backbone_numpy_shared_trunk.py 的 Q8.8 做法）
#   - 數值以「整數 code = round(value * 128)」表示
#   - MAC：x_int @ w_int 後 >> FRAC_BITS 回到 Q7.7
#   - 飽和到 14-bit signed [Q_MIN, Q_MAX]
# ---------------------------------------------------------------------------
SCALE_FP = 1 << FRAC_BITS  # 128
Q_MIN = -(1 << (INT_BITS + FRAC_BITS - 1))  # -8192
Q_MAX = (1 << (INT_BITS + FRAC_BITS - 1)) - 1  # 8191
RND_FP = 1 << (FRAC_BITS - 1)  # 64（>> FRAC_BITS 的四捨五入常數）

RELU6_MAX_Q = 6 * SCALE_FP  # 768
S_Q = int(round(SCALE_FP * (HEAD_DIM ** -0.25)))  # attention scale（Q7.7），HEAD_DIM=64 -> 45
RCP_N_NUM = int(round((1 << 16) / N_TOKENS))  # round(65536/320)=205
RCP_LN = int(round((1 << 16) / EMBED_DIM))    # round(65536/192)=341
LN_MEAN_SHIFT = 16
LN_VAR_SHIFT = 16 + FRAC_BITS  # 23


def sat_q(v: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.int64), Q_MIN, Q_MAX).astype(np.int64)


def as_q(x: np.ndarray) -> np.ndarray:
    """float → Q7.7 整數 code。"""
    return np.round(np.asarray(x, dtype=np.float64) * SCALE_FP).astype(np.int64)


def from_q(x_q: np.ndarray) -> np.ndarray:
    """Q7.7 整數 code → float32（語意等同 fp()）。"""
    return (np.asarray(x_q, dtype=np.float64) / SCALE_FP).astype(np.float32)


def w_q(w: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(w, dtype=np.float64) * SCALE_FP).astype(np.int64)


def rnd_shr_fp(v: np.ndarray) -> np.ndarray:
    """(v + 2^(FRAC-1)) >> FRAC，再飽和到 Q7.7。"""
    return sat_q((np.asarray(v, dtype=np.int64) + RND_FP) >> FRAC_BITS)


def relu6_q(x_q: np.ndarray) -> np.ndarray:
    x = np.asarray(x_q, dtype=np.int64)
    x = np.where(x < 0, 0, x)
    return np.where(x > RELU6_MAX_Q, RELU6_MAX_Q, x).astype(np.int64)


def _msb_index(x: np.ndarray) -> np.ndarray:
    """leading-bit index（x>=1），對齊 RTL LUT 種子的 MSB 編碼。"""
    x = np.maximum(np.asarray(x, dtype=np.int64), 1)
    k = np.zeros_like(x)
    for bit in range(15, -1, -1):
        hit = (x >> bit) & 1
        k = np.where((k == 0) & (hit == 1), np.int64(bit), k)
    return k


# --- inv_sqrt: LUT 種子 + Newton 迭代（Q7.7；對齊 inv_sqrt_lut_seed.v / inv_sqrt_nr.v 結構）---
_INV_SQRT_SEED_Q = np.array(
    [min(Q_MAX, max(1, int(round(SCALE_FP / math.sqrt(3 * (1 << k) / SCALE_FP))))) for k in range(16)],
    dtype=np.int64,
)
_INV_SQRT_NR_ITERS = 3


def _inv_sqrt_nr_q(var_q: np.ndarray) -> np.ndarray:
    """Q7.7 inv-sqrt（LUT 種子 + 3 次 NR，每步四捨五入）。回傳 inv_std 的 Q7.7 code。"""
    v = np.asarray(var_q, dtype=np.int64)
    v_eps = np.where(v <= 0, np.int64(1), v)
    y = _INV_SQRT_SEED_Q[_msb_index(v_eps)].astype(np.int64)
    for _ in range(_INV_SQRT_NR_ITERS):
        y_sq = (y * y + RND_FP) >> FRAC_BITS          # y^2 (Q7.7)
        term = (v_eps * y_sq + (1 << (FRAC_BITS + 1))) >> (FRAC_BITS + 1)  # 0.5*var*y^2 (Q7.7)
        coeff = np.int64(3 * SCALE_FP // 2) - term     # 1.5 - 0.5*var*y^2（1.5 -> 192）
        y = sat_q((y * coeff + RND_FP) >> FRAC_BITS)
    return y


# --- reciprocal: LUT 種子 + Newton 迭代（Q7.7；對齊 recip_lut_seed.v / recip_nr.v 結構）---
_RECIP_SEED_Q = np.array(
    [min(Q_MAX, max(1, int(round((1 << (2 * FRAC_BITS)) / (1.5 * (1 << k)))))) for k in range(16)],
    dtype=np.int64,
)
_RECIP_NR_ITERS = 1
_Q_WIDTH = INT_BITS + FRAC_BITS  # 14
_Q_HALF = 1 << (_Q_WIDTH - 1)
_Q_FULL = 1 << _Q_WIDTH


def _trunc_q_slice(v: np.ndarray) -> np.ndarray:
    """取 (v >> FRAC) 的 14-bit signed 切片（對齊 recip_nr.v 的截位語意）。"""
    v = np.asarray(v, dtype=np.int64)
    s = (v >> FRAC_BITS) & (_Q_FULL - 1)
    return np.where(s >= _Q_HALF, s - _Q_FULL, s).astype(np.int64)


def _recip_nr_q(x_q: np.ndarray) -> np.ndarray:
    """Q7.7 reciprocal（LUT 種子 + 1 次 NR）。回傳 1/x 的 Q7.7 code。"""
    x = np.maximum(np.asarray(x_q, dtype=np.int64), 1)
    y = _RECIP_SEED_Q[_msb_index(x)].astype(np.int64)
    for _ in range(_RECIP_NR_ITERS):
        coeff = np.int64(2 * SCALE_FP) - _trunc_q_slice(x * y)  # 2.0 - x*y（2.0 -> 256）
        y = np.clip(_trunc_q_slice(y * coeff), Q_MIN, Q_MAX).astype(np.int64)
    return y


# --- sigmoid: LUT + 線性內插（Q7.7；對齊 RTL sigmoid LUT 結構）---
_SIG_LUT_N = 64
_SIG_LUT_INT = np.round(
    (1.0 / (1.0 + np.exp(-np.linspace(-8.0, 8.0, _SIG_LUT_N + 1).astype(np.float64)))) * SCALE_FP
).astype(np.int64)
_SIG_BIN_SHIFT = FRAC_BITS - 2  # 5（16 的輸入範圍切 64 格 → 每格 0.25 = 32 = 2^5）


def sigmoid_q(x: np.ndarray) -> np.ndarray:
    """LUT + 線性內插 sigmoid（Q7.7）。輸入 float，回傳 float32（Q7.7 量化）。"""
    x_int = np.round(np.clip(x.astype(np.float64), -8.0, 8.0) * SCALE_FP).astype(np.int64)
    shifted = x_int + (8 * SCALE_FP)
    idx = np.clip(shifted >> _SIG_BIN_SHIFT, 0, _SIG_LUT_N - 1).astype(np.int64)
    frac = (shifted & ((1 << _SIG_BIN_SHIFT) - 1)).astype(np.int64)
    lo = _SIG_LUT_INT[idx]
    hi = _SIG_LUT_INT[idx + 1]
    delta = hi - lo
    result = (lo * (1 << _SIG_BIN_SHIFT) + delta * frac) >> _SIG_BIN_SHIFT
    return (result.astype(np.float64) / SCALE_FP).astype(np.float32)


def linear_sat_q(x_q: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]) -> np.ndarray:
    """Q7.7 linear MAC + bias + 飽和（整數 code 進、整數 code 出）。"""
    x_int = np.asarray(x_q, dtype=np.int64)
    w_int = w_q(weight)
    *batch, in_dim = x_int.shape
    acc = (x_int.reshape(-1, in_dim) @ w_int.T + RND_FP) >> FRAC_BITS
    if bias is not None:
        acc = acc + w_q(bias)
    return sat_q(acc).reshape(*batch, weight.shape[0]).astype(np.int64)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = LN_EPS) -> np.ndarray:
    """Q7.7 整數 LayerNorm（mean/var 整數累加 + LUT inv_sqrt）。回傳 float32（Q7.7）。"""
    N = x.shape[-1]
    rcp = int(round((1 << LN_MEAN_SHIFT) / N))
    x_int = as_q(x)
    w_int = w_q(weight)
    b_int = w_q(bias)

    sum_int = x_int.sum(axis=-1, keepdims=True)
    mean_int = sat_q((sum_int * rcp + (1 << (LN_MEAN_SHIFT - 1))) >> LN_MEAN_SHIFT)
    centered = sat_q(x_int - mean_int)
    sum_sq = (centered * centered).sum(axis=-1, keepdims=True)
    var_int = np.clip(
        (sum_sq * rcp + (1 << (LN_VAR_SHIFT - 1))) >> LN_VAR_SHIFT, Q_MIN, Q_MAX
    ).astype(np.int64)
    inv_std = _inv_sqrt_nr_q(var_int)

    ci_std = rnd_shr_fp(centered * inv_std)
    wci = rnd_shr_fp(w_int * ci_std)
    y_int = sat_q(wci + b_int)
    return from_q(y_int)


def linear(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """Q7.7 整數 MAC linear（float 進、float 回；飽和交給呼叫端 fp()）。"""
    _S = SCALE_FP
    x_int = np.round(x.astype(np.float64) * _S).astype(np.int64)
    w_int = np.round(weight.astype(np.float64) * _S).astype(np.int64)
    *batch, in_dim = x_int.shape
    acc = (x_int.reshape(-1, in_dim) @ w_int.T + RND_FP) >> FRAC_BITS
    if bias is not None:
        acc = acc + np.round(bias.astype(np.float64) * _S).astype(np.int64)
    return (acc.reshape(*batch, weight.shape[0]).astype(np.float64) / _S).astype(np.float32)


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
    """Q7.7 整數 MAC conv2d（folded BN 權重：bias 已含在 folded_bias）。float 進、float 回。"""
    _S = SCALE_FP
    N, C_in, H, W = x.shape
    C_out, c_w, kH, kW = weight.shape
    if c_w != C_in:
        raise ValueError(f"conv2d Cin mismatch: x has {C_in}, weight has {c_w}")

    x_int = np.round(x.astype(np.float64) * _S).astype(np.int64)
    w_int = np.round(weight.astype(np.float64) * _S).astype(np.int64)

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
            w_oc = w_int[oc]
            for oh in range(H_out):
                for ow in range(W_out):
                    patch = x_int[n, :, oh : oh + kH, ow : ow + kW]
                    acc[n, oc, oh, ow] = int(np.sum(patch * w_oc))

    acc_q = (acc + RND_FP) >> FRAC_BITS
    if bias is not None:
        bias_int = np.round(bias.astype(np.float64) * _S).astype(np.int64)
        acc_q = acc_q + bias_int[np.newaxis, :, np.newaxis, np.newaxis]
    return (acc_q.astype(np.float64) / _S).astype(np.float32)


def sigmoid_head(x: np.ndarray) -> np.ndarray:
    """head 的 _sigmoid：LUT sigmoid → Q7.7 → Q7.7 邊界 clamp（對齊 dim32 sigmoid_clamped）。"""
    _LB = 1.0 / SCALE_FP
    _UB = (SCALE_FP - 1.0) / SCALE_FP
    return np.clip(fp(sigmoid_q(x)), _LB, _UB).astype(np.float32)


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
        if np.issubdtype(np.asarray(arr).dtype, np.integer):
            write_bi(from_q(arr), _bi_act_dir / stem, INT_BITS, FRAC_BITS)
        else:
            write_bi(arr, _bi_act_dir / stem, INT_BITS, FRAC_BITS)


def load_w(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# CARE attention（Q7.7 整數路徑，對齊 run_backbone_numpy_shared_trunk.py 的 Q8.8 care_attention）
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

    # qkv linear（整數 MAC）→ split q/k/v（整數 code）
    x_q = as_q(x)
    qkv_q = linear_sat_q(x_q, qkv_w, qkv_b)
    qkv_q = qkv_q.reshape(B, N, 3, H, d).transpose(2, 0, 3, 1, 4)
    q = qkv_q[0].astype(np.int64)
    k = qkv_q[1].astype(np.int64)
    v = qkv_q[2].astype(np.int64)
    save_npy(f"{pf}_attn_after_qkv_q.npy", q)
    save_npy(f"{pf}_attn_after_qkv_k.npy", k)
    save_npy(f"{pf}_attn_after_qkv_v.npy", v)

    # S_SPLIT：q/k 乘上 scale 後 relu6（整數）
    q = relu6_q(rnd_shr_fp(q * S_Q))
    k = relu6_q(rnd_shr_fp(k * S_Q))

    # S_K_MEAN：km[h,d] = mean_n(k)（* RCP_N 近似除 N）
    k_sum = k.sum(axis=2)  # (B,H,d)
    km = sat_q((k_sum * RCP_N_NUM + (1 << 15)) >> 16)

    # S_QK_MEAN：qkm[h,n] = sum_d q*km，回 Q7.7
    qkm = rnd_shr_fp(np.einsum("zhnd,zhd->zhn", q, km))  # (B,H,N)
    qkm_eps = np.maximum(qkm, 1).astype(np.int64)
    zr = _recip_nr_q(qkm_eps)  # (B,H,N)

    # S_KV：kv[h,i,j] = mean_n(k[..,i]*v[..,j])（* RCP_N 近似除 N，再 >>FRAC 回 Q7.7）
    kv_acc = np.einsum("zhni,zhnj->zhij", k, v)  # (B,H,d,d)
    kv = sat_q(
        (kv_acc * RCP_N_NUM + (1 << (16 + FRAC_BITS - 1))) >> (16 + FRAC_BITS)
    )

    # S_ATTN：ao[h,n,j] = rnd_shr(sat(sum_i q*kv) * zr)
    dot = rnd_shr_fp(np.einsum("zhni,zhij->zhnj", q, kv))  # (B,H,N,d)
    ao = rnd_shr_fp(dot * zr[..., np.newaxis])  # (B,H,N,d)
    ao = ao.transpose(0, 2, 1, 3).reshape(B, N, C).astype(np.int64)

    # proj linear（整數 MAC）
    attn_q = linear_sat_q(ao, proj_w, proj_b)
    save_npy(f"{pf}_after_attn_attn_out.npy", attn_q)
    return from_q(attn_q)


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
    x1 = fp(relu(conv2d(opt_feat, w1, b1, padding=1)))
    save_npy("box_head_shared_after_conv1_out.npy", x1)

    w2 = load_w(fb / "box_head_shared_conv2_folded_weight.npy")
    b2 = load_w(fb / "box_head_shared_conv2_folded_bias.npy")
    write_wbi(w2, "box_head_shared_conv2_folded_weight")
    write_wbi(b2, "box_head_shared_conv2_folded_bias")
    x2 = fp(relu(conv2d(x1, w2, b2, padding=1)))
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

    # /FEAT_SZ（=16）以整數移位 >>4 完成（Q7.7 整數域，對齊 dim32 cal_bbox）
    feat_shift = int(round(math.log2(FEAT_SZ)))
    offset_x_q = int(round(float(offset[0, 0]) * SCALE_FP))
    offset_y_q = int(round(float(offset[0, 1]) * SCALE_FP))
    _half = 1 << (feat_shift - 1)
    cx = ((int(idx_x) * SCALE_FP + offset_x_q + _half) >> feat_shift) / SCALE_FP
    cy = ((int(idx_y) * SCALE_FP + offset_y_q + _half) >> feat_shift) / SCALE_FP
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
