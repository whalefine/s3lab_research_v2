"""Pure-numpy SGLATrack backbone + head forward pass from pre-computed post-embed .npy.

使用方式：
    python tracking/run_backbone_numpy.py \\
      --golden-dir output/golden/vit_care_relu6_fixed_golden \\
      --weight-dir output/exported_npy/vit_coco_uav123_care_relu6_ep0050_all \\
      --output-dir output/golden/vit_care_relu6_numpy_out

設計原則：
- 純 numpy，不建立任何 PyTorch model 或呼叫 .forward()
- template_after_pos_add_out.npy / search_after_pos_add_out.npy 以 np.load 從
  --golden-dir 載入，作為 backbone 的真正輸入
- 所有 weight 從 --weight-dir 的 exported npy 載入（float32，不做 Q8.8）
- 每個 op 後的 activation 做 to_fixed_point(8, 8)（使用 np.trunc 向零截斷量化，
  與 PyTorch lib.module 的 round 版不同；偏硬體／參考 rongxuan 的 trunc 語意）
- adaptive selector MLP 屬純軟體路徑，保持 float32 不量化
- 輸出 npy 命名與 dump_golden_intermediate.py 完全一致，可直接與 RTL 輸出比對
- write_bi() 對每個 weight/activation 輸出 .txt（十進位）和 _bi.txt（二補數 binary），
  參考 reference/rongxuan/06_GetBinary/python_check_verilog.py 的插入邏輯
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
# Model constants (vit_coco_uav123_care_relu6_fixed)
# ---------------------------------------------------------------------------
EMBED_DIM = 32
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS        # 64
SCALE = HEAD_DIM ** -0.5                 # 0.125
S = SCALE ** 0.5                         # ≈ 0.353553
LENS_Z = 64
LENS_X = 256
N_TOKENS = LENS_Z + LENS_X              # 320
START_LAYER = 5
LN_EPS = 1e-6
FEAT_SZ = 16                             # search_size // stride = 256 // 16
FEAT_LEN = FEAT_SZ * FEAT_SZ            # 256
SEARCH_SIZE = 256


# ---------------------------------------------------------------------------
# Numpy primitives
# ---------------------------------------------------------------------------

def to_fixed_point(x: np.ndarray, int_bits: int, frac_bits: int) -> np.ndarray:
    """有號定點數量化：scale → trunc（向零截斷）→ saturate → descale。

    使用 np.trunc 對 scaled 值向零取整，再 clip 到 Q 格式整數格點範圍。
    與 lib.module.fixed_point 若採用 round 的實作不同；若要對拍 PyTorch golden 請改回 round。
    """
    scale = 2 ** frac_bits
    qmin = -(2 ** (int_bits + frac_bits - 1))
    qmax = (2 ** (int_bits + frac_bits - 1)) - 1
    scaled = x.astype(np.float64) * scale
    # truncated = np.trunc(scaled)
    truncated = np.trunc(scaled)
    saturated = np.clip(truncated, qmin, qmax)
    return (saturated / scale).astype(np.float32)  # RTL: 整數保留，/scale 僅 numpy 還原 float


def fp(x: np.ndarray) -> np.ndarray:
    """Shorthand: to_fixed_point(x, 8, 8)."""
    return to_fixed_point(x, 8, 8)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = LN_EPS, inv_sqrt_iter: int = 2) -> np.ndarray:
    """硬體友善 LayerNorm（方案 A：Newton-Raphson inv_sqrt）。

    計算流程（對齊 RTL）：
      rcp_n    = round(1/N × 2^16)/2^16   常數乘法 + 右移 16（N=768 → 85/65536）
      mean     = sum(x) × rcp_n           加法樹 + 常數乘
      centered = x - mean                 減法
      var      = sum(centered^2) × rcp_n  平方後累加（float64 避免溢位）再 Q8.8
      inv_std  = _inv_sqrt_nr(var+eps)    LUT 初始值 + inv_sqrt_iter 次 NR 迭代
      output   = weight × (centered × inv_std) + bias
    inv_sqrt_iter 應與 RTL NR 迭代次數一致才能 bit-accurate 對齊。
    """
    N = x.shape[-1]
    rcp_n = float(round(1.0 / N * 65536)) / 65536.0     # ≈ 1/N，RTL 常數乘法

    mean     = fp(x.sum(axis=-1, keepdims=True) * rcp_n)
    centered = fp(x - mean)
    # centered^2 在 float64 累加（最大 128^2×768 ≈ 12M，超出 Q8.8），乘 rcp_n 後再 Q8.8
    var      = fp((centered.astype(np.float64) ** 2).sum(axis=-1, keepdims=True) * rcp_n)
    inv_std  = fp(_inv_sqrt_nr(var + eps, num_iter=inv_sqrt_iter))
    return fp(weight * fp(centered * inv_std) + bias)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Q8.8 fixed-point integer MAC linear（硬體語意：整數 MAC + int64 累加器）。

    計算流程：
      x_int   = trunc(x_float * 2^8)     Q8.8 → int32（trunc 對齊 to_fixed_point）
      w_int   = trunc(w_float * 2^8)     Q8.8 → int32
      acc     = (int64 x_flat) @ (int64 W)^T   每輸出維一條 MAC 樹，無 float64 GEMM
      acc_q88 = acc >> 8                 算術右移 8（Q16.16 → Q8.8）
      acc_q88 += trunc(bias * 2^8)       加 Q8.8 integer bias
      output  = acc_q88 / 2^8             還原 Q8.8 float32

    注意：累加在 int64 上完成；若 int32 全幅隨機且 in_dim 極大，理論上可能溢位。
    本模型 Q8.8 與維度下與原 float64 matmul 結果一致（實務上 acc < 2^63）。
    """
    _SCALE = 1 << 8  # 256

    x_int = np.trunc(x.astype(np.float64) * _SCALE).astype(np.int32)
    w_int = np.trunc(weight.astype(np.float64) * _SCALE).astype(np.int32)

    *batch, in_dim = x_int.shape
    out_dim = w_int.shape[0]

    x64 = x_int.reshape(-1, in_dim).astype(np.int64)
    w64 = w_int.astype(np.int64)
    acc_q16 = x64 @ w64.T                                 # [B*N, out_dim], int64
    acc_q88 = acc_q16 >> 8

    if bias is not None:
        bias_int = np.trunc(bias.astype(np.float64) * _SCALE).astype(np.int64)
        acc_q88 = acc_q88 + bias_int

    return (acc_q88.reshape(*batch, out_dim).astype(np.float64) / _SCALE).astype(np.float32)  # RTL: acc_q88 整數保留，/256 僅 numpy 還原


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(np.float32)


def relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 6.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Sigmoid LUT（模組載入時建立一次）
# RTL 對應：65-entry ROM，覆蓋 x ∈ [-8, 8]，步長 0.25
# W_LUT 由輸出精度決定；本實作存 float64，量化由 sigmoid_clamped 負責。
# ---------------------------------------------------------------------------
_SIGMOID_LUT_N = 64   # 區間數；LUT 有 65 個端點 (index 0..64)
_SIGMOID_LUT = 1.0 / (1.0 + np.exp(
    -np.linspace(-8.0, 8.0, _SIGMOID_LUT_N + 1).astype(np.float64)
))  # shape (65,), float64 精確值
_SIGMOID_LUT_INT = np.round(_SIGMOID_LUT * 256).astype(np.int32)  # Q0.8 整數 [0,255]，對齊 sigmoid_lut.v 的 ROM


def sigmoid(x: np.ndarray) -> np.ndarray:
    """LUT + 線性插值 sigmoid，對齊 RTL 硬體語意。

    硬體計算流程（整數域）：
      x_int   = round(x × 256)                 Q8.8 → 16-bit signed integer
      clamped = clamp(x_int, −2048, 2048)       x 飽和到 [−8, +8]
      shifted = clamped + 2048                  平移：−8→0，+8→4096；範圍 [0, 4096]
      idx     = shifted >> 6                    LUT index [0, 63]（步長 64 = 0.25×256）
      frac6   = shifted & 0x3F                  6-bit 小數 [0, 63]
      output  = lut[idx] + (lut[idx+1]−lut[idx]) × frac6/64

    中間為 float64（RTL 可用定點乘法器 + 加法器取代）；
    輸出 float32，尚未量化—由呼叫端 sigmoid_clamped 做 Q8.8。
    """
    # 原始精確實作（已停用，硬體不支援 exp）：
    # return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    # 原始分段線性近似（已停用，誤差過大約 0.07）：
    # x = x.astype(np.float32); y = 0.5 + (x / 8.0)
    # y = np.where(x <= -4.0, 0.0, y); y = np.where(x >= 4.0, 1.0, y)

    x_int = np.round(
        np.clip(x.astype(np.float64), -8.0, 8.0) * 256.0
    ).astype(np.int32)                                            # Q8.8 整數 [-2048, 2048]
    shifted = x_int + 2048                                        # → [0, 4096]
    idx   = np.clip(shifted >> 6, 0, _SIGMOID_LUT_N - 1).astype(np.int32)  # [0, 63]
    frac6 = (shifted & 0x3F).astype(np.int32)                    # [0, 63]

    # RTL 整數插值（對齊 sigmoid_lut.v Step 5）：
    #   sum = lo×64 + delta×frac6  (14-bit max=16320 < 2^14)
    #   result = sum >> 6           (>> 6，不是 /64)
    lo_int = _SIGMOID_LUT_INT[idx]           # Q0.8 整數 [0,255]
    hi_int = _SIGMOID_LUT_INT[idx + 1]       # idx ≤ 63 → idx+1 ≤ 64，永遠合法
    delta  = (hi_int - lo_int).astype(np.int32)
    result = (lo_int * 64 + delta * frac6) >> 6   # >> 6，對應 Verilog sum[13:6]
    return (result.astype(np.float64) / 256.0).astype(np.float32)  # RTL: 整數保留，/256 僅 numpy 還原


def sigmoid_clamped(x: np.ndarray) -> np.ndarray:
    """head 的 _sigmoid：LUT sigmoid → Q8.8 → Q8.8 邊界 clamp。

    對齊 head_dump.py 的 torch.clamp(t.sigmoid_(), min=1e-4, max=1-1e-4)：
      1e-4  在 Q8.8 round 成 0（消失）→ 改用 1/256 = 0.00390625（Q8.8 最小正值）
      0.9999 → 改用 255/256 = 0.99609375（Q8.8 最大值 < 1）
    sigmoid(x) 輸出 float32（非 Q8.8），fp() 量化到 Q8.8 格點，再 clip 邊界。
    """
    _LB = 1.0 / 256    # Q8.8 最小正值
    _UB = 255.0 / 256  # Q8.8 最大值 < 1
    return np.clip(fp(sigmoid(x)), _LB, _UB).astype(np.float32)


def conv2d(x: np.ndarray, weight: np.ndarray,
           bias: Optional[np.ndarray] = None, padding: int = 1) -> np.ndarray:
    """Q8.8 fixed-point integer MAC conv2d，對齊 RTL 行為。

    計算流程（全部在整數域）：
      x_int   = round(x_float * 2^8)         Q8.8 → int32
      w_int   = round(w_float * 2^8)         Q8.8 → int32
      acc     = Σ x_int * w_int              Q16.16 int64 accumulator
      acc_q88 = acc >> 8                     算術右移 8 bit（truncation，對齊 RTL）
      acc_q88 += round(bias * 2^8)           加 Q8.8 integer bias
      output  = acc_q88 / 2^8               還原為 Q8.8 float32

    x: [N, Cin, H, W]  weight: [Cout, Cin, kH, kW]  bias: [Cout]
    """
    _SCALE = 1 << 8  # 256

    N, C_in, H, W = x.shape
    C_out, c_w, kH, kW = weight.shape
    if c_w != C_in:
        raise ValueError(f"conv2d Cin mismatch: x has {C_in}, weight has {c_w}")

    # Q8.8 float → int32（乘積 int32×int32 最大 ~10^9，累加用 int64）
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
        raise ValueError(
            f"conv2d output size invalid: padded {x_int.shape[2:]}, kernel ({kH},{kW})"
        )

    # int64 累加器（Q16.16）
    acc = np.zeros((N, C_out, H_out, W_out), dtype=np.int64)
    for n in range(N):
        for oc in range(C_out):
            w_oc = w_int[oc].astype(np.int64)          # [Cin, kH, kW]
            for oh in range(H_out):
                for ow in range(W_out):
                    patch = x_int[n, :, oh:oh + kH, ow:ow + kW].astype(np.int64)
                    acc[n, oc, oh, ow] = np.sum(patch * w_oc)

    # Q16.16 → Q8.8：算術右移 8 bit（Python >> 對負數為算術移位，與 RTL 一致）
    acc_q88 = acc >> 8

    if bias is not None:
        bias_int = np.round(bias.astype(np.float64) * _SCALE).astype(np.int64)
        acc_q88 += bias_int[np.newaxis, :, np.newaxis, np.newaxis]

    return (acc_q88.astype(np.float64) / _SCALE).astype(np.float32)  # RTL: acc_q88 整數保留，/256 僅 numpy 還原


def hann1d(sz: int, centered: bool = True) -> np.ndarray:
    """對齊 lib.test.utils.hann.hann1d。"""
    if centered:
        return (0.5 * (1 - np.cos(
            (2 * math.pi / (sz + 1)) * np.arange(1, sz + 1)
        ))).astype(np.float32)
    w = 0.5 * (1 + np.cos((2 * math.pi / (sz + 2)) * np.arange(0, sz // 2 + 1)))
    return np.concatenate([w, w[1: sz - sz // 2][::-1]]).astype(np.float32)


def hann2d(sz_h: int, sz_w: int, centered: bool = True) -> np.ndarray:
    """對齊 lib.test.utils.hann.hann2d，回傳 [1, 1, sz_h, sz_w]。"""
    h = hann1d(sz_h, centered).reshape(-1, 1)
    w = hann1d(sz_w, centered).reshape(1, -1)
    return (h * w).reshape(1, 1, sz_h, sz_w)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_out_dir: Optional[Path] = None
_bi_act_dir: Optional[Path] = None   # _out_dir / "Activation"
_bi_wgt_dir: Optional[Path] = None   # _out_dir / "Weight"


def write_bi(arr: np.ndarray, base: Path, int_bits: int = 8, frac_bits: int = 8) -> None:
    """寫出十進位 .txt 和二補數 binary _bi.txt。
    參考 reference/rongxuan/06_GetBinary/python_check_verilog.py write_bi()。
    """
    flat = arr.flatten()
    scale = 1 << frac_bits
    total_w = int_bits + frac_bits
    min_int = -(1 << (int_bits - 1)) * scale
    max_int = (1 << (int_bits - 1)) * scale - 1
    base_str = str(base)
    with open(base_str + '.txt', 'w') as f_dec, \
         open(base_str + '_bi.txt', 'w') as f_bin:
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
    """Weight 寫出到 _out_dir/Weight/{name}.txt 和 _bi.txt。"""
    if _bi_wgt_dir is not None:
        write_bi(arr, _bi_wgt_dir / name, int_bits, frac_bits)


def save_npy(filename: str, arr: np.ndarray) -> None:
    """儲存 activation npy，同時輸出 write_bi 到 _out_dir/Activation/。"""
    if _out_dir is not None:
        np.save(_out_dir / filename, arr)
    if _bi_act_dir is not None:
        stem = filename[:-4] if filename.endswith('.npy') else filename
        write_bi(arr, _bi_act_dir / stem)


def load_w(path: Path) -> np.ndarray:
    """載入 weight，保持 float32（weight 不做 Q8.8，對齊 PyTorch forward 行為）。"""
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Attention (CARE ReLU6 fixed-point)
# 對齊 vit_CARE_relu6_fixed_dump.AttentionDump.forward
# ---------------------------------------------------------------------------

def _inv_sqrt_nr(v: np.ndarray, num_iter: int = 2) -> np.ndarray:
    """硬體友善 inverse square root：1/sqrt(v) via Newton-Raphson
    y_{n+1} = y_n * (1.5 - 0.5 * v * y_n^2)。

    對應 RTL：LUT 給初始值 → num_iter 次 NR 迭代。
    num_iter=1 ≈ 15-bit 精度；num_iter=2 ≈ 30-bit，對 Q8.8 足夠。
    """
    v64 = v.astype(np.float64)
    y = 1.0 / np.sqrt(v64 + 1e-30)      # 初始值（對應 RTL LUT 輸出）
    for _ in range(num_iter):
        y = y * (1.5 - 0.5 * v64 * y * y)
    return y.astype(np.float32)


def _recip_nr(x: np.ndarray, num_iter: int = 1) -> np.ndarray:
    """硬體友善 reciprocal：1/x via Newton-Raphson  y_{n+1} = y_n*(2 - x*y_n)。

    對應 RTL：6-bit LUT 給初始值 → num_iter 次 NR 迭代達 Q8.8 精度。
    num_iter=0 退化為純 float reciprocal（等同原始行為）。
    """
    x64 = x.astype(np.float64)
    y = 1.0 / x64                        # 初始值（對應 RTL LUT 輸出）
    for _ in range(num_iter):
        y = y * (2.0 - x64 * y)          # NR：一次迭代精度翻倍
    return y.astype(np.float32)


def attention_forward(x: np.ndarray, block_idx: int, wp: Path) -> np.ndarray:
    B, N, C = x.shape
    H, d = NUM_HEADS, HEAD_DIM
    pf = f"backbone_blocks_{block_idx}"
    lp = wp / "linearParam"

    qkv_w  = load_w(lp / f"{pf}_attn_qkv_weight.npy")
    write_wbi(qkv_w,  f"{pf}_attn_qkv_weight")
    qkv_b  = load_w(lp / f"{pf}_attn_qkv_bias.npy")
    write_wbi(qkv_b,  f"{pf}_attn_qkv_bias")
    proj_w = load_w(lp / f"{pf}_attn_proj_weight.npy")
    write_wbi(proj_w, f"{pf}_attn_proj_weight")
    proj_b = load_w(lp / f"{pf}_attn_proj_bias.npy")
    write_wbi(proj_b, f"{pf}_attn_proj_bias")

    # qkv linear → reshape → split → Q8.8
    qkv = linear(x, qkv_w, qkv_b).reshape(B, N, 3, H, d).transpose(2, 0, 3, 1, 4)
    q, k, v = fp(qkv[0]), fp(qkv[1]), fp(qkv[2])   # each [B, H, N, d]
    save_npy(f"{pf}_attn_after_qkv_q.npy", q)
    save_npy(f"{pf}_attn_after_qkv_k.npy", k)
    save_npy(f"{pf}_attn_after_qkv_v.npy", v)

    # scale → ReLU6 → Q8.8（attention 用 ReLU6，不是 ReLU）
    q = fp(relu6(fp(q * S)))
    k = fp(relu6(fp(k * S)))
    v = fp(v)                                         # attn_drop = identity in eval

    # CARE attention：k_mean → qk_mean → reciprocal → kv → output
    #
    # 硬體化修正（三點）：
    #   1. 除以 N：乘定點常數 round(1/N × 2^16)/2^16 → RTL 用常數乘法 + 右移 16
    #   2. eps=1e-5 在 Q8.8 round 成 0，改用 clamp to 1/256（Q8.8 最小正值）
    #   3. reciprocal：Newton-Raphson (num_iter=1)，對應 RTL LUT + 1 次 NR 迭代
    rcp_n = float(round(1.0 / N * 65536)) / 65536.0     # ≈ 1/N，誤差 < 0.1%
    k_mean      = fp(k.sum(axis=-2, keepdims=True) * rcp_n)               # [B, H, 1, d]
    qk_mean     = fp(q @ k_mean.transpose(0, 1, 3, 2))                    # [B, H, N, 1]
    qk_mean_eps = fp(np.maximum(qk_mean, 1.0 / 256))                      # clamp ≥ min Q8.8
    z_recip     = fp(_recip_nr(qk_mean_eps, num_iter=1))                   # [B, H, N, 1]
    kv          = fp((k.transpose(0, 1, 3, 2) @ v) * rcp_n)               # [B, H, d, d]
    attn_out    = fp(fp(q @ kv) * z_recip)                                 # [B, H, N, d]

    # reshape → proj linear → Q8.8
    attn_out = fp(attn_out.transpose(0, 2, 1, 3).reshape(B, N, C))
    attn_out = fp(linear(attn_out, proj_w, proj_b))
    save_npy(f"{pf}_after_attn_attn_out.npy", attn_out)
    return attn_out


# ---------------------------------------------------------------------------
# Transformer block
# 對齊 vit_CARE_relu6_fixed_dump.BlockDump.forward
# MLP activation = ReLU（不是 ReLU6；backbone 的 act_layer 預設為 nn.ReLU）
# ---------------------------------------------------------------------------

def block_forward(x: np.ndarray, block_idx: int, wp: Path) -> np.ndarray:
    pf  = f"backbone_blocks_{block_idx}"
    lp  = wp / "linearParam"
    lap = wp / "layerParam"

    norm1_w = load_w(lap / f"{pf}_norm1_weight.npy")
    write_wbi(norm1_w, f"{pf}_norm1_weight")
    norm1_b = load_w(lap / f"{pf}_norm1_bias.npy")
    write_wbi(norm1_b, f"{pf}_norm1_bias")
    norm2_w = load_w(lap / f"{pf}_norm2_weight.npy")
    write_wbi(norm2_w, f"{pf}_norm2_weight")
    norm2_b = load_w(lap / f"{pf}_norm2_bias.npy")
    write_wbi(norm2_b, f"{pf}_norm2_bias")
    fc1_w   = load_w(lp  / f"{pf}_mlp_fc1_weight.npy")
    write_wbi(fc1_w,   f"{pf}_mlp_fc1_weight")
    fc1_b   = load_w(lp  / f"{pf}_mlp_fc1_bias.npy")
    write_wbi(fc1_b,   f"{pf}_mlp_fc1_bias")
    fc2_w   = load_w(lp  / f"{pf}_mlp_fc2_weight.npy")
    write_wbi(fc2_w,   f"{pf}_mlp_fc2_weight")
    fc2_b   = load_w(lp  / f"{pf}_mlp_fc2_bias.npy")
    write_wbi(fc2_b,   f"{pf}_mlp_fc2_bias")

    # norm1 → attention → residual 1
    x_norm1 = fp(layer_norm(x, norm1_w, norm1_b))
    save_npy(f"{pf}_after_norm1_out.npy", x_norm1)

    attn_out = attention_forward(x_norm1, block_idx, wp)
    x = fp(x + attn_out)
    save_npy(f"{pf}_after_residual_add1_out.npy", x)

    # norm2 → MLP（fc1→Q8.8→relu→Q8.8→fc2→Q8.8）→ residual 2
    x_norm2 = fp(layer_norm(x, norm2_w, norm2_b))
    save_npy(f"{pf}_after_norm2_out.npy", x_norm2)

    mlp_out = fp(
        linear(
            fp(relu(fp(linear(x_norm2, fc1_w, fc1_b)))),
            fc2_w, fc2_b
        )
    )
    save_npy(f"{pf}_mlp_after_mlp_out.npy", mlp_out)

    x = fp(x + mlp_out)
    save_npy(f"{pf}_after_block_out.npy", x)
    return x


# ---------------------------------------------------------------------------
# Head conv branch（ctr / size / offset）
# 對齊 head_hand.CenterPredictor.get_score_map
# conv1~4：folded BN（relu），conv5：raw conv（ctr/size 加 sigmoid，offset 不加）
# ---------------------------------------------------------------------------

def head_branch(x: np.ndarray, branch: str, wp: Path) -> np.ndarray:
    fb = wp / "foldedBN"
    cp = wp / "convParam"

    for i in range(1, 5):
        pf = f"box_head_conv{i}_{branch}"
        w = load_w(fb / f"{pf}_folded_weight.npy")
        write_wbi(w, f"{pf}_folded_weight")
        b = load_w(fb / f"{pf}_folded_bias.npy")
        write_wbi(b, f"{pf}_folded_bias")
        x = fp(relu(conv2d(x, w, b, padding=1)))
        save_npy(f"{pf}_out.npy", x)

    w5 = load_w(cp / f"box_head_conv5_{branch}_weight.npy")
    write_wbi(w5, f"box_head_conv5_{branch}_weight")
    b5 = load_w(cp / f"box_head_conv5_{branch}_bias.npy")
    write_wbi(b5, f"box_head_conv5_{branch}_bias")
    x5 = fp(conv2d(x, w5, b5, padding=0))
    if branch in ("ctr", "size"):
        x5 = sigmoid_clamped(x5)
    x5 = x5.astype(np.float32)
    save_npy(f"box_head_conv5_{branch}_out.npy", x5)
    return x5


# ---------------------------------------------------------------------------
# cal_bbox
# 對齊 head_hand.CenterPredictor.cal_bbox
# ---------------------------------------------------------------------------

def cal_bbox(score_map_ctr: np.ndarray,
             size_map: np.ndarray,
             offset_map: np.ndarray) -> np.ndarray:
    """回傳 [1, 4] = [cx/feat_sz, cy/feat_sz, w, h]（皆已正規化）。"""
    flat = score_map_ctr.reshape(1, -1)                       # [1, FEAT_LEN]
    idx = int(np.argmax(flat, axis=1)[0])
    idx_y, idx_x = idx // FEAT_SZ, idx % FEAT_SZ

    size   = size_map.reshape(1, 2, -1)[:, :, idx]            # [1, 2]
    offset = offset_map.reshape(1, 2, -1)[:, :, idx]          # [1, 2]

    # RTL：(idx_int × 256 + offset_q88) >> 4，等同 /16（FEAT_SZ=16=2^4）
    offset_x_q88 = int(round(float(offset[0, 0]) * 256))   # Q8.8 → int
    offset_y_q88 = int(round(float(offset[0, 1]) * 256))
    cx = ((int(idx_x) * 256 + offset_x_q88) >> 4) / 256.0  # >> 4 in RTL，/256 僅 numpy 還原
    cy = ((int(idx_y) * 256 + offset_y_q88) >> 4) / 256.0  # >> 4 in RTL，/256 僅 numpy 還原
    w  = float(size[0, 0])
    h  = float(size[0, 1])
    return np.array([[cx, cy, w, h]], dtype=np.float32)        # [1, 4]


# ---------------------------------------------------------------------------
# Tracker post-processing helpers
# 對齊 dump_golden_intermediate.py 的 _map_box_back / clip_box
# ---------------------------------------------------------------------------

def map_box_back(state_xywh: list, pred_box_cxcywh: list,
                 resize_factor: float) -> list:
    cx_prev = state_xywh[0] + 0.5 * state_xywh[2]
    cy_prev = state_xywh[1] + 0.5 * state_xywh[3]
    cx, cy, w, h = pred_box_cxcywh
    half_side = 0.5 * SEARCH_SIZE / resize_factor  # ⚠ RTL: resize_factor 為執行期 float，需 reciprocal LUT 或 Newton-Raphson 迭代
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pure-numpy SGLATrack backbone+head from pre-computed post-embed npy."
    )
    p.add_argument(
        "--golden-dir", required=True,
        help="含 template_after_pos_add_out.npy / search_after_pos_add_out.npy 的目錄",
    )
    p.add_argument(
        "--weight-dir",
        default="output/exported_npy/vit_coco_uav123_care_relu6_ep0050_all",
        help="exported weight npy 根目錄（來自 export_checkpoint_npy.py）",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="計算結果 npy 輸出目錄",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # 讀取 golden manifest（用於 tracker 後處理的參數）
    manifest: dict = {}
    manifest_path = gd / "golden_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    # ------------------------------------------------------------------
    # Step 1: 載入 backbone 輸入 activation（已是 Q8.8，再做一次 fp 確保對齊）
    # ------------------------------------------------------------------
    z = fp(np.load(gd / "template_after_pos_add_out.npy").astype(np.float32))  # [1, 64, 768]
    x = fp(np.load(gd / "search_after_pos_add_out.npy").astype(np.float32))    # [1, 256, 768]
    save_npy("template_post_embed_input.npy", z)
    save_npy("search_post_embed_input.npy", x)

    # ------------------------------------------------------------------
    # Step 2: combine_tokens（mode="direct" = concat）+ pos_drop（eval = identity）
    # ------------------------------------------------------------------
    merged = fp(np.concatenate([z, x], axis=1))    # [1, 320, 768]
    save_npy("merged_tokens.npy", merged)
    x = fp(merged)
    save_npy("after_pos_drop_out.npy", x)

    # ------------------------------------------------------------------
    # Step 3: blocks 0..START_LAYER（固定層，所有 sample 都跑）
    # ------------------------------------------------------------------
    for i in range(START_LAYER + 1):               # i = 0, 1, 2, 3, 4, 5
        x = block_forward(x, i, wp)

    # ------------------------------------------------------------------
    # Step 4: adaptive selector
    #
    # 優先從 golden_dir 載入 dump_golden_intermediate.py 已存的 adaptive 結果，
    # 確保選出的 block 與 PyTorch golden 完全一致。
    #
    # 為什麼不重算：
    #   numpy linear（BLAS matmul）與 lib/module/Linear（element-wise × then sum）
    #   浮點累加順序不同，blocks 0~5 的輸出可能差 1 bit，導致 MLP 輸入微小差異，
    #   最壞情況是 selected block 不同，讓後續所有比對失去意義。
    #
    # fallback：若 golden 裡沒有這些檔案（第一次跑），才從 numpy 重算 MLP。
    # ------------------------------------------------------------------
    golden_topk_path = gd / "adaptive_sorted_topk_indices.npy"
    golden_pro_path  = gd / "adaptive_pro.npy"

    if golden_topk_path.exists():
        sorted_topk    = np.load(golden_topk_path)               # [1, 1], int64
        selected_block = int(sorted_topk.flat[0])
        pro = np.load(golden_pro_path).astype(np.float32) if golden_pro_path.exists() else None
        print(f"[adaptive] loaded from golden: selected_block={selected_block}")
    else:
        # fallback：直接從 numpy 計算（結果可能與 golden 不完全一致）
        lp     = wp / "linearParam"
        mlp1_w = load_w(lp / "backbone_MLP_fc1_weight.npy")
        write_wbi(mlp1_w, "backbone_MLP_fc1_weight")
        mlp1_b = load_w(lp / "backbone_MLP_fc1_bias.npy")
        write_wbi(mlp1_b, "backbone_MLP_fc1_bias")
        mlp2_w = load_w(lp / "backbone_MLP_fc2_weight.npy")
        write_wbi(mlp2_w, "backbone_MLP_fc2_weight")
        mlp2_b = load_w(lp / "backbone_MLP_fc2_bias.npy")
        write_wbi(mlp2_b, "backbone_MLP_fc2_bias")
        mlp_in = x[:, :, 0]
        h_mlp  = relu(linear(mlp_in, mlp1_w, mlp1_b))
        pro    = sigmoid(linear(h_mlp, mlp2_w, mlp2_b))
        selected_idx   = int(np.argmax(pro, axis=1)[0])
        selected_block = selected_idx + START_LAYER + 1
        print(f"[adaptive] computed from numpy (no golden): selected_block={selected_block}")

    save_npy("adaptive_pro.npy",
             pro if pro is not None else np.zeros((1, 6), dtype=np.float32))
    save_npy("adaptive_sorted_topk_indices.npy",
             np.array([[selected_block]], dtype=np.int64))
    save_npy("adaptive_selected_layer_index.npy",
             np.array([selected_block], dtype=np.int64))

    # ------------------------------------------------------------------
    # Step 5: 跑 selected block（6~11 中的一個）
    # ------------------------------------------------------------------
    x = block_forward(x, selected_block, wp)

    # ------------------------------------------------------------------
    # Step 6: recover_tokens（mode="direct" = no-op）+ backbone final norm
    # ------------------------------------------------------------------
    save_npy("backbone_after_recover_tokens_out.npy", fp(x))

    lap    = wp / "layerParam"
    norm_w = load_w(lap / "backbone_norm_weight.npy")
    write_wbi(norm_w, "backbone_norm_weight")
    norm_b = load_w(lap / "backbone_norm_bias.npy")
    write_wbi(norm_b, "backbone_norm_bias")
    backbone_out = fp(layer_norm(x, norm_w, norm_b))        # [1, 320, 768]
    save_npy("backbone_after_norm_backbone_out.npy", backbone_out)

    # ------------------------------------------------------------------
    # Step 7: forward_head reshape
    # 對齊 sglatrack.forward_head 的 permute+view：
    #   enc_opt = cat_feature[:, -FEAT_LEN:]        → [1, 256, 768]
    #   opt     = enc_opt.unsqueeze(-1)             → [1, 256, 768, 1]
    #   opt     = opt.permute(0, 3, 2, 1)           → [1, 1, 768, 256]
    #   opt_feat = opt.view(-1, C, feat_sz, feat_sz) → [1, 768, 16, 16]
    # ------------------------------------------------------------------
    enc_opt  = backbone_out[:, -FEAT_LEN:]                          # [1, 256, 768]
    opt      = enc_opt[:, :, :, np.newaxis]                         # [1, 256, 768, 1]
    opt      = opt.transpose(0, 3, 2, 1)                            # [1, 1, 768, 256]
    opt_feat = opt.reshape(-1, EMBED_DIM, FEAT_SZ, FEAT_SZ)         # [1, 768, 16, 16]

    # ------------------------------------------------------------------
    # Step 8: head conv 三個分支
    # ------------------------------------------------------------------
    score_map_ctr = head_branch(opt_feat, "ctr",    wp)    # [1, 1, 16, 16]
    size_map      = head_branch(opt_feat, "size",   wp)    # [1, 2, 16, 16]
    offset_map    = head_branch(opt_feat, "offset", wp)    # [1, 2, 16, 16]

    # ------------------------------------------------------------------
    # Step 9: cal_bbox → pred_boxes（對齊 sglatrack.forward_head 的 reshape）
    # ------------------------------------------------------------------
    bbox       = cal_bbox(score_map_ctr, size_map, offset_map)     # [1, 4]
    pred_boxes = fp(bbox.reshape(1, 1, 4))                          # [1, 1, 4]
    save_npy("box_head_after_forward_head_pred_boxes.npy", pred_boxes)

    # ------------------------------------------------------------------
    # Step 10: tracker 後處理（需 manifest，若不存在則略過）
    # ------------------------------------------------------------------
    if manifest:
        x_resize_factor = float(manifest.get("search_crop_resize_factor", 1.0))
        state           = manifest.get("init_bbox_xywh", [0.0, 0.0, 1.0, 1.0])
        search_size     = int(manifest.get("search_size", SEARCH_SIZE))
        frame2_path     = manifest.get("frame2", "")
        H, W = 1080, 1920   # fallback
        if frame2_path and os.path.exists(frame2_path):
            import cv2
            img = cv2.imread(frame2_path)
            if img is not None:
                H, W = img.shape[:2]

        window   = hann2d(FEAT_SZ, FEAT_SZ, centered=True)            # [1,1,16,16]
        response = fp(window * score_map_ctr)
        save_npy("tracker_after_output_window_response.npy", response)

        bbox_after = fp(cal_bbox(response, size_map, offset_map))      # [1, 4]
        save_npy("tracker_after_cal_bbox_bbox.npy", bbox_after)

        pred_box = (bbox_after[0] * search_size / x_resize_factor).tolist()  # ⚠ RTL: /x_resize_factor 為執行期 float，同 map_box_back
        mapped   = map_box_back(state, pred_box, x_resize_factor)
        save_npy("tracker_after_map_box_back_bbox.npy",
                 fp(np.array(mapped, dtype=np.float32)))

        final_bbox = clip_box_numpy(mapped, H, W, margin=10)
        save_npy("tracker_after_final_bbox_bbox.npy",
                 fp(np.array(final_bbox, dtype=np.float32)))
        x1, y1, bw, bh = final_bbox
        print(
            f"[tracker] 最終 bbox（frame2 像素 xywh）: "
            f"x1={x1:.4f}, y1={y1:.4f}, w={bw:.4f}, h={bh:.4f}"
        )
    else:
        print("[WARNING] golden_manifest.json not found; skipping tracker post-processing.")
        print("[tracker] 最終 bbox: 未計算（無 manifest，已略過後處理）")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"Output dir     : {_out_dir}")
    print(f"  Activation/  : {_bi_act_dir}")
    print(f"  Weight/      : {_bi_wgt_dir}")
    print(f"Selected block : {selected_block}  (pro={pro.tolist()})")
    print(f"pred_boxes     : {pred_boxes.tolist()}")
    print(f"score_map range: [{float(score_map_ctr.min()):.4f}, {float(score_map_ctr.max()):.4f}]")


if __name__ == "__main__":
    main()
