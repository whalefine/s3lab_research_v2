"""Pure-numpy SGLATrack backbone + head（dim256、Q8.8）。

使用方式：
    python tracking/run_backbone_numpy_dim256_q88.py \\
      --golden-dir output/golden/vit_care_relu6_dim256_fixed_q88_golden \\
      --weight-dir output/npy/vit_coco_got10k_distill_care_relu6_teacher_afkd_s60000_bs32_dim256_ep0050 \\
      --output-dir output/golden/vit_care_relu6_numpy_dim256_q88_out

設計原則：
- 純 numpy，不建立任何 PyTorch model 或呼叫 .forward()
- template_after_pos_add_out.npy / search_after_pos_add_out.npy 以 np.load 從
  --golden-dir 載入，作為 backbone 的真正輸入
- 所有 weight 從 --weight-dir 的 exported npy 載入；運算時 round 成 Q8.8 整數
- datapath 對齊 ``verilog_dim256``：整數 MAC 統一 rnd_shr8=(acc+128)>>8、
  LUT+NR inv_sqrt/recip、layer_norm_pip（常數 dim256）
- 注意：``verilog_dim256_backbone`` 若仍是 trunc >>>8，需改成 +128 才能對拍
- 目標：Activation/*_bi.txt 可與 RTL bit-accurate 對拍，同時維持可接受 IoU
- adaptive selector MLP 屬純軟體路徑，保持 float32 不量化
- write_bi() 用 round（與 MAC 權重量化一致）；改動後須重產 ROM（§20）
- 模型常數：embed_dim=256、num_heads=8、head_dim=32、S_Q88=108、LN_RCP=256
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
# Model constants (vit_CARE_relu6_dim256_fixed_q88)
# ---------------------------------------------------------------------------
EMBED_DIM = 256
NUM_HEADS = 8
HEAD_DIM = EMBED_DIM // NUM_HEADS        # 32
SCALE = HEAD_DIM ** -0.5                 # ≈ 0.176777
S = SCALE ** 0.5                         # ≈ 0.420448
LENS_Z = 64
LENS_X = 256
N_TOKENS = LENS_Z + LENS_X              # 320
START_LAYER = 5
LN_EPS = 1e-6
FEAT_SZ = 16                             # search_size // stride = 256 // 16
FEAT_LEN = FEAT_SZ * FEAT_SZ            # 256
SEARCH_SIZE = 256

# CARE / LN constants — must match verilog_dim256_backbone (care_attention / layer_norm_pip)
S_Q88 = 108                              # round(HEAD_DIM^-0.25 * 256)
RELU6_MAX_Q88 = 1536                     # 6.0 * 256
RCP_N_NUM = 205                          # round(65536 / N_TOKENS)
RCP_N_SHIFT = 16
KV_Q88_EXTRA_SHIFT = 8
KV_Q88_ROUND = 1 << (RCP_N_SHIFT + KV_Q88_EXTRA_SHIFT - 1)  # 2^23
LN_RCP = 256                             # round(2^16 / EMBED_DIM)
_INV_SQRT_NR_ITERS = 3
_RECIP_NR_ITERS = 1

_INV_SQRT_LUT_SEED_Q88 = np.array(
    [2364, 1671, 1182, 836, 591, 418, 296, 209, 148, 105, 74, 52, 37, 26, 18, 13],
    dtype=np.int64,
)
_RECIP_LUT_Y0 = np.array(
    [32767, 21845, 10922, 5461, 2731, 1365, 683, 341, 171, 85, 43, 21, 11, 5, 3, 2],
    dtype=np.int64,
)


# ---------------------------------------------------------------------------
# Numpy primitives（Q8.8 integer；對齊 verilog_dim256）
# ---------------------------------------------------------------------------

def to_fixed_point(x: np.ndarray, int_bits: int, frac_bits: int) -> np.ndarray:
    """有號定點數量化：scale → round → saturate → descale。"""
    scale = 2 ** frac_bits
    qmin = -(2 ** (int_bits + frac_bits - 1))
    qmax = (2 ** (int_bits + frac_bits - 1)) - 1
    scaled = x.astype(np.float64) * scale
    rounded = np.round(scaled)
    saturated = np.clip(rounded, qmin, qmax)
    return (saturated / scale).astype(np.float32)


def fp(x: np.ndarray) -> np.ndarray:
    """Shorthand: to_fixed_point(x, 8, 8)."""
    return to_fixed_point(x, 8, 8)


def from_q88(x_q: np.ndarray) -> np.ndarray:
    return (np.asarray(x_q, dtype=np.float64) / 256.0).astype(np.float32)


def as_q88(x: np.ndarray) -> np.ndarray:
    return np.round(x.astype(np.float64) * 256.0).astype(np.int32)


def sat16(v: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.int64), -32768, 32767).astype(np.int32)


def rnd_shr8(v: np.ndarray) -> np.ndarray:
    return sat16((np.asarray(v, dtype=np.int64) + 128) >> 8)


def sat16_from33(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    return np.where(v > 0x7FFF, 0x7FFF, np.where(v < -0x8000, -0x8000, v)).astype(np.int32)


def sat16_from48(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    return np.where(v > 0x7FFF, 0x7FFF, np.where(v < -0x8000, -0x8000, v)).astype(np.int32)


def sat16_from49(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    return np.where(v > 0x7FFF, 0x7FFF, np.where(v < -0x8000, -0x8000, v)).astype(np.int32)


def relu6_q88(x_q: np.ndarray) -> np.ndarray:
    x = np.asarray(x_q, dtype=np.int64)
    x = np.where(x < 0, 0, x)
    return np.where(x > RELU6_MAX_Q88, RELU6_MAX_Q88, x).astype(np.int32)


def w_q88(w: np.ndarray) -> np.ndarray:
    return np.round(w.astype(np.float64) * 256.0).astype(np.int32)


def _inv_sqrt_nr_q88_fixed(var_q88_int: np.ndarray) -> np.ndarray:
    """Bit-accurate vs inv_sqrt_lut_seed.v + inv_sqrt_nr.v（3 NR + round）。"""
    v = np.asarray(var_q88_int, dtype=np.int64)
    v_eps = np.where(v <= 0, np.int64(1), v).astype(np.int64)

    msb = np.zeros_like(v_eps)
    tmp = v_eps.copy()
    for bit in range(15, -1, -1):
        hit = (tmp & (np.int64(1) << bit)) != 0
        msb = np.where((msb == 0) & hit, np.int64(bit), msb)

    y = _INV_SQRT_LUT_SEED_Q88[msb].astype(np.int64)
    for _ in range(_INV_SQRT_NR_ITERS):
        y_sq = (y * y + 128) >> 8
        term = (v_eps * y_sq + 256) >> 9
        coeff = np.int64(384) - term
        y_new = (y * coeff + 128) >> 8
        y_new = ((y_new + 0x8000) & 0xFFFF) - 0x8000
        y = y_new
    return y


def _recip_msb_k(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.int64), 1)
    k = np.zeros_like(x, dtype=np.int64)
    for bit in range(15, -1, -1):
        hit = (x >= (1 << bit)) & (k == 0)
        k = np.where(hit, bit, k)
    return k


def _trunc_q88_slice32(v: np.ndarray) -> np.ndarray:
    """Match recip_nr.v xy_raw[23:8] / y_new_raw[23:8] (truncate, no round)."""
    v = np.asarray(v, dtype=np.int64)
    s = (v >> 8) & 0xFFFF
    return np.where(s >= 0x8000, s - 0x10000, s).astype(np.int64)


def _recip_nr_q88_fixed(x_q88: np.ndarray) -> np.ndarray:
    """Bit-accurate vs recip_lut_seed.v + recip_nr.v（1 NR）。"""
    x = np.maximum(np.asarray(x_q88, dtype=np.int64), 1)
    y = _RECIP_LUT_Y0[_recip_msb_k(x)].astype(np.int64)
    for _ in range(_RECIP_NR_ITERS):
        coeff = 512 - _trunc_q88_slice32(x * y)
        y = _trunc_q88_slice32(y * coeff)
        y = np.clip(y, -32768, 32767).astype(np.int64)
    return y.astype(np.int32)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = LN_EPS, inv_sqrt_iter: int = 2) -> np.ndarray:
    """Bit-accurate LayerNorm vs layer_norm_pip.v（FEAT_DIM=256, RCP_NUM=256）。"""
    N = x.shape[-1]
    if N != EMBED_DIM:
        raise ValueError(f"layer_norm expects FEAT_DIM={EMBED_DIM} (got {N})")
    RCP = LN_RCP
    _ = eps, inv_sqrt_iter  # RTL uses var<=0 → 1 LSB，無 float eps

    x_int = np.round(x.astype(np.float64) * 256.0).astype(np.int64)
    w_int = np.round(weight.astype(np.float64) * 256.0).astype(np.int64)
    b_int = np.round(bias.astype(np.float64) * 256.0).astype(np.int64)

    def _sat16(v):
        return np.clip(v, -32768, 32767).astype(np.int64)

    def _rnd_q16(v):
        return _sat16((v + 128) >> 8)

    sum_int = x_int.sum(axis=-1, keepdims=True)
    mean_int = _sat16((sum_int * RCP + 32768) >> 16)
    centered = _sat16(x_int - mean_int)
    sum_sq = (centered * centered).sum(axis=-1, keepdims=True)
    var_int = np.clip((sum_sq * RCP + 8388608) >> 24, -32768, 32767).astype(np.int64)
    inv_std = _inv_sqrt_nr_q88_fixed(var_int)

    ci_std = _rnd_q16(centered * inv_std)
    wci = _rnd_q16(w_int[..., :] * ci_std)
    y_int = _sat16(wci + b_int[..., :])
    return (y_int.astype(np.float64) / 256.0).astype(np.float32)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Q8.8 integer MAC linear（rnd_shr8，對齊 verilog_dim256/mlp_ws）。

    MAC 後統一：(acc + 128) >> 8（四捨五入），再加 bias。
    勿用 trunc >>>8：與 verilog_dim256 不一致，且 IoU 較差。
    """
    _SCALE = 1 << 8
    x_int = np.round(x.astype(np.float64) * _SCALE).astype(np.int32)
    w_int = np.round(weight.astype(np.float64) * _SCALE).astype(np.int32)

    *batch, in_dim = x_int.shape
    out_dim = w_int.shape[0]
    x64 = x_int.reshape(-1, in_dim).astype(np.int64)
    w64 = w_int.astype(np.int64)
    acc_q88 = (x64 @ w64.T + 128) >> 8
    if bias is not None:
        bias_int = np.round(bias.astype(np.float64) * _SCALE).astype(np.int64)
        acc_q88 = acc_q88 + bias_int
    return (acc_q88.reshape(*batch, out_dim).astype(np.float64) / _SCALE).astype(np.float32)


def linear_sat_q88(x_q: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Q8.8 linear MAC + rnd_shr8 + bias + sat16（對齊 verilog_dim256/care_attention）。"""
    x_int = np.asarray(x_q, dtype=np.int64).reshape(-1, weight.shape[1])
    w_int = w_q88(weight).astype(np.int64)
    acc = (x_int @ w_int.T + 128) >> 8
    if bias is not None:
        acc = acc + np.round(bias.astype(np.float64) * 256.0).astype(np.int64)
    out = sat16(acc)
    return out.astype(np.int32).reshape(*np.asarray(x_q).shape[:-1], weight.shape[0])


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(np.float32)


def relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 6.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Sigmoid LUT
# ---------------------------------------------------------------------------
_SIGMOID_LUT_N = 64
_SIGMOID_LUT = 1.0 / (1.0 + np.exp(
    -np.linspace(-8.0, 8.0, _SIGMOID_LUT_N + 1).astype(np.float64)
))
_SIGMOID_LUT_INT = np.round(_SIGMOID_LUT * 256).astype(np.int32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """LUT + 線性插值 sigmoid，對齊 sigmoid_lut.v。"""
    x_int = np.round(
        np.clip(x.astype(np.float64), -8.0, 8.0) * 256.0
    ).astype(np.int32)
    shifted = x_int + 2048
    idx = np.clip(shifted >> 6, 0, _SIGMOID_LUT_N - 1).astype(np.int32)
    frac6 = (shifted & 0x3F).astype(np.int32)
    lo_int = _SIGMOID_LUT_INT[idx]
    hi_int = _SIGMOID_LUT_INT[idx + 1]
    delta = (hi_int - lo_int).astype(np.int32)
    result = (lo_int * 64 + delta * frac6) >> 6
    return (result.astype(np.float64) / 256.0).astype(np.float32)


def sigmoid_clamped(x: np.ndarray) -> np.ndarray:
    _LB = 1.0 / 256
    _UB = 255.0 / 256
    return np.clip(fp(sigmoid(x)), _LB, _UB).astype(np.float32)


def conv2d(x: np.ndarray, weight: np.ndarray,
           bias: Optional[np.ndarray] = None, padding: int = 1) -> np.ndarray:
    """Q8.8 integer MAC conv2d（rnd_shr8，對齊 verilog_dim256/conv2d_ws）。"""
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
        raise ValueError(
            f"conv2d output size invalid: padded {x_int.shape[2:]}, kernel ({kH},{kW})"
        )

    acc = np.zeros((N, C_out, H_out, W_out), dtype=np.int64)
    for n in range(N):
        for oc in range(C_out):
            w_oc = w_int[oc].astype(np.int64)
            for oh in range(H_out):
                for ow in range(W_out):
                    patch = x_int[n, :, oh:oh + kH, ow:ow + kW].astype(np.int64)
                    acc[n, oc, oh, ow] = np.sum(patch * w_oc)

    acc_q88 = (acc + 128) >> 8
    if bias is not None:
        bias_int = np.round(bias.astype(np.float64) * _SCALE).astype(np.int64)
        acc_q88 += bias_int[np.newaxis, :, np.newaxis, np.newaxis]

    return (acc_q88.astype(np.float64) / _SCALE).astype(np.float32)


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
    """寫出十進位 .txt 和二補數 binary _bi.txt（round，對齊 MAC / ROM）。"""
    flat = arr.flatten()
    scale = 1 << frac_bits
    total_w = int_bits + frac_bits
    min_int = -(1 << (int_bits - 1)) * scale
    max_int = (1 << (int_bits - 1)) * scale - 1
    base_str = str(base)
    with open(base_str + '.txt', 'w') as f_dec, \
         open(base_str + '_bi.txt', 'w') as f_bin:
        for num in flat:
            fixed = int(round(float(num) * scale))
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
        if np.issubdtype(arr.dtype, np.integer):
            write_bi(from_q88(arr), _bi_act_dir / stem)
        else:
            write_bi(arr, _bi_act_dir / stem)


def load_w(path: Path) -> np.ndarray:
    """載入 weight float32；進 MAC 前由 w_q88 / linear 做 Q8.8 round。"""
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# Attention (CARE ReLU6 Q8.8 integer — 對齊 care_attention.v)
# ---------------------------------------------------------------------------

def _care_split_qk_q88(q: np.ndarray, k: np.ndarray) -> tuple:
    q = relu6_q88(rnd_shr8(q.astype(np.int64) * S_Q88))
    k = relu6_q88(rnd_shr8(k.astype(np.int64) * S_Q88))
    return q, k


def _care_k_mean_q88(k: np.ndarray) -> np.ndarray:
    k_sum = k.astype(np.int64).sum(axis=2)
    km_scaled = k_sum * RCP_N_NUM
    return sat16_from48((km_scaled + 32768) >> RCP_N_SHIFT)


def _care_qk_mean_q88(q: np.ndarray, km: np.ndarray) -> np.ndarray:
    """Vectorized S_QK_MEAN；語意等同 shared_trunk 逐元素 rnd_shr8。"""
    acc = np.einsum("bhnd,bhd->bhn", q.astype(np.int64), km.astype(np.int64))
    return sat16_from33((acc + 128) >> 8)[0]  # (H, N)


def _care_kv_q88(k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """S_KV: kv[h,d_out,d_k] = mean_n(k*v)。"""
    _B, H, _N, d = k.shape
    kv = np.zeros((H, d, d), dtype=np.int32)
    for h in range(H):
        acc = k[0, h].astype(np.int64).T @ v[0, h].astype(np.int64)
        kv_scaled = acc * RCP_N_NUM
        kv[h] = sat16_from48(
            (kv_scaled + KV_Q88_ROUND) >> (RCP_N_SHIFT + KV_Q88_EXTRA_SHIFT)
        )
    return kv


def _care_attn_q88(q: np.ndarray, kv: np.ndarray, zr: np.ndarray) -> np.ndarray:
    """S_ATTN: ao = rnd_shr8(sat(q@kv) * zr)；kv[h,dk,dout]。"""
    _B, H, N, d = q.shape
    ao = np.zeros((N, EMBED_DIM), dtype=np.int32)
    for h in range(H):
        dots = q[0, h].astype(np.int64) @ kv[h].astype(np.int64)
        dot_sat = sat16_from49((dots + 128) >> 8)
        zr_h = zr[h].astype(np.int64)[:, np.newaxis]
        ao[:, h * d:(h + 1) * d] = rnd_shr8(dot_sat.astype(np.int64) * zr_h)
    return ao.reshape(1, N, EMBED_DIM)


def attention_forward(x: np.ndarray, block_idx: int, wp: Path) -> np.ndarray:
    """CARE attention Q8.8 整數路徑；回傳 float32 供 residual。"""
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

    x_q = as_q88(x)
    qkv_q88 = linear_sat_q88(x_q, qkv_w, qkv_b)
    qkv = qkv_q88.reshape(B, N, 3, H, d).transpose(2, 0, 3, 1, 4)
    q = qkv[0].astype(np.int32)
    k = qkv[1].astype(np.int32)
    v = qkv[2].astype(np.int32)
    save_npy(f"{pf}_attn_after_qkv_q.npy", q)
    save_npy(f"{pf}_attn_after_qkv_k.npy", k)
    save_npy(f"{pf}_attn_after_qkv_v.npy", v)

    q, k = _care_split_qk_q88(q, k)
    km = _care_k_mean_q88(k)
    qkm = _care_qk_mean_q88(q, km)
    qkm_eps = np.maximum(qkm, 1).astype(np.int32)
    zr = _recip_nr_q88_fixed(qkm_eps)
    kv = _care_kv_q88(k, v)
    ao_q88 = _care_attn_q88(q, kv, zr)

    attn_q88 = linear_sat_q88(ao_q88, proj_w, proj_b)
    save_npy(f"{pf}_after_attn_attn_out.npy", attn_q88)
    return from_q88(attn_q88)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

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
    # 對齊 RTL 固定點：sum_x = idx*256 + off_x；後續 >>>4 等價於整數截斷鏈。
    offset_x_q88 = int(np.round(float(offset[0, 0]) * 256))   # Q8.8 → int（對齊 dump round）
    offset_y_q88 = int(np.round(float(offset[0, 1]) * 256))
    cx = ((int(idx_x) * 256 + offset_x_q88) >> 4) / 256.0  # >> 4 in RTL，/256 僅 numpy 還原
    cy = ((int(idx_y) * 256 + offset_y_q88) >> 4) / 256.0  # >> 4 in RTL，/256 僅 numpy 還原
    w  = float(size[0, 0])
    h  = float(size[0, 1])
    return np.array([[cx, cy, w, h]], dtype=np.float32)        # [1, 4]


def smooth_bbox_q88(bbox_cxcywh: np.ndarray,
                    target_size_norm: tuple[float, float],
                    center_alpha_q88: int,
                    size_alpha_q88: int) -> np.ndarray:
    """Q8.8 bbox smoothing，可直接映射到 RTL 乘加右移。

    alpha=256 保留模型輸出；alpha=0 完全使用 prior。
    center prior 是 search crop 中心 (0.5, 0.5)，size prior 是上一幀 bbox
    換算到 normalized search 座標後的 w/h。
    """
    ca = int(np.clip(center_alpha_q88, 0, 256))
    sa = int(np.clip(size_alpha_q88, 0, 256))
    bbox_q = as_q88(np.asarray(bbox_cxcywh, dtype=np.float32).reshape(1, 4)).astype(np.int64)
    target_w_q = int(np.round(float(target_size_norm[0]) * 256.0))
    target_h_q = int(np.round(float(target_size_norm[1]) * 256.0))
    target_q = np.array([[128, 128, target_w_q, target_h_q]], dtype=np.int64)
    alpha_q = np.array([[ca, ca, sa, sa]], dtype=np.int64)
    smoothed_q = sat16(((bbox_q * alpha_q) + (target_q * (256 - alpha_q)) + 128) >> 8)
    return from_q88(smoothed_q)


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


def iou_xywh(box_a, box_b) -> float:
    """IoU of two boxes in xywh (image pixel)."""
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _load_frame2_gt(manifest: dict):
    """Try to load frame2 ground-truth bbox from dataset annotation.

    Supports UAV123, UAV123_10fps, DTB70, UAVTrack112, UAVTrack.
    Returns [x, y, w, h] or None if annotation cannot be found.
    """
    frame1 = manifest.get("frame1", "")
    frame2 = manifest.get("frame2", "")
    if not frame1 or not frame2:
        return None
    seq = os.path.basename(os.path.dirname(frame1))
    frame2_stem = os.path.splitext(os.path.basename(frame2))[0]
    frame2_idx = int(frame2_stem) - 1

    anno_candidates = []
    data_root = frame1
    if "/UAV123/data_seq/UAV123/" in data_root:
        base = data_root.split("/UAV123/data_seq/UAV123/")[0]
        anno_candidates.append(os.path.join(base, "UAV123", "anno", "UAV123", f"{seq}.txt"))
    if "/UAV123_10fps/data_seq/UAV123_10fps/" in data_root:
        base = data_root.split("/UAV123_10fps/data_seq/UAV123_10fps/")[0]
        anno_candidates.append(os.path.join(base, "UAV123_10fps", "anno", "UAV123_10fps", f"{seq}.txt"))
    if "/DTB70/" in data_root:
        parent = os.path.dirname(os.path.dirname(frame1))
        anno_candidates.append(os.path.join(parent, "groundtruth_rect.txt"))
    if "/V4RFlight112/data_seq/" in data_root:
        base = data_root.split("/V4RFlight112/data_seq/")[0]
        anno_candidates.append(os.path.join(base, "V4RFlight112", "anno", f"{seq}.txt"))
        anno_candidates.append(os.path.join(base, "V4RFlight112", "anno_l", f"{seq}.txt"))

    for anno_path in anno_candidates:
        if not os.path.exists(anno_path):
            continue
        try:
            gt = np.loadtxt(anno_path, delimiter=",", dtype=np.float64)
            if gt.ndim == 1:
                gt = gt.reshape(1, -1)
            if frame2_idx < len(gt):
                return gt[frame2_idx].tolist()
        except Exception:
            continue
    return None


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
        default="output/npy/vit_coco_got10k_distill_care_relu6_teacher_afkd_s60000_bs32_dim256_ep0050",
        help="exported weight npy 根目錄（來自 export_checkpoint_npy.py）",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="計算結果 npy 輸出目錄",
    )
    p.add_argument(
        "--tracker-smooth-center-alpha-q88",
        type=int,
        default=128,
        help="tracker 中心 smoothing alpha(Q8.8): 256=原始模型輸出，0=search 中心 prior",
    )
    p.add_argument(
        "--tracker-smooth-size-alpha-q88",
        type=int,
        default=128,
        help="tracker 尺寸 smoothing alpha(Q8.8): 256=原始模型輸出，0=上一幀尺寸 prior",
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
    z = fp(np.load(gd / "template_after_pos_add_out.npy").astype(np.float32))  # [1, 64, 256]
    x = fp(np.load(gd / "search_after_pos_add_out.npy").astype(np.float32))    # [1, 256, 256]
    save_npy("template_post_embed_input.npy", z)
    save_npy("search_post_embed_input.npy", x)

    # ------------------------------------------------------------------
    # Step 2: combine_tokens（mode="direct" = concat）+ pos_drop（eval = identity）
    # ------------------------------------------------------------------
    merged = fp(np.concatenate([z, x], axis=1))    # [1, 320, 256]
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
    backbone_out = fp(layer_norm(x, norm_w, norm_b))        # [1, 320, 256]
    save_npy("backbone_after_norm_backbone_out.npy", backbone_out)

    # ------------------------------------------------------------------
    # Step 7: forward_head reshape
    # 對齊 sglatrack.forward_head 的 permute+view：
    #   enc_opt = cat_feature[:, -FEAT_LEN:]        → [1, 256, 256]
    #   opt     = enc_opt.unsqueeze(-1)             → [1, 256, 256, 1]
    #   opt     = opt.permute(0, 3, 2, 1)           → [1, 1, 256, 256]
    #   opt_feat = opt.view(-1, C, feat_sz, feat_sz) → [1, 256, 16, 16]
    # ------------------------------------------------------------------
    enc_opt  = backbone_out[:, -FEAT_LEN:]                          # [1, 256, 256]
    opt      = enc_opt[:, :, :, np.newaxis]                         # [1, 256, 256, 1]
    opt      = opt.transpose(0, 3, 2, 1)                            # [1, 1, 256, 256]
    opt_feat = opt.reshape(-1, EMBED_DIM, FEAT_SZ, FEAT_SZ)         # [1, 256, 16, 16]

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

        target_size_norm = (
            float(state[2]) * x_resize_factor / search_size,
            float(state[3]) * x_resize_factor / search_size,
        )
        ca = int(np.clip(args.tracker_smooth_center_alpha_q88, 0, 256))
        sa = int(np.clip(args.tracker_smooth_size_alpha_q88, 0, 256))
        # RTL bbox-smooth config（非 weight、非 feature map）
        # layout Q8.8: [center_alpha, size_alpha, target_w, target_h]
        #   target_w/h = init_bbox size in normalized search coords
        #   center prior 固定為 0.5 (=128)，不必寫進檔
        tw = float(target_size_norm[0])
        th = float(target_size_norm[1])
        save_npy(
            "tracker_smooth_config_q88.npy",
            from_q88(np.array(
                [ca, sa, int(np.round(tw * 256.0)), int(np.round(th * 256.0))],
                dtype=np.int32,
            )),
        )
        bbox_after = fp(smooth_bbox_q88(
            bbox_after,
            target_size_norm,
            ca,
            sa,
        ))
        save_npy("tracker_after_smooth_bbox_bbox.npy", bbox_after)

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
        print(
            f"[tracker] IoU(最終 bbox, init_bbox) = {iou_xywh(final_bbox, state):.4f} "
            f"(init xywh={float(state[0]):.4f},{float(state[1]):.4f},"
            f"{float(state[2]):.4f},{float(state[3]):.4f})"
        )
        gt2_bbox = _load_frame2_gt(manifest)
        if gt2_bbox is not None:
            iou_gt2 = iou_xywh(final_bbox, gt2_bbox)
            print(
                f"[tracker] IoU(最終 bbox, frame2_GT) = {iou_gt2:.4f} "
                f"(gt2 xywh={gt2_bbox[0]:.4f},{gt2_bbox[1]:.4f},"
                f"{gt2_bbox[2]:.4f},{gt2_bbox[3]:.4f})"
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
