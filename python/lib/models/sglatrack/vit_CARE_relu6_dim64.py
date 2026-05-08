"""ViT CARE ReLU6 變體：embed_dim=64，可由 ViT-Tiny (192 維) checkpoint 做左上角截斷載入。

與 vit_CARE_relu6_dim32.py 相同邏輯，僅 student 維度改為 64；工廠函數為 `vit_tiny64_care_patch16_224`。
預訓練載入：將 timm ViT-Tiny 的線性權重 [...,192] 投影為 [...,64]（取每個 Q/K/V 與 MLP 子塊的對應列／行前綴）。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from timm.models.helpers import named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.sglatrack.vit_CARE_relu6 import VisionTransformer


# ---------------------------------------------------------------------------
# 維度常數（ViT-Tiny teacher vs 本模型 student）
# ---------------------------------------------------------------------------
TEACHER_EMBED_DIM = 192
STUDENT_EMBED_DIM = 64
STUDENT_NUM_HEADS = 4  # 64 % 4 == 0
STUDENT_DEPTH = 12
MLP_RATIO = 4


def _unwrap_checkpoint(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """支援裸 state_dict、{'state_dict':...}、{'model':...}。"""
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            sd = checkpoint["model"]
        else:
            sd = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        key = k
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = v
    return out


def _project_qkv_weight(w_t: torch.Tensor, t_dim: int, s_dim: int) -> torch.Tensor:
    """w_t: [3*t_dim, t_dim] -> [3*s_dim, s_dim]"""
    assert w_t.shape == (3 * t_dim, t_dim)
    parts = []
    for h in range(3):
        block = w_t[h * t_dim : (h + 1) * t_dim, :]
        parts.append(block[:s_dim, :s_dim])
    return torch.cat(parts, dim=0)


def _project_qkv_bias(b_t: torch.Tensor, t_dim: int, s_dim: int) -> torch.Tensor:
    assert b_t.shape == (3 * t_dim,)
    parts = [b_t[h * t_dim : h * t_dim + s_dim] for h in range(3)]
    return torch.cat(parts, dim=0)


def _project_mlp_fc1(w_t: torch.Tensor, b_t: torch.Tensor, t_dim: int, s_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """fc1: [4*t_dim, t_dim], bias [4*t_dim]"""
    th, tw = 4 * t_dim, t_dim
    sh, sw = 4 * s_dim, s_dim
    return w_t[:sh, :sw].clone(), b_t[:sh].clone()


def _project_mlp_fc2(w_t: torch.Tensor, b_t: torch.Tensor, t_dim: int, s_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """fc2: [t_dim, 4*t_dim], bias [t_dim]"""
    return w_t[:s_dim, : 4 * s_dim].clone(), b_t[:s_dim].clone()


def project_vit_tiny_state_dict_to_dim64(
    teacher_sd: Dict[str, torch.Tensor],
    model: VisionTransformer,
    teacher_dim: int = TEACHER_EMBED_DIM,
    student_dim: int = STUDENT_EMBED_DIM,
) -> Dict[str, torch.Tensor]:
    """
    將 ViT-Tiny (embed_dim=192) 的 state_dict 鍵映射到本模型 (embed_dim=64)。
    僅覆寫可截斷對齊的參數；其餘鍵不放入 out，由 load_state_dict(strict=False) 保留隨機初始化。
    """
    s = student_dim
    t = teacher_dim
    student_sd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}

    def tbias(key_w: str) -> torch.Tensor:
        bkey = key_w.replace("weight", "bias")
        if bkey not in teacher_sd:
            raise KeyError(f"Teacher missing bias key: {bkey}")
        return teacher_sd[bkey]

    for k in student_sd.keys():
        if k not in teacher_sd:
            continue
        vt = teacher_sd[k]

        if k == "patch_embed.proj.weight":
            out[k] = vt[:s].clone()
        elif k == "patch_embed.proj.bias":
            out[k] = vt[:s].clone()
        elif k == "cls_token":
            out[k] = vt[:, :, :s].clone()
        elif k == "pos_embed":
            out[k] = vt[:, :, :s].clone()
        elif k == "norm.weight":
            out[k] = vt[:s].clone()
        elif k == "norm.bias":
            out[k] = vt[:s].clone()
        elif "norm1.weight" in k or "norm2.weight" in k:
            out[k] = vt[:s].clone()
        elif "norm1.bias" in k or "norm2.bias" in k:
            out[k] = vt[:s].clone()
        elif "attn.qkv.weight" in k:
            out[k] = _project_qkv_weight(vt, t, s)
        elif "attn.qkv.bias" in k:
            out[k] = _project_qkv_bias(vt, t, s)
        elif "attn.proj.weight" in k:
            out[k] = vt[:s, :s].clone()
        elif "attn.proj.bias" in k:
            out[k] = vt[:s].clone()
        elif "mlp.fc1.weight" in k:
            wn, bn = _project_mlp_fc1(vt, tbias(k), t, s)
            out[k] = wn
            out[k.replace("weight", "bias")] = bn
        elif "mlp.fc1.bias" in k:
            if k in out:
                continue
            out[k] = vt[: 4 * s].clone()
        elif "mlp.fc2.weight" in k:
            wn, bn = _project_mlp_fc2(vt, tbias(k), t, s)
            out[k] = wn
            out[k.replace("weight", "bias")] = bn
        elif "mlp.fc2.bias" in k:
            if k in out:
                continue
            out[k] = vt[:s].clone()
        else:
            vs = student_sd[k]
            if vt.shape == vs.shape:
                out[k] = vt.clone()

    return out


def load_vit_tiny_pretrained_dim64(model: VisionTransformer, checkpoint_path: str) -> Tuple[list, list]:
    """從 .pth 讀取 ViT-Tiny 權重，投影後寫入 64 維模型。回傳 (missing_keys, unexpected_keys)。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher_sd = _unwrap_checkpoint(checkpoint)
    projected = project_vit_tiny_state_dict_to_dim64(teacher_sd, model)
    incomp = model.load_state_dict(projected, strict=False)
    missing = getattr(incomp, "missing_keys", incomp[0])
    unexpected = getattr(incomp, "unexpected_keys", incomp[1])
    print(f"[vit_CARE_relu6_dim64] Loaded projected pretrained from: {checkpoint_path}")
    print(f"  load_state_dict strict=False -> missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
    if missing:
        print(f"  missing (first 20): {missing[:20]}")
    if unexpected:
        print(f"  unexpected (first 20): {unexpected[:20]}")
    return missing, unexpected


def _create_vision_transformer_dim64(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        embed_dim=STUDENT_EMBED_DIM,
        depth=STUDENT_DEPTH,
        num_heads=STUDENT_NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        qkv_bias=True,
    )
    model_kwargs.update(kwargs)
    model = VisionTransformer(**model_kwargs)

    if pretrained:
        if isinstance(pretrained, str) and pretrained.endswith(".pth"):
            load_vit_tiny_pretrained_dim64(model, pretrained)
        else:
            raise ValueError(
                "dim64 預訓練請傳入 ViT-Tiny .pth 路徑（例如 pretrained_models/vit_tiny_patch16_224.pth）"
            )
    return model


def vit_tiny64_care_patch16_224(pretrained=False, **kwargs):
    """
    CARE ReLU6，embed_dim=64、depth=12、num_heads=4。
    若 `pretrained` 為字串路徑，則從 ViT-Tiny (192) checkpoint 做截斷投影載入。
    """
    return _create_vision_transformer_dim64(pretrained=pretrained, **kwargs)
