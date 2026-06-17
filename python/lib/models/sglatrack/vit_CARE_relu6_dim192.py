"""ViT CARE ReLU6 變體：embed_dim=192（ViT-Tiny 尺度），浮點訓練 backbone。

與 ``vit_CARE_relu6.py`` 結構相同；工廠 ``vit_tiny192_care_patch16_224``。
預訓練：自 ViT-Tiny / augreg Tiny ``.pth`` 載入 shape 相符的權重（``strict=False``）。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from lib.models.sglatrack.vit_CARE_relu6 import VisionTransformer


STUDENT_EMBED_DIM = 192
STUDENT_NUM_HEADS = 3
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


def load_vit_tiny_pretrained_dim192(model: VisionTransformer, checkpoint_path: str) -> Tuple[list, list]:
    """從 ViT-Tiny (192 維) ``.pth`` 載入與 student 形狀一致的權重。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher_sd = _unwrap_checkpoint(checkpoint)
    student_sd = model.state_dict()
    projected: Dict[str, torch.Tensor] = {}
    for k, vs in student_sd.items():
        if k not in teacher_sd:
            continue
        vt = teacher_sd[k]
        if vt.shape == vs.shape:
            projected[k] = vt.clone()
    incomp = model.load_state_dict(projected, strict=False)
    missing = getattr(incomp, "missing_keys", incomp[0])
    unexpected = getattr(incomp, "unexpected_keys", incomp[1])
    print(f"[vit_CARE_relu6_dim192] Loaded pretrained from: {checkpoint_path}")
    print(f"  load_state_dict strict=False -> missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
    if missing:
        print(f"  missing (first 20): {missing[:20]}")
    if unexpected:
        print(f"  unexpected (first 20): {unexpected[:20]}")
    return missing, unexpected


def _create_vision_transformer_dim192(pretrained=False, **kwargs):
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
            load_vit_tiny_pretrained_dim192(model, pretrained)
        elif isinstance(pretrained, str) and (
            pretrained.endswith(".pth.tar") or "sglatrack" in pretrained
        ):
            pass
        elif pretrained:
            raise ValueError(
                "dim192 預訓練請傳入 ViT-Tiny .pth 或 sglatrack checkpoint 路徑"
            )
    return model


def vit_tiny192_care_patch16_224(pretrained=False, **kwargs):
    """CARE ReLU6 浮點，embed_dim=192、depth=12、num_heads=3。"""
    return _create_vision_transformer_dim192(pretrained=pretrained, **kwargs)
