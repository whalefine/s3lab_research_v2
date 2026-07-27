"""ViT CARE ReLU6 變體：embed_dim=256，可由 ViT 寬模型 checkpoint 做左上角截斷投影載入。

支援 teacher embed_dim=192（ViT-Tiny）或 768（MAE ViT-Base / vit_base_patch16_224）。
與 vit_CARE_relu6.py 結構相同，僅預設維度與 `vit_care_dim256_patch16_224` 工廠函數不同。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from lib.models.sglatrack.vit_CARE_relu6 import VisionTransformer


# ---------------------------------------------------------------------------
# 維度常數
# ---------------------------------------------------------------------------
VIT_TINY_EMBED_DIM = 192
VIT_BASE_EMBED_DIM = 768
TEACHER_EMBED_DIM = VIT_TINY_EMBED_DIM  # 相容舊名
STUDENT_EMBED_DIM = 256
STUDENT_NUM_HEADS = 8  # 256 % 8 == 0
STUDENT_DEPTH = 12
MLP_RATIO = 4


def _unwrap_checkpoint(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """支援裸 state_dict、{'state_dict':...}、{'model':...}、{'net':...}。"""
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            sd = checkpoint["model"]
        elif "net" in checkpoint and isinstance(checkpoint["net"], dict):
            sd = checkpoint["net"]
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


def _normalize_backbone_keys(teacher_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """去掉 backbone. 前綴，使 SGLATrack ckpt 的 backbone 權重可對齊 ViT backbone 鍵名。"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in teacher_sd.items():
        key = k[len("backbone.") :] if k.startswith("backbone.") else k
        out[key] = v
    return out


def _infer_teacher_embed_dim(teacher_sd: Dict[str, torch.Tensor]) -> int:
    sd = _normalize_backbone_keys(teacher_sd)
    w = sd.get("patch_embed.proj.weight")
    if w is None:
        raise KeyError(
            "Cannot infer teacher embed_dim: missing patch_embed.proj.weight "
            "(expected ViT backbone keys, optionally with backbone. prefix)."
        )
    return int(w.shape[0])


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
    sh, sw = 4 * s_dim, s_dim
    return w_t[:sh, :sw].clone(), b_t[:sh].clone()


def _project_mlp_fc2(w_t: torch.Tensor, b_t: torch.Tensor, t_dim: int, s_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """fc2: [t_dim, 4*t_dim], bias [t_dim]"""
    return w_t[:s_dim, : 4 * s_dim].clone(), b_t[:s_dim].clone()


def project_vit_state_dict_to_dim256(
    teacher_sd: Dict[str, torch.Tensor],
    model: VisionTransformer,
    teacher_dim: int | None = None,
    student_dim: int = STUDENT_EMBED_DIM,
) -> Dict[str, torch.Tensor]:
    """
    將 ViT 寬模型 (192 / 768 等) state_dict 投影到 embed_dim=256 的 student。
    僅覆寫可截斷對齊的參數；其餘鍵不放入 out，由 load_state_dict(strict=False) 保留隨機初始化。
    """
    teacher_sd = _normalize_backbone_keys(teacher_sd)
    s = student_dim
    t = int(teacher_dim) if teacher_dim is not None else _infer_teacher_embed_dim(teacher_sd)
    if t < s:
        raise ValueError(f"teacher_dim ({t}) must be >= student_dim ({s})")

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


def project_vit_tiny_state_dict_to_dim256(
    teacher_sd: Dict[str, torch.Tensor],
    model: VisionTransformer,
    teacher_dim: int = VIT_TINY_EMBED_DIM,
    student_dim: int = STUDENT_EMBED_DIM,
) -> Dict[str, torch.Tensor]:
    """相容舊名：ViT-Tiny (192) → dim256。"""
    return project_vit_state_dict_to_dim256(
        teacher_sd, model, teacher_dim=teacher_dim, student_dim=student_dim
    )


def load_projected_pretrained_dim256(model: VisionTransformer, checkpoint_path: str) -> Tuple[list, list]:
    """從 .pth / .pth.tar 讀取 ViT 寬模型權重，自動推斷維度並投影後寫入 256 維 student。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher_sd = _unwrap_checkpoint(checkpoint)
    teacher_dim = _infer_teacher_embed_dim(_normalize_backbone_keys(teacher_sd))
    projected = project_vit_state_dict_to_dim256(teacher_sd, model, teacher_dim=teacher_dim)
    incomp = model.load_state_dict(projected, strict=False)
    missing = getattr(incomp, "missing_keys", incomp[0])
    unexpected = getattr(incomp, "unexpected_keys", incomp[1])
    print(
        f"[vit_CARE_relu6_dim256] Loaded projected pretrained from: {checkpoint_path} "
        f"(teacher_dim={teacher_dim} -> student_dim={STUDENT_EMBED_DIM})"
    )
    print(f"  load_state_dict strict=False -> missing: {len(missing)} keys, unexpected: {len(unexpected)} keys")
    if missing:
        print(f"  missing (first 20): {missing[:20]}")
    if unexpected:
        print(f"  unexpected (first 20): {unexpected[:20]}")
    return missing, unexpected


def load_vit_tiny_pretrained_dim256(model: VisionTransformer, checkpoint_path: str) -> Tuple[list, list]:
    """相容舊名：等同 load_projected_pretrained_dim256。"""
    return load_projected_pretrained_dim256(model, checkpoint_path)


def _create_vision_transformer_dim256(pretrained=False, **kwargs):
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
        if isinstance(pretrained, str) and (pretrained.endswith(".pth") or pretrained.endswith(".pth.tar")):
            load_projected_pretrained_dim256(model, pretrained)
        else:
            raise ValueError(
                "dim256 預訓練請傳入 .pth / .pth.tar 路徑"
                "（例如 pretrained_models/vit_tiny_patch16_224.pth 或 mae_pretrain_vit_base.pth）"
            )
    return model


def vit_care_dim256_patch16_224(pretrained=False, **kwargs):
    """
    CARE ReLU6，embed_dim=256、depth=12、num_heads=8。
    若 `pretrained` 為字串路徑，則從 ViT 寬模型 (192 / 768 等) checkpoint 做截斷投影載入。
    """
    return _create_vision_transformer_dim256(pretrained=pretrained, **kwargs)
