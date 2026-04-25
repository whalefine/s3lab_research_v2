"""Export SGLATrack checkpoint parameters to .npy for RTL / Python check.

對齊 .cursor/rules/python-to-rtl-plan.mdc 的命名規範：
- 檔名風格為 `{module_path_flat}_{leaf_name}.npy`
- `module_path_flat` 由 Python module path 的 `.` 換成 `_` 組成
- 例：`backbone.blocks.0.attn.qkv.weight` -> `backbone_blocks_0_attn_qkv_weight.npy`

Scope：
- all         : 匯出所有 state_dict tensor（含 patch_embed、pos_embed，用於 Python 完整對照）
- post_embed  : 僅匯出 post-embedding 之後 RTL 真正會用到的權重（backbone blocks + 最後 norm + box_head）
- backbone    : 只 backbone 範圍
- head        : 只 box_head 範圍

若 scope 包含 head，會額外匯出 Conv2d+BatchNorm2d 的 folded new_gamma / new_beta
(`{prefix}_folded_weight.npy` / `_folded_bias.npy`)，供硬體直接使用。

備註（與 ``vit_CARE_relu6_fixed_hand.py`` 相容）：
- 當 backbone 採用 hand 版（或其 dump 變體）時，`Linear` / `LayerNorm` 實作位於
  ``lib.module``；它們不是 ``nn.Linear`` / ``nn.LayerNorm`` 的子類別。
- 本檔 `category_for_module` 會同時辨識原生 PyTorch 版與 ``lib.module`` 版，
  以保證參數被放到 ``linearParam`` / ``layerParam`` 正確子資料夾。
"""
import argparse
import datetime as _dt
import importlib
import json
import os
import sys
from pathlib import Path

prj_path = os.path.join(os.path.dirname(__file__), "..")
if prj_path not in sys.path:
    sys.path.append(prj_path)

import numpy as np
import torch
import torch.nn as nn

from lib.module import Linear as HandLinear
from lib.module import LayerNorm as HandLayerNorm


# ---------------------------------------------------------------------------
# 命名規範
# ---------------------------------------------------------------------------

def flatten_key(key: str) -> str:
    """Python module-path style key ('a.b.c.weight') -> 'a_b_c_weight' (檔名安全)."""
    return key.replace(".", "_").replace("/", "_")


def normalize_filename(key: str) -> str:
    return f"{flatten_key(key)}.npy"


# ---------------------------------------------------------------------------
# Scope 與排除規則
# ---------------------------------------------------------------------------

# post_embed scope：RTL 不需要的、只在 Python 端使用的 embedding 前段權重
POST_EMBED_EXCLUDE_PREFIXES = (
    "backbone.patch_embed.",
    "backbone.pos_embed",            # 含 pos_embed / pos_embed_z / pos_embed_x
    "backbone.cls_token",
    "backbone.dist_token",
    "backbone.MLP.",                 # ThreeLayerMLP，CARE adaptive skip 控制用；如 RTL 要納入再保留
)


def _block_index_from_key(key: str):
    prefix = "backbone.blocks."
    if not key.startswith(prefix):
        return None
    remain = key[len(prefix):]
    parts = remain.split(".", 1)
    if not parts or not parts[0].isdigit():
        return None
    return int(parts[0])


def key_in_scope(key: str, scope: str, selected_layer: int = None) -> bool:
    if scope == "all":
        return True
    if scope == "backbone":
        return key.startswith("backbone.")
    if scope == "head":
        return key.startswith("box_head.")
    if scope == "post_embed":
        if key.startswith("box_head."):
            return True
        if key.startswith("backbone."):
            for bad in POST_EMBED_EXCLUDE_PREFIXES:
                if key.startswith(bad):
                    return False
            if selected_layer is not None:
                block_idx = _block_index_from_key(key)
                if block_idx is not None:
                    # RTL 七層：保留 0~5 與 adaptive selector 選出的單一層
                    return block_idx <= 5 or block_idx == selected_layer
            return True
        return False
    return True


def category_for_module(module: nn.Module) -> str:
    if isinstance(module, nn.Conv2d):
        return "conv"
    if isinstance(module, (nn.Linear, HandLinear)):
        return "linear"
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return "batchnorm"
    if isinstance(module, (nn.LayerNorm, HandLayerNorm)):
        return "layernorm"
    return "other"


def build_key_category_map(model: nn.Module):
    key_to_category = {}
    key_to_module_name = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        category = category_for_module(module)
        for tensor_name, _ in module.named_parameters(recurse=False):
            key_to_category[prefix + tensor_name] = category
            key_to_module_name[prefix + tensor_name] = module_name
        for tensor_name, _ in module.named_buffers(recurse=False):
            key_to_category[prefix + tensor_name] = category
            key_to_module_name[prefix + tensor_name] = module_name
    return key_to_category, key_to_module_name


def ensure_dirs(base_dir: Path):
    subdirs = {
        "allParam": base_dir / "allParam",
        "linearParam": base_dir / "linearParam",
        "layerParam": base_dir / "layerParam",
        "convParam": base_dir / "convParam",
        "bnParam": base_dir / "bnParam",
        "foldedBN": base_dir / "foldedBN",
        "other": base_dir / "other",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def load_cfg(script_name: str, config_name: str):
    yaml_path = os.path.join(prj_path, "experiments", script_name, f"{config_name}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    config_module = importlib.import_module(f"lib.config.{script_name}.config")
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_path)
    return cfg, yaml_path


def load_model(script_name: str, cfg):
    model_module = importlib.import_module("lib.models")
    if script_name != "sglatrack":
        raise NotImplementedError(f"Unsupported script: {script_name}")
    return model_module.build_sglatrack(cfg, training=False)


def load_state_dict(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        return checkpoint["net"], "checkpoint['net']"
    if isinstance(checkpoint, dict):
        tensor_like = all(torch.is_tensor(v) for v in checkpoint.values())
        if tensor_like:
            return checkpoint, "checkpoint"
    raise ValueError(
        "Unsupported checkpoint format. Expected state_dict or trainer checkpoint with key 'net'."
    )


# ---------------------------------------------------------------------------
# Folded BN
# ---------------------------------------------------------------------------

def _find_conv_bn_pairs(model: nn.Module):
    """在 nn.Sequential(Conv2d, BN, ReLU) 結構下，找出 Conv2d + BatchNorm2d 配對。"""
    pairs = []
    for parent_name, parent in model.named_modules():
        if isinstance(parent, nn.Sequential):
            children = list(parent.named_children())
            for i in range(len(children) - 1):
                name_a, mod_a = children[i]
                name_b, mod_b = children[i + 1]
                if isinstance(mod_a, nn.Conv2d) and isinstance(mod_b, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    conv_full = f"{parent_name}.{name_a}" if parent_name else name_a
                    bn_full = f"{parent_name}.{name_b}" if parent_name else name_b
                    pairs.append((parent_name or "", conv_full, bn_full, mod_a, mod_b))
    return pairs


def _fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """回傳 (new_weight, new_bias) numpy arrays（等效 Conv2d 單層）。"""
    eps = bn.eps
    w = conv.weight.detach().cpu().double()
    b = conv.bias.detach().cpu().double() if conv.bias is not None else torch.zeros(conv.out_channels, dtype=torch.float64)
    gamma = bn.weight.detach().cpu().double()
    beta = bn.bias.detach().cpu().double()
    rm = bn.running_mean.detach().cpu().double()
    rv = bn.running_var.detach().cpu().double()
    scale = gamma / torch.sqrt(rv + eps)
    new_weight = w * scale.reshape(-1, 1, 1, 1)
    new_bias = (b - rm) * scale + beta
    return new_weight.float().numpy(), new_bias.float().numpy()


def export_folded_bn(model: nn.Module, scope: str, folded_dir: Path, manifest_entries: list, selected_layer: int = None) -> int:
    count = 0
    for parent_name, conv_name, bn_name, conv, bn in _find_conv_bn_pairs(model):
        key_conv_w = f"{conv_name}.weight"
        if not key_in_scope(key_conv_w, scope, selected_layer=selected_layer):
            continue
        new_w, new_b = _fold_bn_into_conv(conv, bn)
        w_filename = f"{flatten_key(parent_name)}_folded_weight.npy"
        b_filename = f"{flatten_key(parent_name)}_folded_bias.npy"
        np.save(folded_dir / w_filename, new_w)
        np.save(folded_dir / b_filename, new_b)
        manifest_entries.append({
            "kind": "folded_bn",
            "parent_module": parent_name,
            "conv_source": conv_name,
            "bn_source": bn_name,
            "filenames": [w_filename, b_filename],
            "weight_shape": list(new_w.shape),
            "bias_shape": list(new_b.shape),
            "dtype": str(new_w.dtype),
            "eps": float(bn.eps),
        })
        count += 1
    return count


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a SGLATrack checkpoint (.pth/.pth.tar) into per-tensor .npy files."
    )
    parser.add_argument("--script", type=str, default="sglatrack", choices=["sglatrack"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--scope", type=str, default="post_embed",
        choices=["all", "backbone", "head", "post_embed"],
        help="post_embed: 只保留 RTL 會用到的參數（排除 patch_embed / pos_embed / cls_token / MLP）",
    )
    parser.add_argument(
        "--no-folded-bn", action="store_true",
        help="停用 Conv+BN 融合匯出（預設會匯出 folded_weight / folded_bias）",
    )
    parser.add_argument(
        "--selected-layer", type=int, default=None,
        help="僅在 --scope post_embed 生效。指定 adaptive selector 選出的 layer index（6~11），"
             "只匯出 blocks.0~5 與 blocks.<selected-layer> 的 backbone 參數。",
    )
    return parser.parse_args()


def _category_dir(category: str, dirs: dict) -> Path:
    if category == "linear":
        return dirs["linearParam"]
    if category == "layernorm":
        return dirs["layerParam"]
    if category == "conv":
        return dirs["convParam"]
    if category == "batchnorm":
        return dirs["bnParam"]
    return dirs["other"]


def main():
    args = parse_args()
    if args.selected_layer is not None and args.scope != "post_embed":
        raise ValueError("--selected-layer 僅支援搭配 --scope post_embed")
    if args.selected_layer is not None and not (6 <= args.selected_layer <= 11):
        raise ValueError("--selected-layer 必須在 [6, 11]")

    cfg, yaml_path = load_cfg(args.script, args.config)
    model = load_model(args.script, cfg)
    checkpoint_state_dict, state_source = load_state_dict(args.checkpoint)

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)
    # 由 model 的 state_dict 出發（包含 buffer 如 running_mean），以保證 folded BN 有完整資料
    full_state = model.state_dict()

    key_to_category, key_to_module_name = build_key_category_map(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(output_dir)

    manifest = {
        "script": args.script,
        "config": args.config,
        "config_path": yaml_path,
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "state_source": state_source,
        "scope": args.scope,
        "selected_layer": args.selected_layer,
        "export_time": _dt.datetime.now().isoformat(timespec="seconds"),
        "naming_rule": "module_path_flat + leaf, '.'->'_'; see python-to-rtl-plan.mdc",
        "default_q_format": "Q8.8 (signed, 1 sign + 8 int + 8 frac)",
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "tensors": [],
        "folded_bn": [],
    }

    exported = 0
    for key, tensor in full_state.items():
        if not torch.is_tensor(tensor):
            continue
        if not key_in_scope(key, args.scope, selected_layer=args.selected_layer):
            continue

        category = key_to_category.get(key, "other")
        module_name = key_to_module_name.get(key, "")
        array = tensor.detach().cpu().numpy()
        filename = normalize_filename(key)

        np.save(dirs["allParam"] / filename, array)
        np.save(_category_dir(category, dirs) / filename, array)

        manifest["tensors"].append({
            "key": key,
            "module": module_name,
            "category": category,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "filename": filename,
        })
        exported += 1

    folded_count = 0
    if not args.no_folded_bn:
        folded_count = export_folded_bn(
            model,
            args.scope,
            dirs["foldedBN"],
            manifest["folded_bn"],
            selected_layer=args.selected_layer,
        )

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Config      : {yaml_path}")
    print(f"Checkpoint  : {os.path.abspath(args.checkpoint)}")
    print(f"State source: {state_source}")
    print(f"Scope       : {args.scope}")
    if args.selected_layer is not None:
        print(f"Selected layer (adaptive): {args.selected_layer}")
    print(f"Exported tensors: {exported}")
    print(f"Folded BN pairs : {folded_count}")
    print(f"Output dir  : {output_dir}")
    print(f"Manifest    : {manifest_path}")
    if missing_keys:
        print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:10]}")


if __name__ == "__main__":
    main()
