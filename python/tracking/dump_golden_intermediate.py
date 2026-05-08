"""Dump SGLATrack test-time golden intermediate tensors for Python / RTL check.

Pipeline（對齊 .cursor/rules/python-to-rtl-plan.mdc）：
- frame1 (+ init_bbox) -> template crop -> preprocess -> patch_embed + pos_embed_z -> template_post_embed
- frame2 -> search crop -> preprocess -> patch_embed + pos_embed_x -> search_post_embed
- 呼叫 backbone.forward_test_from_post_embed(template_post_embed, search_post_embed)
  內部會 dump（來自 vit_CARE_relu6_fixed_dump.py，該檔現以 vit_CARE_relu6_fixed_hand.py 為基底）：
    template_post_embed_input, search_post_embed_input,
    merged_tokens, after_pos_drop_out,
    每個 block 的 norm1 / qkv(q,k,v) / attn / residual1 / norm2 / mlp / block_out,
    backbone_after_recover_tokens_out, backbone_after_norm_backbone_out,
    adaptive_pro, adaptive_sorted_topk_indices.
- 另外再跑 tracker 後處理（Hann window + cal_bbox + mean + map_box_back + clip_box），
  dump tracker_after_output_window_response, tracker_after_cal_bbox_bbox,
  tracker_after_map_box_back_bbox, tracker_after_final_bbox_bbox，並於終端列印最終
  frame2 像素 xywh（格式對齊 ``run_backbone_numpy.py``）。

最後輸出 golden_manifest.json 紀錄每個檔案對應的 stage / shape / dtype，
並包含 adaptive selector 選出的 block index（RTL 僅需跑第 0~5 層 + 此 selected 層）。

注意：
- 本腳本預設使用 `vit_coco_uav123_care_relu6_fixed_dump.yaml`，其 backbone 是
  `vit_CARE_relu6_fixed_dump.py` 中的 `VisionTransformerDump`。
- `vit_CARE_relu6_fixed_dump.py` 現已對齊 `vit_CARE_relu6_fixed_hand.py`：
  Linear / LayerNorm / Dropout / DropPath / Mlp / ReLU6 全部改用 `lib.module`
  的手刻版本，父類別改為 `BaseBackboneFix`，並包含 `qk_mean` / `qk_mean_eps` 的
  pre-reciprocal 量化節點。所有 dump 邏輯都在該 dump 檔案裡；
  `vit_CARE_relu6_fixed_hand.py` 本身完全純淨、可正常 train/test。
- Head 使用 `CENTER_DUMP` 或 `CENTER_DUMP_SHARED_TRUNK` 時會自動輸出 head conv / map / bbox dump；
  本腳本另外補存 `box_head_after_forward_head_pred_boxes.npy` 與 manifest 欄位。

允許的 dump backbone：`vit_care_relu6_fixed_dump_base_patch16_224`（768 維）、
`vit_care_relu6_dim32_fixed_dump_base_patch16_224`（32 維）、
`vit_care_relu6_dim32_fixed_shared_trunk_dump_base_patch16_224`（32 維，與上者同一 backbone，供 shared-trunk head dump yaml）。
"""

from __future__ import annotations

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

from lib.module import to_fixed_point
from lib.test.tracker.data_utils import Preprocessor
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box

# 須為實作 forward_test_from_post_embed 且與下方流程相容的 *_fixed_dump 系 backbone。
_DUMP_ALLOWED_BACKBONE_TYPES = (
    "vit_care_relu6_fixed_dump_base_patch16_224",
    "vit_care_relu6_dim32_fixed_dump_base_patch16_224",
    "vit_care_relu6_dim32_fixed_shared_trunk_dump_base_patch16_224",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump SGLATrack test-time golden intermediate tensors (frame1->template, frame2->search)."
    )
    parser.add_argument("--script", type=str, default="sglatrack", choices=["sglatrack"])
    parser.add_argument(
        "--config", type=str,
        default="vit_coco_uav123_care_relu6_fixed_dump",
        help="config name under experiments/<script>/ (預設：dump-only 變體)",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--frame1", type=str, required=True, help="第一幀影像路徑 (BGR/RGB 皆可，沿用 OpenCV)")
    parser.add_argument("--frame2", type=str, required=True, help="第二幀影像路徑")
    parser.add_argument(
        "--init-bbox", type=float, nargs=4, required=True,
        help="frame1 上的 init bbox，格式 x y w h（image pixel）",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--dump-tracker-post", action="store_true", default=True,
        help="同時 dump tracker 後處理（hann window / cal_bbox / map_box_back / clip_box）",
    )
    parser.add_argument("--no-dump-tracker-post", dest="dump_tracker_post", action="store_false")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
    )
    return parser.parse_args()


def _load_cfg(script_name: str, config_name: str):
    yaml_path = os.path.join(prj_path, "experiments", script_name, f"{config_name}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    config_module = importlib.import_module(f"lib.config.{script_name}.config")
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_path)
    return cfg, yaml_path


def _load_model(script_name: str, cfg, checkpoint_path: str, device: str):
    model_module = importlib.import_module("lib.models")
    model = model_module.build_sglatrack(cfg, training=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if device == "cuda":
        model = model.cuda()
    model.eval()
    return model, missing, unexpected


def _map_box_back(state_xywh, pred_box_cxcywh, resize_factor, search_size):
    cx_prev = state_xywh[0] + 0.5 * state_xywh[2]
    cy_prev = state_xywh[1] + 0.5 * state_xywh[3]
    cx, cy, w, h = pred_box_cxcywh
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def _save_npy(out_dir: Path, filename: str, tensor: torch.Tensor, manifest: list,
              stage: str, source: str):
    arr = tensor.detach().cpu().numpy()
    np.save(out_dir / filename, arr)
    manifest.append({
        "filename": filename,
        "stage": stage,
        "source_module": source,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    })


def main():
    args = parse_args()
    import cv2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, yaml_path = _load_cfg(args.script, args.config)

    backbone_type = cfg.MODEL.BACKBONE.TYPE
    if backbone_type not in _DUMP_ALLOWED_BACKBONE_TYPES:
        raise RuntimeError(
            "dump_golden_intermediate.py 需配合 *_fixed_dump 系 backbone，"
            f"目前 cfg.MODEL.BACKBONE.TYPE = {backbone_type}。"
            f"請使用下列之一於 yaml：{list(_DUMP_ALLOWED_BACKBONE_TYPES)} "
            "(例：`--config vit_coco_uav123_care_relu6_fixed_dump` 或 "
            "`vit_coco_uav123_care_relu6_dim32_fixed_dump`)。"
        )

    model, missing, unexpected = _load_model(args.script, cfg, args.checkpoint, args.device)

    # 開啟 backbone / box_head 自動 dump（dump-only 變體）
    model.backbone.dump_enabled = True
    model.backbone.dump_dir = str(output_dir.resolve())
    if hasattr(model, "box_head"):
        model.box_head.dump_enabled = True
        model.box_head.dump_dir = str(output_dir.resolve())

    # --- frame1: template -------------------------------------------------
    frame1 = cv2.imread(args.frame1)
    if frame1 is None:
        raise FileNotFoundError(f"Cannot read frame1: {args.frame1}")
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    init_bbox = list(args.init_bbox)

    template_factor = cfg.TEST.TEMPLATE_FACTOR
    template_size = cfg.TEST.TEMPLATE_SIZE
    search_factor = cfg.TEST.SEARCH_FACTOR
    search_size = cfg.TEST.SEARCH_SIZE
    feat_sz = search_size // cfg.MODEL.BACKBONE.STRIDE

    z_patch_arr, z_resize_factor, z_amask_arr = sample_target(
        frame1, init_bbox, template_factor, output_sz=template_size,
    )
    preproc = Preprocessor() if args.device == "cuda" else None
    if args.device == "cuda":
        z_nested = preproc.process(z_patch_arr, z_amask_arr)
        template_tensor = z_nested.tensors
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = torch.from_numpy(z_patch_arr).float().permute(2, 0, 1).unsqueeze(0)
        template_tensor = ((t / 255.0) - mean) / std

    # --- frame2: search ---------------------------------------------------
    frame2 = cv2.imread(args.frame2)
    if frame2 is None:
        raise FileNotFoundError(f"Cannot read frame2: {args.frame2}")
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    H, W = frame2.shape[:2]

    # 沿用第一幀的 state 當作第二幀 search crop 的中心
    state = list(init_bbox)
    x_patch_arr, x_resize_factor, x_amask_arr = sample_target(
        frame2, state, search_factor, output_sz=search_size,
    )
    if args.device == "cuda":
        x_nested = preproc.process(x_patch_arr, x_amask_arr)
        search_tensor = x_nested.tensors
    else:
        t = torch.from_numpy(x_patch_arr).float().permute(2, 0, 1).unsqueeze(0)
        search_tensor = ((t / 255.0) - mean) / std

    # --- patch_embed + pos_embed (Python-only 前處理) --------------------
    # 這段同步 dump 以便跨檢查，但 RTL 的正式輸入是 template_post_embed / search_post_embed。
    # 與 vit_CARE_relu6_fixed_hand.py / BaseBackboneFix._forward_impl 一致，
    # patch_embed、pos_embed 以及相加結果都要做 Q8.8 截斷，確保 bit-accurate。
    with torch.no_grad():
        z_patch = model.backbone.patch_embed(template_tensor)
        x_patch = model.backbone.patch_embed(search_tensor)
        z_patch = to_fixed_point(z_patch, 8, 8)
        x_patch = to_fixed_point(x_patch, 8, 8)
        z_pos = to_fixed_point(model.backbone.pos_embed_z, 8, 8)
        x_pos = to_fixed_point(model.backbone.pos_embed_x, 8, 8)
        template_post_embed = to_fixed_point(z_patch + z_pos, 8, 8)
        search_post_embed = to_fixed_point(x_patch + x_pos, 8, 8)

    manifest_entries = []
    _save_npy(output_dir, "template_after_patch_embed_out.npy", z_patch, manifest_entries,
              stage="pre_rtl.patch_embed", source="backbone.patch_embed(template)")
    _save_npy(output_dir, "search_after_patch_embed_out.npy", x_patch, manifest_entries,
              stage="pre_rtl.patch_embed", source="backbone.patch_embed(search)")
    _save_npy(output_dir, "template_pos_embed.npy", z_pos, manifest_entries,
              stage="pre_rtl.pos_embed", source="backbone.pos_embed_z")
    _save_npy(output_dir, "search_pos_embed.npy", x_pos, manifest_entries,
              stage="pre_rtl.pos_embed", source="backbone.pos_embed_x")
    _save_npy(output_dir, "template_after_pos_add_out.npy", template_post_embed, manifest_entries,
              stage="pre_rtl.pos_add", source="z_patch + pos_embed_z")
    _save_npy(output_dir, "search_after_pos_add_out.npy", search_post_embed, manifest_entries,
              stage="pre_rtl.pos_add", source="x_patch + pos_embed_x")

    # --- 正式 RTL 入口：兩路 post-embedding 進入 backbone --------------
    with torch.no_grad():
        feat, aux_dict = model.backbone.forward_test_from_post_embed(
            template_post_embed=template_post_embed,
            search_post_embed=search_post_embed,
        )
        # 走 head（CENTER_DUMP 會在 head 內部自動 dump）
        out = model.forward_head(feat, None)
    pred_score_map = out["score_map"]
    pred_size_map = out["size_map"]
    pred_offset_map = out["offset_map"]
    pred_boxes_head = out["pred_boxes"]
    selected_layer_indices = aux_dict.get("selected_layer_indices", None)
    selected_layer_index = None
    if selected_layer_indices is not None:
        selected_layer_index = int(selected_layer_indices.reshape(-1)[0].item())

    # head input / score_map / size_map / offset_map / cal_bbox_bbox 由
    # CenterPredictorDump 內部 dump（透過 HEAD.TYPE=CENTER_DUMP 啟用）。
    # 這裡僅補存 sglatrack.forward_head 在 reshape 後的 pred_boxes。
    # 依 Q8.8 對拍規則，存檔前先做固定點截斷。
    pred_boxes_head = to_fixed_point(pred_boxes_head, 8, 8)
    _save_npy(output_dir, "box_head_after_forward_head_pred_boxes.npy",
              pred_boxes_head, manifest_entries,
              stage="head.forward_head", source="sglatrack.forward_head.pred_boxes")

    # --- tracker 後處理（可選） ------------------------------------------
    if args.dump_tracker_post:
        device = pred_score_map.device
        output_window = hann2d(torch.tensor([feat_sz, feat_sz]).long(), centered=True).to(device)
        response = output_window * pred_score_map
        response = to_fixed_point(response, 8, 8)
        _save_npy(output_dir, "tracker_after_output_window_response.npy", response,
                  manifest_entries, stage="tracker.post", source="hann2d * score_map")
        bbox_after = model.box_head.cal_bbox(response, pred_size_map, pred_offset_map)
        bbox_after = to_fixed_point(bbox_after, 8, 8)
        _save_npy(output_dir, "tracker_after_cal_bbox_bbox.npy", bbox_after,
                  manifest_entries, stage="tracker.post",
                  source="box_head.cal_bbox(response, size_map, offset_map)")
        pred_boxes = bbox_after.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * search_size / x_resize_factor).tolist()
        mapped = _map_box_back(state, pred_box, x_resize_factor, search_size)
        mapped_t = torch.tensor(mapped)
        mapped_t = to_fixed_point(mapped_t, 8, 8)
        _save_npy(output_dir, "tracker_after_map_box_back_bbox.npy", mapped_t,
                  manifest_entries, stage="tracker.post", source="tracker.map_box_back")
        final_bbox = clip_box(mapped, H, W, margin=10)
        final_t = torch.tensor(final_bbox)
        final_t = to_fixed_point(final_t, 8, 8)
        _save_npy(output_dir, "tracker_after_final_bbox_bbox.npy", final_t,
                  manifest_entries, stage="tracker.post", source="clip_box")
        x1, y1, bw, bh = final_bbox
        print(
            f"[tracker] 最終 bbox（frame2 像素 xywh）: "
            f"x1={float(x1):.4f}, y1={float(y1):.4f}, w={float(bw):.4f}, h={float(bh):.4f}"
        )
    else:
        print("[tracker] 最終 bbox: 未計算（已使用 --no-dump-tracker-post）")

    # --- 蒐集所有已 dump 的檔案（含 backbone 自動 dump 的） ------------
    all_files = sorted(f.name for f in output_dir.iterdir()
                       if f.is_file() and f.suffix == ".npy")

    manifest = {
        "script": args.script,
        "config": args.config,
        "config_path": yaml_path,
        "backbone_type": backbone_type,
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "frame1": os.path.abspath(args.frame1),
        "frame2": os.path.abspath(args.frame2),
        "init_bbox_xywh": init_bbox,
        "template_crop_resize_factor": float(z_resize_factor),
        "search_crop_resize_factor": float(x_resize_factor),
        "template_size": template_size,
        "search_size": search_size,
        "feat_sz": feat_sz,
        "export_time": _dt.datetime.now().isoformat(timespec="seconds"),
        "naming_rule": "{module_path_flat}_after_{function_name}_{signal_name}.npy",
        "default_q_format": "Q8.8",
        "device": args.device,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "adaptive_selected_layer_index": selected_layer_index,
        "rtl_expected_active_backbone_blocks": list(range(0, 6)) + ([selected_layer_index] if selected_layer_index is not None else []),
        "explicit_entries": manifest_entries,
        "all_npy_files": all_files,
    }

    manifest_path = output_dir / "golden_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Output dir : {output_dir}")
    print(f"Manifest   : {manifest_path}")
    print(f"Dumped .npy files : {len(all_files)}")
    print(f"template_post_embed shape : {tuple(template_post_embed.shape)}")
    print(f"search_post_embed   shape : {tuple(search_post_embed.shape)}")


if __name__ == "__main__":
    main()
