"""Dump-only variant of CenterPredictor.

- 繼承 `CenterPredictor`，結構／state_dict 鍵名完全一致（所以可直接載入同一個
  checkpoint 的 `box_head.*` 權重）。
- 僅覆寫 `forward` / `get_score_map` 插入 `.npy` dump 點。
- 數值計算路徑與 `CenterPredictor` 完全相同（dump 僅旁路寫檔）。

啟用方式：
    head = CenterPredictorDump(...)
    head.dump_enabled = True
    head.dump_dir = '/path/to/out_dir'

搭配 `build_box_head` 的 dump 分支使用，由 `cfg.MODEL.HEAD.TYPE == 'CENTER_DUMP'`
觸發（見 head.py build_box_head）。
"""
from __future__ import annotations

import os

import numpy as np
import torch

from lib.models.layers.head import CenterPredictor
from lib.module import to_fixed_point


def _save_npy(enabled: bool, dump_dir: str, filename: str, tensor: torch.Tensor) -> None:
    if not enabled or not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    np.save(os.path.join(dump_dir, filename), tensor.detach().cpu().numpy())


class CenterPredictorDump(CenterPredictor):
    """CenterPredictor + RTL golden intermediate dump。"""

    def __init__(self, *args, int_bits: int = 8, frac_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        # 由外部指派（建議由 sglatrack build 之後設定）
        self.dump_enabled = False
        self.dump_dir = ""
        self.int_bits = int_bits
        self.frac_bits = frac_bits

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return to_fixed_point(x, self.int_bits, self.frac_bits)

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        bbox = torch.cat(
            [
                (idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                size.squeeze(-1),
            ],
            dim=1,
        )
        bbox = self._q(bbox)

        if return_score:
            return bbox, max_score
        return bbox

    # ------------------------------------------------------------------
    # forward：保留原本數值行為，插入 head_input / score_map / size_map /
    # offset_map / bbox dump。
    # ------------------------------------------------------------------
    def forward(self, x, gt_score_map=None):
        en = self.dump_enabled
        dd = self.dump_dir
        _save_npy(en, dd, "box_head_head_input.npy", x)

        score_map_ctr, size_map, offset_map = self.get_score_map(x)
        _save_npy(en, dd, "box_head_after_forward_head_score_map.npy", score_map_ctr)
        _save_npy(en, dd, "box_head_after_forward_head_size_map.npy", size_map)
        _save_npy(en, dd, "box_head_after_forward_head_offset_map.npy", offset_map)

        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        _save_npy(en, dd, "box_head_after_cal_bbox_bbox.npy", bbox)
        return score_map_ctr, bbox, size_map, offset_map

    # ------------------------------------------------------------------
    # get_score_map：插入每層 conv 的中間輸出 dump。數值路徑與父類一致。
    # ------------------------------------------------------------------
    def get_score_map(self, x):
        en = self.dump_enabled
        dd = self.dump_dir

        def _sigmoid_q(t):
            y = torch.clamp(t.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return self._q(y)

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        _save_npy(en, dd, "box_head_ctr_after_conv1_out.npy", x_ctr1)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        _save_npy(en, dd, "box_head_ctr_after_conv2_out.npy", x_ctr2)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        _save_npy(en, dd, "box_head_ctr_after_conv3_out.npy", x_ctr3)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        _save_npy(en, dd, "box_head_ctr_after_conv4_out.npy", x_ctr4)
        score_map_ctr = self._q(self.conv5_ctr(x_ctr4))
        _save_npy(en, dd, "box_head_ctr_after_conv5_out.npy", score_map_ctr)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        _save_npy(en, dd, "box_head_offset_after_conv1_out.npy", x_offset1)
        x_offset2 = self.conv2_offset(x_offset1)
        _save_npy(en, dd, "box_head_offset_after_conv2_out.npy", x_offset2)
        x_offset3 = self.conv3_offset(x_offset2)
        _save_npy(en, dd, "box_head_offset_after_conv3_out.npy", x_offset3)
        x_offset4 = self.conv4_offset(x_offset3)
        _save_npy(en, dd, "box_head_offset_after_conv4_out.npy", x_offset4)
        score_map_offset = self._q(self.conv5_offset(x_offset4))
        _save_npy(en, dd, "box_head_offset_after_conv5_out.npy", score_map_offset)

        # size branch
        x_size1 = self.conv1_size(x)
        _save_npy(en, dd, "box_head_size_after_conv1_out.npy", x_size1)
        x_size2 = self.conv2_size(x_size1)
        _save_npy(en, dd, "box_head_size_after_conv2_out.npy", x_size2)
        x_size3 = self.conv3_size(x_size2)
        _save_npy(en, dd, "box_head_size_after_conv3_out.npy", x_size3)
        x_size4 = self.conv4_size(x_size3)
        _save_npy(en, dd, "box_head_size_after_conv4_out.npy", x_size4)
        score_map_size = self._q(self.conv5_size(x_size4))
        _save_npy(en, dd, "box_head_size_after_conv5_out.npy", score_map_size)

        return _sigmoid_q(score_map_ctr), _sigmoid_q(score_map_size), score_map_offset
