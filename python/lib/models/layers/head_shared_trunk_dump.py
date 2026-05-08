"""Dump-only variant for shared-trunk CenterPredictor（對齊 ``head_fixed_shared_trunk`` 結構）。

繼承 ``CenterPredictorFixedSharedTrunk``；插入 ``shared_conv1/2``、三分支 tail 與最終 map 的 .npy dump。
Tail 輸出與 bbox 路徑對齊 ``head_dump``：使用 ``to_fixed_point``（預設 Q8.8）。

搭配 ``cfg.MODEL.HEAD.TYPE == 'CENTER_DUMP_SHARED_TRUNK'``（見 ``head.py`` build_box_head）。
"""

from __future__ import annotations

import os

import numpy as np
import torch

from lib.models.layers.head_fixed_shared_trunk import CenterPredictorFixedSharedTrunk
from lib.module import to_fixed_point


def _save_npy(enabled: bool, dump_dir: str, filename: str, tensor: torch.Tensor) -> None:
    if not enabled or not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    np.save(os.path.join(dump_dir, filename), tensor.detach().cpu().numpy())


class CenterPredictorSharedTrunkDump(CenterPredictorFixedSharedTrunk):
    """Shared-trunk head + RTL golden intermediate dump（含 Q8.8 量化節點）。"""

    def __init__(self, *args, int_bits: int = 8, frac_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
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

    def get_score_map(self, x):
        en = self.dump_enabled
        dd = self.dump_dir

        def _sigmoid_q(t: torch.Tensor) -> torch.Tensor:
            y = torch.clamp(t.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return self._q(y)

        x1 = self.shared_conv1(x)
        _save_npy(en, dd, "box_head_shared_after_conv1_out.npy", x1)
        x2 = self.shared_conv2(x1)
        _save_npy(en, dd, "box_head_shared_after_conv2_out.npy", x2)

        raw_ctr = self.tail_ctr(x2)
        raw_size = self.tail_size(x2)
        raw_off = self.tail_offset(x2)

        q_ctr = self._q(raw_ctr)
        q_size = self._q(raw_size)
        q_off = self._q(raw_off)
        _save_npy(en, dd, "box_head_tail_ctr_after_conv_out.npy", q_ctr)
        _save_npy(en, dd, "box_head_tail_size_after_conv_out.npy", q_size)
        _save_npy(en, dd, "box_head_tail_offset_after_conv_out.npy", q_off)

        score_map_ctr = _sigmoid_q(q_ctr.clone())
        score_map_size = _sigmoid_q(q_size.clone())
        score_map_offset = q_off

        _save_npy(en, dd, "box_head_tail_ctr_after_sigmoid_out.npy", score_map_ctr)
        _save_npy(en, dd, "box_head_tail_size_after_sigmoid_out.npy", score_map_size)
        _save_npy(en, dd, "box_head_tail_offset_final_out.npy", score_map_offset)

        return score_map_ctr, score_map_size, score_map_offset
