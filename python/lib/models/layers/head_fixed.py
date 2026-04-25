"""Fixed-point variant of CenterPredictor (minimal Q8.8 insertion).

只在硬體上「MAC 結尾 / 非線性結尾 / 介面」這幾個真正需要截斷的節點插 to_fixed_point：
    - 每個 branch 的 conv5 輸出（ctr / size / offset）
    - score_map_ctr / score_map_size 的 sigmoid 輸出
    - cal_bbox 最後輸出的 bbox

中間 conv1~conv4 不再逐層量化（留給 RTL accumulator 的 guard bits 即可，
與 `reference/rongxuan/02_FixedPoint` 的做法一致）。
輸入 `x` 來自 backbone self.norm 的輸出，已於 backbone 尾端量化，
因此此處不再重複量化。
"""

from __future__ import annotations

import torch

from lib.models.layers.head import CenterPredictor
from lib.module import to_fixed_point


class CenterPredictorFixed(CenterPredictor):
    """CenterPredictor with minimal fixed-point quantization in forward path."""

    def __init__(self, *args, int_bits: int = 8, frac_bits: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.int_bits = int_bits
        self.frac_bits = frac_bits

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        return to_fixed_point(x, self.int_bits, self.frac_bits)

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        # score_map_ctr / size_map / offset_map 皆已於 get_score_map 內量化。
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

    def get_score_map(self, x):
        def _sigmoid_q(t):
            y = torch.clamp(t.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return self._q(y)

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self._q(self.conv5_ctr(x_ctr4))

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self._q(self.conv5_offset(x_offset4))

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self._q(self.conv5_size(x_size4))

        return _sigmoid_q(score_map_ctr), _sigmoid_q(score_map_size), score_map_offset
