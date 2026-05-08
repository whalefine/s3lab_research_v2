"""Shared-trunk CenterPredictor：兩層 3x3（32→96→48）+ 三分支 1x1 tail（浮點，無 Q8.8）。

與 `head.CenterPredictor` 數值路徑一致：ctr/size 經 sigmoid 並 clamp；offset 純 conv 輸出；bbox 無額外量化。
類別名保留 CenterPredictorFixedSharedTrunk 以相容 yaml `CENTER_FIXED_SHARED_TRUNK`。

`channel`（舊 yaml 的 NUM_CHANNELS）保留為相容參數，此結構不使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lib.models.layers.head import conv

SHARED_CH1 = 96
SHARED_CH2 = 48


class CenterPredictorFixedSharedTrunk(nn.Module):
    """Shared trunk + per-branch 1x1 tails，與 CenterPredictor 相同 forward 介面。"""

    def __init__(
        self,
        inplanes=64,
        channel=256,
        feat_sz=20,
        stride=16,
        freeze_bn=False,
        int_bits: int = 8,
        frac_bits: int = 8,
    ):
        super().__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        # int_bits / frac_bits 保留簽名相容，訓練已不使用定點
        self.shared_conv1 = conv(
            inplanes, SHARED_CH1, kernel_size=3, stride=1, padding=1, freeze_bn=freeze_bn
        )
        self.shared_conv2 = conv(
            SHARED_CH1, SHARED_CH2, kernel_size=3, stride=1, padding=1, freeze_bn=freeze_bn
        )

        self.tail_ctr = nn.Conv2d(SHARED_CH2, 1, kernel_size=1)
        self.tail_size = nn.Conv2d(SHARED_CH2, 2, kernel_size=1)
        self.tail_offset = nn.Conv2d(SHARED_CH2, 2, kernel_size=1)

        for m in (self.tail_ctr, self.tail_size, self.tail_offset):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        score_map_ctr, size_map, offset_map = self.get_score_map(x)
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)
        return score_map_ctr, bbox, size_map, offset_map

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

        if return_score:
            return bbox, max_score
        return bbox

    def get_score_map(self, x):
        def _sigmoid(x):
            return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

        trunk = self.shared_conv2(self.shared_conv1(x))

        score_map_ctr = self.tail_ctr(trunk)
        score_map_size = self.tail_size(trunk)
        score_map_offset = self.tail_offset(trunk)

        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset
