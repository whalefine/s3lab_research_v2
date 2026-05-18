import math
import os
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.layers.head_hand import build_box_head as build_box_head_hand
from lib.models.sglatrack.vit import vit_base_patch16_224
from lib.models.sglatrack.vit_square import vit_base_patch16_224 as vit_square_base_patch16_224
from lib.models.sglatrack.vit_sima import vit_base_patch16_224 as vit_sima_base_patch16_224
from lib.models.sglatrack.vit_CARE import vit_base_patch16_224 as vit_care_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu import vit_base_patch16_224 as vit_care_relu_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu_fixed import vit_base_patch16_224 as vit_care_relu_fixed_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6 import vit_base_patch16_224 as vit_care_relu6_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_dim32 import (
    vit_tiny32_care_patch16_224 as vit_care_relu6_dim32_base_patch16_224,
)
from lib.models.sglatrack.vit_CARE_relu6_fixed_shared_trunk_dim32 import (
    vit_care_relu6_shared_trunk_dim32_base_patch16_224,
)
from lib.models.sglatrack.vit_CARE_relu6_dim64 import (
    vit_tiny64_care_patch16_224 as vit_care_relu6_dim64_base_patch16_224,
)
from lib.models.sglatrack.vit_CARE_relu6_hand import vit_base_patch16_224 as vit_care_relu6_hand_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed import vit_base_patch16_224 as vit_care_relu6_fixed_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q48 import vit_base_patch16_224 as vit_care_relu6_fixed_q48_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q58 import vit_base_patch16_224 as vit_care_relu6_fixed_q58_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q64 import vit_base_patch16_224 as vit_care_relu6_fixed_q64_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q65 import vit_base_patch16_224 as vit_care_relu6_fixed_q65_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q66 import vit_base_patch16_224 as vit_care_relu6_fixed_q66_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q67 import vit_base_patch16_224 as vit_care_relu6_fixed_q67_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q68 import vit_base_patch16_224 as vit_care_relu6_fixed_q68_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q69 import vit_base_patch16_224 as vit_care_relu6_fixed_q69_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q610 import vit_base_patch16_224 as vit_care_relu6_fixed_q610_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q611 import vit_base_patch16_224 as vit_care_relu6_fixed_q611_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q612 import vit_base_patch16_224 as vit_care_relu6_fixed_q612_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q78 import vit_base_patch16_224 as vit_care_relu6_fixed_q78_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q84 import vit_base_patch16_224 as vit_care_relu6_fixed_q84_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q85 import vit_base_patch16_224 as vit_care_relu6_fixed_q85_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q86 import vit_base_patch16_224 as vit_care_relu6_fixed_q86_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q87 import vit_base_patch16_224 as vit_care_relu6_fixed_q87_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q88 import vit_base_patch16_224 as vit_care_relu6_fixed_q88_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q89 import vit_base_patch16_224 as vit_care_relu6_fixed_q89_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q98 import vit_base_patch16_224 as vit_care_relu6_fixed_q98_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q108 import vit_base_patch16_224 as vit_care_relu6_fixed_q108_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q118 import vit_base_patch16_224 as vit_care_relu6_fixed_q118_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q128 import vit_base_patch16_224 as vit_care_relu6_fixed_q128_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q810 import vit_base_patch16_224 as vit_care_relu6_fixed_q810_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q811 import vit_base_patch16_224 as vit_care_relu6_fixed_q811_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_q812 import vit_base_patch16_224 as vit_care_relu6_fixed_q812_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_hand import vit_base_patch16_224 as vit_care_relu6_fixed_hand_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_fixed_dump import vit_base_patch16_224 as vit_care_relu6_fixed_dump_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu6_dim32_fixed_dump import (
    vit_tiny32_care_fixed_dump_patch16_224 as vit_care_relu6_dim32_fixed_dump_base_patch16_224,
)
from lib.models.sglatrack.vit_CARE_relu6_dim32_fixed_shared_trunk_dump import (
    vit_tiny32_care_fixed_shared_trunk_dump_patch16_224 as vit_care_relu6_dim32_fixed_shared_trunk_dump_base_patch16_224,
)
from lib.models.sglatrack.vit_CARE_relu6_BN import vit_base_patch16_224 as vit_care_relu6_bn_base_patch16_224
from lib.models.sglatrack.vit_CARE_gelu import vit_base_patch16_224 as vit_care_gelu_base_patch16_224
from lib.models.sglatrack.vit_MALA import vit_base_patch16_224 as vit_mala_base_patch16_224
from lib.models.sglatrack.vit_MALA_CR import vit_base_patch16_224 as vit_mala_cr_base_patch16_224
from lib.models.sglatrack.vit_MALA_relu import vit_base_patch16_224 as vit_mala_relu_base_patch16_224
from lib.models.sglatrack.vit_MALA_relu6 import vit_base_patch16_224 as vit_mala_relu6_base_patch16_224
from lib.models.sglatrack.vit_MALA_relu6_BN import vit_base_patch16_224 as vit_mala_relu6_bn_base_patch16_224
from lib.models.sglatrack.vit_MALA_relu_eps import vit_base_patch16_224 as vit_mala_relu_eps_base_patch16_224
from lib.models.sglatrack.deit import deit_tiny_distilled_patch16_224
from lib.models.sglatrack.deit_MALA_relu import deit_tiny_distilled_mala_relu_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh

# vit_coco_uav123_care_relu6_fixed_q*.yaml：BACKBONE.TYPE → factory（各檔內 to_fixed_point 位寬已寫死）
_VIT_CARE_RELU6_FIXED_Q_BACKBONE_FACTORIES = {
    "vit_care_relu6_fixed_q48_base_patch16_224": vit_care_relu6_fixed_q48_base_patch16_224,
    "vit_care_relu6_fixed_q58_base_patch16_224": vit_care_relu6_fixed_q58_base_patch16_224,
    "vit_care_relu6_fixed_q64_base_patch16_224": vit_care_relu6_fixed_q64_base_patch16_224,
    "vit_care_relu6_fixed_q65_base_patch16_224": vit_care_relu6_fixed_q65_base_patch16_224,
    "vit_care_relu6_fixed_q66_base_patch16_224": vit_care_relu6_fixed_q66_base_patch16_224,
    "vit_care_relu6_fixed_q67_base_patch16_224": vit_care_relu6_fixed_q67_base_patch16_224,
    "vit_care_relu6_fixed_q68_base_patch16_224": vit_care_relu6_fixed_q68_base_patch16_224,
    "vit_care_relu6_fixed_q69_base_patch16_224": vit_care_relu6_fixed_q69_base_patch16_224,
    "vit_care_relu6_fixed_q610_base_patch16_224": vit_care_relu6_fixed_q610_base_patch16_224,
    "vit_care_relu6_fixed_q611_base_patch16_224": vit_care_relu6_fixed_q611_base_patch16_224,
    "vit_care_relu6_fixed_q612_base_patch16_224": vit_care_relu6_fixed_q612_base_patch16_224,
    "vit_care_relu6_fixed_q78_base_patch16_224": vit_care_relu6_fixed_q78_base_patch16_224,
    "vit_care_relu6_fixed_q84_base_patch16_224": vit_care_relu6_fixed_q84_base_patch16_224,
    "vit_care_relu6_fixed_q85_base_patch16_224": vit_care_relu6_fixed_q85_base_patch16_224,
    "vit_care_relu6_fixed_q86_base_patch16_224": vit_care_relu6_fixed_q86_base_patch16_224,
    "vit_care_relu6_fixed_q87_base_patch16_224": vit_care_relu6_fixed_q87_base_patch16_224,
    "vit_care_relu6_fixed_q88_base_patch16_224": vit_care_relu6_fixed_q88_base_patch16_224,
    "vit_care_relu6_fixed_q89_base_patch16_224": vit_care_relu6_fixed_q89_base_patch16_224,
    "vit_care_relu6_fixed_q98_base_patch16_224": vit_care_relu6_fixed_q98_base_patch16_224,
    "vit_care_relu6_fixed_q108_base_patch16_224": vit_care_relu6_fixed_q108_base_patch16_224,
    "vit_care_relu6_fixed_q118_base_patch16_224": vit_care_relu6_fixed_q118_base_patch16_224,
    "vit_care_relu6_fixed_q128_base_patch16_224": vit_care_relu6_fixed_q128_base_patch16_224,
    "vit_care_relu6_fixed_q810_base_patch16_224": vit_care_relu6_fixed_q810_base_patch16_224,
    "vit_care_relu6_fixed_q811_base_patch16_224": vit_care_relu6_fixed_q811_base_patch16_224,
    "vit_care_relu6_fixed_q812_base_patch16_224": vit_care_relu6_fixed_q812_base_patch16_224,
}


class sglatrack(nn.Module):

    def __init__(
        self,
        transformer,
        box_head,
        aux_loss=False,
        head_type="CORNER",
        feat_len_t=64,
        orr_enable=False,
        orr_random_mask=False,
        orr_block_sz=16,
        orr_mask_ratio=0.3,
        orr_gaussian_sigma=64,
    ):
        """Initializes the model.

        orr_*: ORTrack-style optional occlusion / template masking (training only).
        feat_len_t: number of template tokens (H_t*W_t at stride) for ORR sim_loss alignment.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type in ("CORNER", "CENTER", "CENTER_HAND", "CENTER_FIXED", "CENTER_FIXED_SHARED_TRUNK"):
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        self.feat_len_t = int(feat_len_t)
        self.orr_enable = bool(orr_enable)
        self.orr_random_mask = bool(orr_random_mask)
        self.orr_block_sz = int(orr_block_sz)
        self.orr_mask_ratio = float(orr_mask_ratio)
        self.orr_gaussian_sigma = float(orr_gaussian_sigma)
        self._orr_intensity = None

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    @staticmethod
    def _merge_template_input(template):
        if isinstance(template, (list, tuple)):
            if len(template) == 0:
                raise ValueError("template list is empty")
            if len(template) == 1:
                return template[0]
            return torch.stack(template, dim=0).mean(dim=0)
        return template

    def random_masking(self, n, h, w, d, mask_ratio, device):
        len_keep = int(h * w * (1 - mask_ratio))
        noise = torch.rand(n, h, w, device=device)
        noise_vec = torch.reshape(noise, (n, h * w))
        ids_shuffle = torch.argsort(noise_vec, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones([n, h, w], device=device)
        mask_vec = torch.reshape(mask, (n, h * w))
        mask_vec[:, :len_keep] = 0
        mask_vec = torch.gather(mask_vec, dim=1, index=ids_restore)
        mask = torch.reshape(mask_vec, (n, h, w))
        return mask

    def simulate_inhomogeneous_poisson_process(self, intensity):
        num_points = np.random.poisson(intensity.max() * np.prod(intensity.shape), 1)[0]
        x_points = (np.floor(np.random.uniform(0, intensity.shape[1], num_points))).astype(np.int32)
        y_points = (np.floor(np.random.uniform(0, intensity.shape[0], num_points))).astype(np.int32)
        accept_prob = intensity[x_points, y_points] / intensity.max()
        accepted_points = np.random.rand(num_points) < accept_prob
        x_points = x_points[accepted_points]
        y_points = y_points[accepted_points]
        return x_points, y_points

    def random_masking_cox_process(self, intensity, n, h, w, mask_ratio, device):
        poisson_mean = int(h * w * mask_ratio)
        poisson_samples = np.random.poisson(poisson_mean, n)
        masks = []
        for i in range(n):
            inh_poisson_intensity = poisson_samples[i] * intensity
            x_points, y_points = self.simulate_inhomogeneous_poisson_process(inh_poisson_intensity)
            mask = torch.ones([1, h, w], device=device)
            mask[:, y_points, x_points] = 0
            masks.append(mask)
        return torch.cat(masks, dim=0)

    def masking_cox_process(self, n, intensity, block_sz, mask_ratio, device):
        h, w = intensity.shape
        hb = int(h / block_sz)
        wb = int(w / block_sz)
        assert h % block_sz == 0 and w % block_sz == 0, 'template size must be divisible by ORR_BLOCK_SZ'
        intensity = cv2.resize(intensity, dsize=(wb, hb))
        intensity = intensity / intensity.sum()
        mask = self.random_masking_cox_process(intensity, n, hb, wb, mask_ratio, device)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(h, w), mode='nearest')
        return mask

    def masking(self, template, block_sz, mask_ratio, device):
        n, d, h, w = template.shape
        hb = h // block_sz
        wb = w // block_sz
        assert h % block_sz == 0 and w % block_sz == 0, 'template size must be divisible by ORR_BLOCK_SZ'
        mask = self.random_masking(n, hb, wb, d, mask_ratio, device)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(h, w), mode='nearest')
        return mask

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                is_distill=False,
                **kwargs,
                ):
        template = self._merge_template_input(template)
        mask = None
        if (not is_distill) and self.training and self.orr_enable:
            if self.orr_random_mask:
                mask = self.masking(template, self.orr_block_sz, self.orr_mask_ratio, template.device)
                mask = mask.repeat(1, template.shape[1], 1, 1)
            else:
                if self._orr_intensity is None:
                    # 僅 ORR + 非 random mask 路徑需要 scipy；其他 yaml 不 import，避免多一層硬依賴。
                    from scipy.stats import multivariate_normal

                    template_r = int(template.shape[-1] / 2)
                    sigma = self.orr_gaussian_sigma
                    gx, gy = np.mgrid[-template_r:template_r:1, -template_r:template_r:1]
                    pos = np.dstack((gx, gy))
                    intensity = multivariate_normal(
                        [0.0, 0.0],
                        [[sigma * template_r, 0.0], [0.0, sigma * template_r]],
                    ).pdf(pos)
                    intensity = intensity / intensity.sum()
                    self._orr_intensity = intensity
                intensity = self._orr_intensity
                mask = self.masking_cox_process(
                    template.shape[0], intensity, self.orr_block_sz, self.orr_mask_ratio, template.device
                )
                mask = mask.repeat(1, template.shape[1], 1, 1)

        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        if self.training and (not is_distill) and mask is not None:
            x1, _ = self.backbone(z=template * mask, x=search,
                                  ce_template_mask=ce_template_mask,
                                  ce_keep_rate=ce_keep_rate,
                                  return_last_attn=return_last_attn, )
            sim_loss = F.mse_loss(
                x[:, :self.feat_len_t], x1[:, :self.feat_len_t].detach()
            )
        else:
            sim_loss = torch.tensor(0.0, device=x.device)

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['sim_loss'] = sim_loss
        return out

    def forward_test(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone.forward_test(z=template, x=search )

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['sim_loss'] = torch.tensor(0.0, device=feat_last.device)
        return out


    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:] # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
            'score_map': score_map,
            }
            return out

        elif self.head_type in ("CENTER", "CENTER_HAND", "CENTER_FIXED", "CENTER_FIXED_SHARED_TRUNK"):
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
            'score_map': score_map_ctr,
            'size_map': size_map,
            'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_sglatrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    _pf = str(cfg.MODEL.PRETRAIN_FILE or '')
    # MAE / ImageNet weights live under pretrained_models; full training checkpoints are .pth.tar (see below).
    if _pf and (not _pf.endswith('.pth.tar')) and ('sglatrack' not in _pf) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_square_base_patch16_224':
        squaremax_eps = float(getattr(cfg.MODEL.BACKBONE, 'SQUAREMAX_EPS', 1e-6))
        backbone = vit_square_base_patch16_224(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            squaremax_eps=squaremax_eps,
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_sima_base_patch16_224':
        backbone = vit_sima_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_base_patch16_224':
        backbone = vit_care_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu_base_patch16_224':
        backbone = vit_care_relu_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu_fixed_base_patch16_224':
        backbone = vit_care_relu_fixed_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_base_patch16_224':
        backbone = vit_care_relu6_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_dim32_base_patch16_224':
        # embed_dim=32；pretrained 為 vit_tiny_patch16_224.pth（192 維）時由 dim32 模組內投影載入
        backbone = vit_care_relu6_dim32_base_patch16_224(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE in (
        'vit_care_relu6_shared_trunk_dim32_base_patch16_224',
        'vit_care_relu6_fixed_shared_trunk_dim32_base_patch16_224',
    ):
        # embed_dim=32 浮點 CARE ReLU6（無 Q8.8）；預訓練載入同 dim32（Tiny 投影）
        backbone = vit_care_relu6_shared_trunk_dim32_base_patch16_224(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_dim64_base_patch16_224':
        # embed_dim=64；pretrained 為 vit_tiny_patch16_224.pth（192 維）時由 dim64 模組內投影載入
        backbone = vit_care_relu6_dim64_base_patch16_224(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_hand_base_patch16_224':
        backbone = vit_care_relu6_hand_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_fixed_base_patch16_224':
        backbone = vit_care_relu6_fixed_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE in _VIT_CARE_RELU6_FIXED_Q_BACKBONE_FACTORIES:
        _fac = _VIT_CARE_RELU6_FIXED_Q_BACKBONE_FACTORIES[cfg.MODEL.BACKBONE.TYPE]
        backbone = _fac(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_fixed_hand_base_patch16_224':
        # Hand-coded inference-friendly variant: keeps architecture/key names but uses explicit module dataflow.
        backbone = vit_care_relu6_fixed_hand_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_fixed_dump_base_patch16_224':
        # Dump-only variant：僅用於產生 RTL golden intermediate .npy，不應用於 training
        backbone = vit_care_relu6_fixed_dump_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_dim32_fixed_dump_base_patch16_224':
        # Dump-only dim32（與 vit_CARE_relu6_dim32 student 結構對齊）
        backbone = vit_care_relu6_dim32_fixed_dump_base_patch16_224(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_dim32_fixed_shared_trunk_dump_base_patch16_224':
        # 與 dim32_fixed_dump 同一 backbone；鍵名供 shared-trunk head dump yaml
        backbone = vit_care_relu6_dim32_fixed_shared_trunk_dump_base_patch16_224(
            pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu6_bn_base_patch16_224':
        backbone = vit_care_relu6_bn_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_gelu_base_patch16_224':
        backbone = vit_care_gelu_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_mala_base_patch16_224':
        backbone = vit_mala_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_mala_cr_base_patch16_224':
        backbone = vit_mala_cr_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_mala_relu_base_patch16_224':
        backbone = vit_mala_relu_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_mala_relu6_base_patch16_224':
        backbone = vit_mala_relu6_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_mala_relu6_bn_base_patch16_224':
        backbone = vit_mala_relu6_bn_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_mala_relu_eps_base_patch16_224':
        eps_init = getattr(cfg.MODEL.BACKBONE, 'EPS_INIT', 1e-3)
        # YAML/edict may parse scientific notation as string in some environments
        eps_init = float(eps_init)
        backbone = vit_mala_relu_eps_base_patch16_224(
            pretrained,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
            eps_init=eps_init,
        )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE in ('deit_tiny_distilled_patch16', 'deit_tiny_distilled_patch16_224'):
        backbone = deit_tiny_distilled_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 2
    elif cfg.MODEL.BACKBONE.TYPE in (
        'deit_tiny_distilled_mala_relu_patch16_224',
        'deit_tiny_distilled_mala_relu_patch16',
    ):
        backbone = deit_tiny_distilled_mala_relu_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 2
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    _bt = str(cfg.MODEL.BACKBONE.TYPE)
    if _bt in _VIT_CARE_RELU6_FIXED_Q_BACKBONE_FACTORIES and cfg.MODEL.HEAD.TYPE != "CENTER_FIXED":
        raise ValueError(
            f"BACKBONE.TYPE={_bt!r} 必須搭配 MODEL.HEAD.TYPE=CENTER_FIXED（CenterPredictorFixed），"
            f"目前為 {cfg.MODEL.HEAD.TYPE!r}。"
        )

    # The hand-coded backbone variant expects the hand head implementation as well.
    box_head_builder = build_box_head_hand if cfg.MODEL.HEAD.TYPE == "CENTER_HAND" else build_box_head
    box_head = box_head_builder(cfg, hidden_dim)

    # CENTER_DUMP / CENTER_FIXED / CENTER_FIXED_SHARED_TRUNK 在 sglatrack 層級行為都與 CENTER 相同
    head_type_for_sglatrack = (
        "CENTER"
        if cfg.MODEL.HEAD.TYPE in (
            "CENTER_DUMP",
            "CENTER_FIXED",
            "CENTER_FIXED_SHARED_TRUNK",
            "CENTER_DUMP_SHARED_TRUNK",
        )
        else cfg.MODEL.HEAD.TYPE
    )

    tpl = int(cfg.DATA.TEMPLATE.SIZE)
    stride = int(cfg.MODEL.BACKBONE.STRIDE)
    feat_len_t = (tpl // stride) ** 2

    model = sglatrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=head_type_for_sglatrack,
        feat_len_t=feat_len_t,
        orr_enable=bool(getattr(cfg.MODEL, "ORR_ENABLE", False)),
        orr_random_mask=bool(getattr(cfg.MODEL, "ORR_RANDOM_MASK", False)),
        orr_block_sz=int(getattr(cfg.MODEL, "ORR_BLOCK_SZ", 16)),
        orr_mask_ratio=float(getattr(cfg.MODEL, "ORR_MASK_RATIO", 0.3)),
        orr_gaussian_sigma=float(getattr(cfg.MODEL, "ORR_GAUSSIAN_SIGMA", 64)),
    )

    if training and (_pf.endswith('.pth.tar') or ('sglatrack' in _pf)):
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        checkpoint_model = checkpoint["net"]

        # Handle position embedding size mismatch
        if 'backbone.pos_embed_x' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['backbone.pos_embed_x']
            pos_embed_model = model.backbone.pos_embed_x

            if pos_embed_checkpoint.shape != pos_embed_model.shape:
                print(f'Position embedding size mismatch for pos_embed_x:')
                print(f'  Checkpoint: {pos_embed_checkpoint.shape}')
                print(f'  Model: {pos_embed_model.shape}')
                print(f'  Resizing position embedding...')

                pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 1)
                old_size = int(pos_embed_checkpoint.shape[2] ** 0.5)
                new_size = int(pos_embed_model.shape[1] ** 0.5)
                pos_embed_checkpoint = pos_embed_checkpoint.reshape(1, pos_embed_checkpoint.shape[1], old_size, old_size)
                pos_embed_checkpoint = torch.nn.functional.interpolate(
                    pos_embed_checkpoint, size=(new_size, new_size), mode='bicubic', align_corners=False
                )
                pos_embed_checkpoint = pos_embed_checkpoint.reshape(1, pos_embed_checkpoint.shape[1], -1).permute(0, 2, 1)
                checkpoint_model['backbone.pos_embed_x'] = pos_embed_checkpoint
                print(f'  Resized to: {pos_embed_checkpoint.shape}')

        if 'backbone.pos_embed_z' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['backbone.pos_embed_z']
            pos_embed_model = model.backbone.pos_embed_z

            if pos_embed_checkpoint.shape != pos_embed_model.shape:
                print(f'Position embedding size mismatch for pos_embed_z:')
                pos_embed_checkpoint = pos_embed_checkpoint.permute(0, 2, 1)
                old_size = int(pos_embed_checkpoint.shape[2] ** 0.5)
                new_size = int(pos_embed_model.shape[1] ** 0.5)
                pos_embed_checkpoint = pos_embed_checkpoint.reshape(1, pos_embed_checkpoint.shape[1], old_size, old_size)
                pos_embed_checkpoint = torch.nn.functional.interpolate(
                    pos_embed_checkpoint, size=(new_size, new_size), mode='bicubic', align_corners=False
                )
                pos_embed_checkpoint = pos_embed_checkpoint.reshape(1, pos_embed_checkpoint.shape[1], -1).permute(0, 2, 1)
                checkpoint_model['backbone.pos_embed_z'] = pos_embed_checkpoint

        # Remove MLP weights if size mismatch
        mlp_keys_to_remove = []
        for key in list(checkpoint_model.keys()):
            if 'MLP' in key:
                checkpoint_weight = checkpoint_model[key]
                if hasattr(model, 'backbone') and hasattr(model.backbone, 'MLP'):
                    model_key = key.replace('backbone.', '')
                    try:
                        model_weight = dict(model.backbone.named_parameters())[model_key]
                        if checkpoint_weight.shape != model_weight.shape:
                            mlp_keys_to_remove.append(key)
                    except KeyError:
                        pass

        for key in mlp_keys_to_remove:
            del checkpoint_model[key]

        if mlp_keys_to_remove:
            print(f'Removed {len(mlp_keys_to_remove)} MLP weights from checkpoint due to size mismatch')

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
