import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.sglatrack.vit import vit_base_patch16_224
from lib.models.sglatrack.vit_square import vit_base_patch16_224 as vit_square_base_patch16_224
from lib.models.sglatrack.vit_sima import vit_base_patch16_224 as vit_sima_base_patch16_224
from lib.models.sglatrack.vit_CARE import vit_base_patch16_224 as vit_care_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu import vit_base_patch16_224 as vit_care_relu_base_patch16_224
from lib.models.sglatrack.vit_CARE_relu_BN import vit_base_patch16_224 as vit_care_relu_bn_base_patch16_224
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


class sglatrack(nn.Module):

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_test(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone.forward_test(z=template, x=search )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
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

        elif self.head_type == "CENTER":
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

    if cfg.MODEL.PRETRAIN_FILE and ('sglatrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
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
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_care_relu_bn_base_patch16_224':
        backbone = vit_care_relu_bn_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
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

    box_head = build_box_head(cfg, hidden_dim)

    model = sglatrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'sglatrack' in cfg.MODEL.PRETRAIN_FILE and training:
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
