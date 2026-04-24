"""Dump-only variant of vit_CARE_relu6_fixed_hand.

此檔是 ``vit_CARE_relu6_fixed_hand.py`` 的複製版本，**僅在 test-time 用於產生 RTL
golden intermediate ``.npy``**。

設計原則：
1. 不修改 ``vit_CARE_relu6_fixed_hand.py``、``base_backbone_fix.py``、``head_fixed.py``、
   ``sglatrack.py`` 的原始行為。這些檔案維持純淨，供正常 train/test 使用。
2. 本檔內所有 dump 邏輯 100% 自包含。數值計算結果與 ``vit_CARE_relu6_fixed_hand.py``
   **完全一致**（dump 僅是旁路寫檔，不改任何中間張量）。
3. 所有 dump 都發生在 ``to_fixed_point(..., 8, 8)`` 之後，確保寫入磁碟的就是 Q8.8
   量化後值（bit-accurate with RTL）。
4. RTL 的真正入口是 ``forward_test_from_post_embed(template_post_embed, search_post_embed)``
   （兩路 post-embedding token），外層 patch_embed + pos_add 屬於 Python-only 前處理。

與 ``vit_CARE_relu6_fixed_dump`` 先前（基於 ``vit_CARE_relu6_fixed.py``）的差異：
- Linear / LayerNorm / Dropout / DropPath / Mlp / ReLU6 全部改用 ``lib.module``
  的手刻版本（與 ``vit_CARE_relu6_fixed_hand.py`` 完全一致）。
- Backbone 父類別由 ``BaseBackbone`` 改為 ``BaseBackboneFix``（含 Q8.8 截斷的
  ``ThreeLayerMLP`` 與 ``_forward_impl`` 對齊版本）。
- Attention 內新增 ``qk_mean`` / ``qk_mean_eps`` 的 pre-reciprocal 量化節點，
  與 hand 版的最新設計一致。

用法：
    cfg.MODEL.BACKBONE.TYPE: 'vit_care_relu6_fixed_dump_base_patch16_224'
    並給 backbone 設定：
        backbone.dump_enabled = True
        backbone.dump_dir = '/path/to/out_dir'
"""
import math
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from lib.module import (
    Dropout,
    DropPath,
    LayerNorm,
    Linear,
    Mlp,
    relu6,
    to_fixed_point,
)
from lib.models.layers.patch_embed import PatchEmbed
from lib.models.sglatrack.base_backbone_fix import (
    BaseBackboneFix,
    enabled_layer_num,
    start_layer,
)
from lib.models.sglatrack.utils import combine_tokens, recover_tokens


def _save_npy(enabled: bool, dump_dir: str, filename: str, tensor: torch.Tensor) -> None:
    """Dump tensor as .npy if enabled. 所有呼叫點都保證 tensor 已經過 Q8.8 量化。"""
    if not enabled or not dump_dir:
        return
    os.makedirs(dump_dir, exist_ok=True)
    np.save(os.path.join(dump_dir, filename), tensor.detach().cpu().numpy())


class AttentionDump(nn.Module):
    """與 ``vit_CARE_relu6_fixed_hand.Attention`` 完全一致的數值路徑；僅在幾個關鍵節點插入 dump。"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        en = getattr(self, "dump_enabled", False)
        dd = getattr(self, "dump_dir", "")
        pf = getattr(self, "dump_prefix", "")

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = to_fixed_point(q, 8, 8)
        k = to_fixed_point(k, 8, 8)
        v = to_fixed_point(v, 8, 8)
        _save_npy(en, dd, f"{pf}_attn_after_qkv_q.npy", q)
        _save_npy(en, dd, f"{pf}_attn_after_qkv_k.npy", k)
        _save_npy(en, dd, f"{pf}_attn_after_qkv_v.npy", v)

        s = self.scale ** 0.5
        qs = q * s
        ks = k * s
        qs = to_fixed_point(qs, 8, 8)
        ks = to_fixed_point(ks, 8, 8)
        q = relu6(qs)
        k = relu6(ks)
        q = to_fixed_point(q, 8, 8)
        k = to_fixed_point(k, 8, 8)
        v = self.attn_drop(v)
        v = to_fixed_point(v, 8, 8)
        k_mean = k.mean(dim=-2, keepdim=True)
        k_mean = to_fixed_point(k_mean, 8, 8)
        qk_mean = q @ k_mean.transpose(-2, -1)
        qk_mean = to_fixed_point(qk_mean, 8, 8)
        qk_mean_eps = qk_mean + 1e-5
        qk_mean_eps = to_fixed_point(qk_mean_eps, 8, 8)
        z = 1.0 / qk_mean_eps
        z = to_fixed_point(z, 8, 8)
        kv = (k.transpose(-2, -1) @ v) / float(N)
        kv = to_fixed_point(kv, 8, 8)
        x = (q @ kv) * z
        x = to_fixed_point(x, 8, 8)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = to_fixed_point(x, 8, 8)
        x = self.proj(x)
        x = to_fixed_point(x, 8, 8)
        x = self.proj_drop(x)
        x = to_fixed_point(x, 8, 8)
        _save_npy(en, dd, f"{pf}_after_attn_attn_out.npy", x)

        if return_attention:
            attn = qs @ ks.transpose(-2, -1)
            attn = to_fixed_point(attn, 8, 8)
            return x, attn
        return x


class BlockDump(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = AttentionDump(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        en = getattr(self, "dump_enabled", False)
        dd = getattr(self, "dump_dir", "")
        pf = getattr(self, "dump_prefix", "")

        self.attn.dump_enabled = en
        self.attn.dump_dir = dd
        self.attn.dump_prefix = pf

        if return_attention:
            x_norm1 = self.norm1(x)
            x_norm1 = to_fixed_point(x_norm1, 8, 8)
            _save_npy(en, dd, f"{pf}_after_norm1_out.npy", x_norm1)
            feat, attn = self.attn(x_norm1, True)
            feat = to_fixed_point(feat, 8, 8)
            x = x + self.drop_path(feat)
            x = to_fixed_point(x, 8, 8)
            _save_npy(en, dd, f"{pf}_after_residual_add1_out.npy", x)
            x_norm2 = self.norm2(x)
            x_norm2 = to_fixed_point(x_norm2, 8, 8)
            _save_npy(en, dd, f"{pf}_after_norm2_out.npy", x_norm2)
            mlp_out = self.mlp(x_norm2)
            mlp_out = to_fixed_point(mlp_out, 8, 8)
            _save_npy(en, dd, f"{pf}_mlp_after_mlp_out.npy", mlp_out)
            x = x + self.drop_path(mlp_out)
            x = to_fixed_point(x, 8, 8)
            _save_npy(en, dd, f"{pf}_after_block_out.npy", x)
            return x, attn
        else:
            x_norm1 = self.norm1(x)
            x_norm1 = to_fixed_point(x_norm1, 8, 8)
            _save_npy(en, dd, f"{pf}_after_norm1_out.npy", x_norm1)
            attn_out = self.attn(x_norm1)
            attn_out = to_fixed_point(attn_out, 8, 8)
            x = x + self.drop_path(attn_out)
            x = to_fixed_point(x, 8, 8)
            _save_npy(en, dd, f"{pf}_after_residual_add1_out.npy", x)
            x_norm2 = self.norm2(x)
            x_norm2 = to_fixed_point(x_norm2, 8, 8)
            _save_npy(en, dd, f"{pf}_after_norm2_out.npy", x_norm2)
            mlp_out = self.mlp(x_norm2)
            mlp_out = to_fixed_point(mlp_out, 8, 8)
            _save_npy(en, dd, f"{pf}_mlp_after_mlp_out.npy", mlp_out)
            x = x + self.drop_path(mlp_out)
            x = to_fixed_point(x, 8, 8)
            _save_npy(en, dd, f"{pf}_after_block_out.npy", x)
            return x


class VisionTransformerDump(BaseBackboneFix):
    """Dump 專用 backbone：結構／state_dict 鍵名與 ``VisionTransformer``（hand 版）相同。

    關鍵差異：
    - ``forward_test`` 會做 dump（patch embed / pos add / merged / pos_drop / blocks / recover / norm）
    - 額外提供 ``forward_test_from_post_embed(template_post_embed, search_post_embed)``，
      對應 RTL 的真正入口：直接吃兩路 post-embedding token，略過 patch_embed + pos_add。
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.ReLU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            BlockDump(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

        self.dump_enabled = False
        self.dump_dir = ""

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    # ------------------------------------------------------------------
    # dump 傳播
    # ------------------------------------------------------------------
    def _propagate_dump(self):
        for i, blk in enumerate(self.blocks):
            blk.dump_enabled = self.dump_enabled
            blk.dump_dir = self.dump_dir
            blk.dump_prefix = f"backbone_blocks_{i}"

    def _dump(self, tensor: torch.Tensor, filename: str):
        _save_npy(self.dump_enabled, self.dump_dir, filename, tensor)

    # ------------------------------------------------------------------
    # 正常 test：從影像 (z, x) 做完整 patch_embed + pos_add，含 dump
    # ------------------------------------------------------------------
    def forward(self, z, x, **kwargs):
        return self.forward_test(z, x)

    def forward_test(self, z, x):
        self._propagate_dump()

        B = x.shape[0]

        z_patch = self.patch_embed(z)
        x_patch = self.patch_embed(x)
        z_patch = to_fixed_point(z_patch, 8, 8)
        x_patch = to_fixed_point(x_patch, 8, 8)
        self._dump(z_patch, "template_after_patch_embed_out.npy")
        self._dump(x_patch, "search_after_patch_embed_out.npy")

        z_pos = to_fixed_point(self.pos_embed_z, 8, 8)
        x_pos = to_fixed_point(self.pos_embed_x, 8, 8)
        self._dump(z_pos, "template_pos_embed.npy")
        self._dump(x_pos, "search_pos_embed.npy")

        z = to_fixed_point(z_patch + z_pos, 8, 8)
        x = to_fixed_point(x_patch + x_pos, 8, 8)
        self._dump(z, "template_post_embed_input.npy")
        self._dump(x, "search_post_embed_input.npy")

        return self._forward_from_post_embed(z, x, B)

    def forward_test_from_post_embed(self, template_post_embed: torch.Tensor,
                                     search_post_embed: torch.Tensor):
        """RTL 的正式入口：兩路 post-embedding token 直接進入。

        ``template_post_embed``、``search_post_embed`` 必須是 ``patch_embed(img) + pos_embed``
        之後、且已做過 Q8.8 截斷的張量。Python 端同步 dump
        ``template_post_embed_input.npy`` / ``search_post_embed_input.npy`` 供與 RTL 比對。
        """
        self._propagate_dump()
        B = template_post_embed.shape[0]
        self._dump(template_post_embed, "template_post_embed_input.npy")
        self._dump(search_post_embed, "search_post_embed_input.npy")
        return self._forward_from_post_embed(template_post_embed, search_post_embed, B)

    def _forward_from_post_embed(self, z, x, B):
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        x = combine_tokens(z, x, mode=self.cat_mode)
        x = to_fixed_point(x, 8, 8)
        self._dump(x, "merged_tokens.npy")

        x = self.pos_drop(x)
        x = to_fixed_point(x, 8, 8)
        self._dump(x, "after_pos_drop_out.npy")

        pro = None
        sorted_topk_indices = None
        selected_layer_indices = None
        for i, blk in enumerate(self.blocks):
            if i < start_layer:
                x = blk(x)
                x = to_fixed_point(x, 8, 8)
            elif i == start_layer:
                x = blk(x)
                x = to_fixed_point(x, 8, 8)
                pro = self.MLP(x[:, :, 0].clone())
                _, topk_indices = torch.topk(pro, enabled_layer_num, dim=1)
                sorted_topk_indices = torch.sort(topk_indices, dim=1).values + start_layer + 1
                selected_layer_indices = sorted_topk_indices[:, 0]
                self._dump(pro, "adaptive_pro.npy")
                self._dump(sorted_topk_indices, "adaptive_sorted_topk_indices.npy")
                self._dump(selected_layer_indices, "adaptive_selected_layer_index.npy")
            else:
                idx = torch.where(sorted_topk_indices[:, :] == i)[0]
                if len(idx) > 0:
                    x[idx] = blk(x[idx])
                    x[idx] = to_fixed_point(x[idx], 8, 8)
                    break

        # RTL 七層流程：由軟體 adaptive selector 決定唯一一個 6~11 層 block，
        # RTL 端只需要接收 selected layer 的 index 與該層相關資料。
        # 這裡不再額外執行 6~11 的其它 block（避免產生多餘 dump）。
        cos_tensor = torch.zeros(B, 12 - 1 - start_layer, device=x.device)

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)
        x = to_fixed_point(x, 8, 8)
        self._dump(x, "backbone_after_recover_tokens_out.npy")

        x_norm = self.norm(x)
        x_norm = to_fixed_point(x_norm, 8, 8)
        self._dump(x_norm, "backbone_after_norm_backbone_out.npy")

        aux_dict = {
            "attn": None,
            "cos_tensor": cos_tensor.detach(),
            "pro": pro,
            "selected_layer_indices": selected_layer_indices,
        }
        return x_norm, aux_dict


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """與 hand 版同時支援 ``nn.Linear`` / ``Linear`` 及 ``nn.LayerNorm`` / ``LayerNorm``。"""
    if isinstance(module, (nn.Linear, Linear)):
        trunc_normal_(module.weight, std=.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    model = VisionTransformerDump(**kwargs)
    if pretrained:
        if 'npz' in pretrained:
            raise RuntimeError('npz pretrained not supported for dump variant')
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        print('Load pretrained model from: ' + pretrained)
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """Dump-only ViT-Base (ViT-B/16)。結構與 state_dict 鍵名與 ``vit_CARE_relu6_fixed_hand`` 完全相同。"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model
