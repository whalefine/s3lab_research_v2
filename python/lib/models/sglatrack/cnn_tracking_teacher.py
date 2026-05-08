# Frozen CNN (timm) feature-map teacher + lightweight projection to student tokens.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _merge_template(tpl):
    if isinstance(tpl, (list, tuple)):
        if len(tpl) == 0:
            raise ValueError("empty template list")
        if len(tpl) == 1:
            return tpl[0]
        return torch.stack(tpl, dim=0).mean(dim=0)
    return tpl


def _embed_dim(cfg):
    t = str(cfg.MODEL.BACKBONE.TYPE)
    if "tiny" in t:
        return 192
    if "small" in t:
        return 384
    if "base" in t:
        return 768
    return 192


def _token_grid_sizes(cfg):
    stride = int(cfg.MODEL.BACKBONE.STRIDE)
    ht = int(cfg.DATA.TEMPLATE.SIZE)
    wt = int(getattr(cfg.DATA.TEMPLATE, "SIZE", ht))
    hs = int(cfg.DATA.SEARCH.SIZE)
    ws = int(getattr(cfg.DATA.SEARCH, "SIZE", hs))
    return (ht // stride, wt // stride), (hs // stride, ws // stride)


class CNNTrackingTeacher(nn.Module):
    """CNN on template + search -> pooled feature maps -> tokens (B, L, C)."""

    def __init__(self, cnn_name, template_grid, search_grid, embed_dim, pretrained=True, checkpoint_path=""):
        super().__init__()
        self.template_grid = tuple(int(x) for x in template_grid)
        self.search_grid = tuple(int(x) for x in search_grid)
        self.num_tokens = int(self.template_grid[0] * self.template_grid[1] + self.search_grid[0] * self.search_grid[1])
        self.embed_dim = int(embed_dim)
        ckpt = str(checkpoint_path).strip()
        use_timm_pretrained = bool(pretrained) and not ckpt

        self.backbone = timm.create_model(
            cnn_name,
            pretrained=use_timm_pretrained,
            features_only=True,
        )
        chs = getattr(self.backbone, "feature_info", None).channels() if hasattr(self.backbone, "feature_info") else None
        if not chs:
            raise ValueError(f"timm model {cnn_name} does not expose feature_info; can't be used as CNN teacher.")
        nf = int(chs[-1])
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.proj = nn.Conv2d(nf, self.embed_dim, kernel_size=1, bias=True)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

        if ckpt and os.path.isfile(ckpt):
            self._load_backbone_checkpoint(ckpt)

    def _load_backbone_checkpoint(self, path):
        blob = torch.load(path, map_location="cpu")
        state = blob.get("model", blob.get("state_dict", blob))
        if not isinstance(state, dict):
            return
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if len(missing) > len(self.backbone.state_dict()) * 0.5:
            stripped = {}
            for k, v in state.items():
                nk = k
                for pref in ("module.", "backbone."):
                    if nk.startswith(pref):
                        nk = nk[len(pref):]
                if "head" in nk.lower():
                    continue
                stripped[nk] = v
            missing, unexpected = self.backbone.load_state_dict(stripped, strict=False)
        print(
            "[CNN teacher] Loaded backbone checkpoint:",
            path,
            "missing:",
            len(missing),
            "unexpected:",
            len(unexpected),
            flush=True,
        )

    def forward(self, template, search):
        z = _merge_template(template)
        x = search
        with torch.no_grad():
            fz = self.backbone(z)[-1]
            fx = self.backbone(x)[-1]

        fz = F.adaptive_avg_pool2d(fz, output_size=self.template_grid)
        fx = F.adaptive_avg_pool2d(fx, output_size=self.search_grid)
        tz = self.proj(fz).flatten(2).transpose(1, 2)
        tx = self.proj(fx).flatten(2).transpose(1, 2)
        return torch.cat([tz, tx], dim=1)


def build_cnn_tracking_teacher(cfg):
    tg, sg = _token_grid_sizes(cfg)
    embed_dim = _embed_dim(cfg)
    name = getattr(cfg.MODEL, "TEACHER_CNN_NAME", "regnety_160")
    use_pretrained = bool(getattr(cfg.MODEL, "TEACHER_CNN_PRETRAINED", True))
    ckpt = getattr(cfg.MODEL, "TEACHER_PRETRAIN_FILE", "") or ""
    mod = CNNTrackingTeacher(
        cnn_name=name,
        template_grid=tg,
        search_grid=sg,
        embed_dim=embed_dim,
        pretrained=use_pretrained,
        checkpoint_path=str(ckpt).strip(),
    )
    print(
        "[CNN teacher]",
        name,
        "grid_t",
        tg,
        "grid_s",
        sg,
        "tokens",
        mod.num_tokens,
        "C",
        embed_dim,
        "proj_in",
        mod.proj.in_channels,
        flush=True,
    )
    return mod
