import os
import copy
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
from lib.train.trainers import LTRTrainer
from .base_functions import *
from lib.models.sglatrack import build_sglatrack
from lib.models.sglatrack.cnn_tracking_teacher import build_cnn_tracking_teacher
from lib.train.actors.sglatrack_distill import sglatrackDistillActor
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'sglatrack training (with optional feature distillation)'

    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    update_settings(settings, cfg)

    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    print("[Model] Creating network and loading pretrained weights (may take a moment) ...", flush=True)
    net_teacher = None
    is_distill = bool(getattr(cfg.MODEL, "IS_DISTILL", False))
    if settings.script_name == "sglatrack":
        teacher_type = str(getattr(cfg.MODEL, "TEACHER_TYPE", "sglatrack")).lower()
        if is_distill and teacher_type not in ("sglatrack", "cnn"):
            raise ValueError("MODEL.TEACHER_TYPE must be 'sglatrack' or 'cnn' when IS_DISTILL=True.")
        if is_distill and teacher_type == "sglatrack":
            tpath = getattr(cfg.MODEL, "TEACHER_PRETRAIN_FILE", "") or ""
            if not str(tpath).strip():
                raise ValueError(
                    "MODEL.IS_DISTILL=True with TEACHER_TYPE=sglatrack requires MODEL.TEACHER_PRETRAIN_FILE."
                )
            cfg_teacher = copy.deepcopy(cfg)
            cfg_teacher.MODEL.PRETRAIN_FILE = tpath
            cfg_teacher.MODEL.IS_DISTILL = False
            tb = str(getattr(cfg.MODEL, "TEACHER_BACKBONE_TYPE", "") or "").strip()
            if tb:
                cfg_teacher.MODEL.BACKBONE.TYPE = tb
                if settings.local_rank in [-1, 0]:
                    print("[Model] Teacher backbone TYPE override:", tb, flush=True)
            net_teacher = build_sglatrack(cfg_teacher, training=True)
        net = build_sglatrack(cfg)
        # Teacher 768 / student 32 等：在 student 上掛可訓練 Linear，蒸餾時把 teacher backbone_feat 對齊到 student 維度。
        if is_distill and net_teacher is not None and teacher_type == "sglatrack":
            tw = net_teacher.module if hasattr(net_teacher, "module") else net_teacher
            sw = net.module if hasattr(net, "module") else net
            tdim = int(tw.backbone.embed_dim)
            sdim = int(sw.backbone.embed_dim)
            if tdim != sdim:
                sw.add_module("distill_teacher_feat_align", torch.nn.Linear(tdim, sdim))
                if settings.local_rank in [-1, 0]:
                    print(f"[Distill] distill_teacher_feat_align trainable: {tdim} -> {sdim}", flush=True)
        if is_distill and teacher_type == "cnn":
            net_teacher = None
            net.add_module("distill_cnn_adapter", build_cnn_tracking_teacher(cfg))
            if settings.local_rank in [-1, 0]:
                print("[Model] CNN teacher (timm) attached as net.distill_cnn_adapter.", flush=True)
    else:
        raise ValueError("illegal script name")

    if net_teacher is not None:
        net_teacher.eval()
        for _p in net_teacher.parameters():
            _p.requires_grad_(False)
    # 單卡訓練：不使用 DDP。請搭配 tracking/train.py --mode single，且 NUM_GPUS=1。
    if torch.cuda.is_available():
        settings.device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        settings.device = torch.device("cpu")
    net = net.to(settings.device)
    if net_teacher is not None:
        net_teacher = net_teacher.to(settings.device)
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    if settings.script_name == "sglatrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {
            'giou': cfg.TRAIN.GIOU_WEIGHT,
            'l1': cfg.TRAIN.L1_WEIGHT,
            'focal': 1.,
            'cls': 1.0,
            'cos': 0.2,
            'sim_loss': float(getattr(cfg.TRAIN, "SIM_LOSS_WEIGHT", 0.0)),
            'distill_loss': float(getattr(cfg.TRAIN, "DISTILL_LOSS_WEIGHT", 2e-5)),
        }
        actor = sglatrackDistillActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        if is_distill:
            actor.net_teacher = net_teacher
        net.is_distill_training = bool(is_distill)
    else:
        raise ValueError("illegal script name")

    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    trainer.train(int(cfg.TRAIN.EPOCH), load_latest=True, fail_safe=True)
