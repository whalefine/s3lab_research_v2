from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.TYPE = "sglatrack"
cfg.MODEL.NECK_TYPE = "BOTBLOCK"
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = "sine"
cfg.MODEL.BACKBONE_MULTIPLIER = 0.1
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
# Feature distillation (optional; default off — existing yamls unchanged)
cfg.MODEL.IS_DISTILL = False
cfg.MODEL.TEACHER_TYPE = "sglatrack"
cfg.MODEL.TEACHER_PRETRAIN_FILE = ""
cfg.MODEL.TEACHER_CNN_NAME = "regnety_160"
cfg.MODEL.TEACHER_CNN_PRETRAINED = True
# 若為空字串：teacher 與 student 共用 MODEL.BACKBONE.TYPE（舊行為）。
# 若設為例如 vit_base_patch16_224：僅 teacher 用該 backbone 載入 TEACHER_PRETRAIN_FILE（與 student 不同架構時使用）。
cfg.MODEL.TEACHER_BACKBONE_TYPE = ""
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.AQA_QUERY = edict()
cfg.MODEL.AQA_QUERY.ENABLE = False

# ORTrack-style template masking + sim_loss (training only; default off)
cfg.MODEL.ORR_ENABLE = False
cfg.MODEL.ORR_RANDOM_MASK = False
cfg.MODEL.ORR_BLOCK_SZ = 16
cfg.MODEL.ORR_MASK_RATIO = 0.3
cfg.MODEL.ORR_GAUSSIAN_SIGMA = 64

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.TRANSFORMER
cfg.MODEL.TRANSFORMER = edict()
cfg.MODEL.TRANSFORMER.NHEADS = 8
cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
cfg.MODEL.TRANSFORMER.DEC_LAYERS = 6
cfg.MODEL.TRANSFORMER.PRE_NORM = False
cfg.MODEL.TRANSFORMER.DIVIDE_NORM = False

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'

cfg.MODEL.BACKBONE.EPS_INIT = 1e-3  # for vit_MALA_relu_eps: phi(x)=relu(x)+eps (eps learnable init)
cfg.MODEL.BACKBONE.SQUAREMAX_EPS = 1e-6  # vit_square: Squaremax denom clamp

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256
# CenterPredictorFixed（head_fixed / head_hand）：預設與 Q8.8 一致；實驗 yaml 可覆寫（供 _update_config 白名單合併）
cfg.MODEL.HEAD.FIXED_INT_BITS = 8
cfg.MODEL.HEAD.FIXED_FRAC_BITS = 8

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.LR_BACKBONE = 0.00001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone
cfg.TRAIN.DISTILL = False
cfg.TRAIN.DISTILL_LOSS_TYPE = "KL"
cfg.TRAIN.DISTILL_LOSS_WEIGHT = 0.0
cfg.TRAIN.SIM_LOSS_WEIGHT = 0.0
cfg.TRAIN.AFKD_TAU0 = 1.0
cfg.TRAIN.AFKD_RHO = 0.0

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.DEFAULT = 25
cfg.TEST.UPDATE_THRESHOLD = 0.6
cfg.TEST.VIS = 0
cfg.TEST.WINDOW_PENALTY = False
cfg.TEST.PENALTY_K = 0.0
cfg.TEST.NUM_OBJECT_QUERIES = 1


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def _ensure_int_fields(target):
    """Force int for fields that must be integers (avoids YAML float parsing issues)."""
    def _set_int(obj, keys, val):
        for k in keys[:-1]:
            obj = getattr(obj, k, None) or obj[k]
        if hasattr(obj, keys[-1]):
            setattr(obj, keys[-1], int(getattr(obj, keys[-1])))
        elif isinstance(obj, dict) and keys[-1] in obj:
            obj[keys[-1]] = int(obj[keys[-1]])

    for key_path in [
        "TRAIN.EPOCH", "TRAIN.BATCH_SIZE", "TRAIN.NUM_WORKER",
        "TRAIN.LR_DROP_EPOCH", "TRAIN.VAL_EPOCH_INTERVAL", "TRAIN.PRINT_INTERVAL",
        "DATA.MAX_SAMPLE_INTERVAL", "DATA.TRAIN.SAMPLE_PER_EPOCH", "DATA.VAL.SAMPLE_PER_EPOCH",
        "DATA.TEMPLATE.NUMBER", "DATA.SEARCH.NUMBER",
    ]:
        try:
            parts = key_path.split(".")
            obj = target
            for p in parts[:-1]:
                obj = getattr(obj, p) if hasattr(obj, p) else obj[p]
            v = getattr(obj, parts[-1], None) if hasattr(obj, parts[-1]) else (obj.get(parts[-1]) if isinstance(obj, dict) else None)
            if v is not None:
                setattr(obj, parts[-1], int(v)) if hasattr(obj, parts[-1]) else obj.__setitem__(parts[-1], int(v))
        except (TypeError, ValueError, KeyError, AttributeError):
            pass


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
            _ensure_int_fields(base_cfg)
        else:
            _update_config(cfg, exp_config)
            _ensure_int_fields(cfg)
