---
name: sglatrack-qat
description: >-
  Generate QAT (Quantization-Aware Training) version of any SGLATrack vit_*.py
  backbone for FPGA INT8 deployment. Use when the user asks to add QAT,
  quantization, or INT8 support to a vit variant, or wants to create a new
  vit_*_qat.py file.
---

# SGLATrack QAT Conversion

Convert any `vit_*.py` backbone into a QAT-instrumented version for FPGA INT8 deployment.

## Prerequisites

Before starting, read the source `vit_*.py` file and identify:

1. **Attention.forward** — locate the attention computation, especially:
   - Activation function (ReLU / ReLU6 / ELU+1 / etc.)
   - Whether there is a dynamic division (e.g. `1.0 / (q @ k_mean + eps)`)
   - Whether there is softmax (standard ViT) or linear attention
2. **Block.__init__** — what `act_layer` and `norm_layer` are used
3. **MLP** — whether it uses `timm.Mlp` (with GELU) or a custom one

Also read these reference files to understand the integration points:
- `python/lib/models/sglatrack/base_backbone.py` (forward_ / forward_test with dynamic control flow)
- `python/lib/models/sglatrack/sglatrack.py` (build_sglatrack imports and elif branches)
- `python/lib/train/train_script.py` (QAT prepare hook)
- `python/lib/config/sglatrack/config.py` (cfg.QUANT defaults)
- `python/lib/models/sglatrack/vit_CARE_relu_qat.py` (reference implementation)

## Step-by-step workflow

### Step 1: Create `vit_<NAME>_qat.py`

Copy from the source `vit_<NAME>.py` and apply these modifications:

#### 1a. Add QAT imports

```python
from torch.ao.quantization import QuantStub, DeQuantStub
```

#### 1b. Replace MLP with QATMlp (GELU → ReLU)

If the source uses `timm.Mlp` (which defaults to GELU), replace with a custom `QATMlp` using `nn.ReLU()`. Keep `fc1`, `fc2` key names identical for checkpoint compatibility.

```python
class QATMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

#### 1c. Wrap float-fallback regions in Attention

For any operation that cannot be done in INT8 (dynamic division, exp, softmax, etc.), wrap with `DeQuantStub` → float op → `QuantStub`:

```python
# In Attention.__init__:
self.dequant_z = DeQuantStub()
self.quant_z = QuantStub()

# In Attention.forward, replace:
#   z = 1.0 / (q @ k_mean.transpose(-2, -1) + 1e-5)
# with:
denom = q @ k_mean.transpose(-2, -1) + 1e-5
denom = self.dequant_z(denom)
z = 1.0 / denom
z = self.quant_z(z)
```

#### 1d. Add stubs to VisionTransformer

```python
# In VisionTransformer.__init__:
self.quant_in = QuantStub()       # quantise raw image input
self.dequant_out = DeQuantStub()  # dequantise final output for loss
self.dequant_pe = DeQuantStub()   # float-fallback for pos_embed addition
self.quant_pe = QuantStub()
```

#### 1e. Override forward methods

CRITICAL: `BaseBackbone.forward_` and `forward_test` have dynamic control flow (`topk`, `torch.where`) that cannot be traced by FX. The QAT VisionTransformer MUST override `forward_`, `forward_test`, and `forward` to:

1. Wrap `patch_embed` input with `quant_in`
2. Wrap `pos_embed` addition with `dequant_pe` / `quant_pe` (for converted INT8 compat)
3. Wrap final `norm` output with `dequant_out`
4. Copy the block iteration logic from `base_backbone.py` exactly

#### 1f. Add QAT utility functions

Include these at the bottom of the file (copy from `vit_CARE_relu_qat.py`):
- `_get_qat_qconfig()` — returns QConfig with quint8 activation, per-channel qint8 weight
- `prepare_qat_model(model)` — assigns qconfig, sets LayerNorm to None, calls prepare_qat
- `freeze_bn_stats(model)` — sets all BN to eval
- `convert_qat_model(model)` — converts to true INT8 (for export only)
- `evaluate_qat_model(model, template, search)` — inference with fake-quant
- `export_quant_params(model, filepath)` — exports scale/zero_point dict

#### 1g. Verify state_dict compatibility

The QAT file MUST keep identical `__init__` parameter names for all layers that carry pretrained weights: `qkv`, `proj`, `mlp.fc1`, `mlp.fc2`, `norm1`, `norm2`, `gate_q`, `gate_k`, etc. Extra keys (QuantStub, DeQuantStub, observer) are fine — load with `strict=False`.

### Step 2: Create YAML config

Create `python/experiments/sglatrack/vit_coco_uav123_<name>_qat.yaml`:

- Copy from the source variant's YAML
- Change `MODEL.BACKBONE.TYPE` to the new QAT builder name
- Adjust training hyperparameters for QAT finetune:
  - `EPOCH`: 10-20 (not full 50)
  - `LR`: source LR / 10
  - `LR_BACKBONE`: source LR_BACKBONE / 10
  - `LR_DROP_EPOCH`: ~75% of EPOCH
  - `TEST.EPOCH`: same as TRAIN.EPOCH
- Add QUANT section:

```yaml
QUANT:
  ENABLE: True
  METHOD: QAT
  BITWIDTH: 8
  FREEZE_BN_EPOCH: 5
  FP32_CHECKPOINT: ""
```

### Step 3: Register in sglatrack.py

Add import and elif branch in `python/lib/models/sglatrack/sglatrack.py`:

```python
from lib.models.sglatrack.vit_<NAME>_qat import vit_base_patch16_224 as vit_<name>_qat_base_patch16_224

# In build_sglatrack():
elif cfg.MODEL.BACKBONE.TYPE == 'vit_<name>_qat_base_patch16_224':
    backbone = vit_<name>_qat_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
```

### Step 4: Verify (no train_script changes needed)

`train_script.py` and `ltr_trainer.py` already have QAT hooks (checking `cfg.QUANT.ENABLE`). No modification needed unless the new variant has special requirements.

### Step 5: Smoke test

Run this test to verify everything works:

```python
from lib.models.sglatrack.sglatrack import build_sglatrack
from lib.config.sglatrack.config import cfg, update_config_from_file
update_config_from_file('experiments/sglatrack/vit_coco_uav123_<name>_qat.yaml')
cfg.MODEL.PRETRAIN_FILE = ''
model = build_sglatrack(cfg, training=False)

from lib.models.sglatrack.vit_<NAME>_qat import prepare_qat_model
prepare_qat_model(model)
model.train()
t = torch.randn(2, 3, 128, 128)
s = torch.randn(2, 3, 256, 256)
out = model(t, s)
out['pred_boxes'].sum().backward()
print('QAT smoke test passed')
```

## Key constraints

- **Eager Mode only** — FX Mode cannot trace BaseBackbone's dynamic control flow
- **LayerNorm stays FP32** — set `qconfig=None` on all LayerNorm modules
- **state_dict keys must match** source vit_*.py for checkpoint loading
- Activation dtype is `quint8` (PyTorch FBGEMM requirement); for FPGA signed INT8, convert scale/zp during RTL export
- Converted INT8 model is for **parameter export only** (not PyTorch inference) due to LayerNorm/add limitations

## Attention variant cheatsheet

| Source attention type | Float-fallback region | Notes |
|-----------------------|-----------------------|-------|
| CARE-ReLU (`vit_CARE_relu`) | `1/(q @ k_mean + eps)` | Already done in reference impl |
| MALA-ReLU (`vit_MALA_relu`) | `1/(sum(phi(k)) + eps)` per-head | Similar pattern, wrap the reciprocal |
| MALA-ReLU6 (`vit_MALA_relu6`) | Same as MALA-ReLU | ReLU6 is HW-friendly, keep it |
| Standard softmax (`vit`) | Entire `softmax(QK^T/sqrt(d))` | Large fallback region; QAT benefit reduced |
| SiMA (`vit_sima`) | Depends on SimA normalisation | Inspect forward carefully |

## Files modified/created per QAT conversion

```
NEW:  python/lib/models/sglatrack/vit_<NAME>_qat.py
NEW:  python/experiments/sglatrack/vit_coco_uav123_<name>_qat.yaml
EDIT: python/lib/models/sglatrack/sglatrack.py  (import + elif)
```

`train_script.py`, `ltr_trainer.py`, `config.py` already have QAT support — no changes needed.
