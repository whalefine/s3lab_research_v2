# Terminal 錯誤修正完成

## 🔴 發現的錯誤

### 錯誤 1: VAL dataset 使用了不存在的 GOT10K
```
FileNotFoundError: [Errno 2] No such file or directory: 
'.../data/got10k/train/list.txt'
```

**原因**: 配置檔案中 VAL 設定為 `GOT10K_votval`，但您沒有這個資料集

### 錯誤 2: SAMPLE_PER_EPOCH 還是 20000
```python
'SAMPLE_PER_EPOCH': 20000
```

**原因**: 配置檔案沒有更新為討論後的 30000

### 錯誤 3: 訓練腳本位置錯誤
```bash
./train_uav123_finetune.sh
bash: ./train_uav123_finetune.sh: No such file or directory
```

**原因**: 腳本在 `md/` 目錄中，不在根目錄

---

## ✅ 已修正的內容

### 1. 修正 VAL dataset

**檔案**: `experiments/sglatrack/uav123_coco_finetune.yaml`

```yaml
# ❌ 修正前
VAL:
  DATASETS_NAME:
  - GOT10K_votval  # 您沒有這個資料集

# ✓ 修正後
VAL:
  DATASETS_NAME:
  - UAV123          # 使用您有的資料集
  DATASETS_RATIO:
  - 1
  SAMPLE_PER_EPOCH: 5000
```

### 2. 修正 SAMPLE_PER_EPOCH

```yaml
# ❌ 修正前
TRAIN:
  SAMPLE_PER_EPOCH: 20000

# ✓ 修正後
TRAIN:
  SAMPLE_PER_EPOCH: 30000  # 根據 COCO (11萬) + UAV123 (8萬) 的資料量
```

### 3. 修正 VAL_EPOCH_INTERVAL

```yaml
# ❌ 修正前
TRAIN:
  VAL_EPOCH_INTERVAL: 999  # 幾乎不驗證

# ✓ 修正後
TRAIN:
  VAL_EPOCH_INTERVAL: 5    # 每 5 個 epoch 驗證一次
```

### 4. 添加缺少的訓練參數

```yaml
TRAIN:
  # ... 其他參數 ...
  CE_START_EPOCH: 20       # ✓ 新增
  CE_WARM_EPOCH: 80        # ✓ 新增
  FREEZE_LAYERS: [0]       # ✓ 新增
```

### 5. 完善 MODEL 配置

```yaml
MODEL:
  BACKBONE:
    TYPE: deit_tiny_distilled_patch16
    STRIDE: 16
    MID_PE: False           # ✓ 新增
    SEP_SEG: False          # ✓ 新增
    CAT_MODE: 'direct'      # ✓ 新增
    MERGE_LAYER: 0          # ✓ 新增
    ADD_CLS_TOKEN: False    # ✓ 新增
    CLS_TOKEN_USE_MODE: 'ignore'  # ✓ 新增
    CE_LOC: []              # ✓ 新增
    CE_KEEP_RATIO: []       # ✓ 新增
    CE_TEMPLATE_RANGE: 'ALL' # ✓ 新增
```

### 6. 複製訓練腳本到根目錄

```bash
cp md/train_uav123_finetune.sh ./
chmod +x train_uav123_finetune.sh
```

---

## 🚀 現在可以重新訓練

```bash
./train_uav123_finetune.sh
```

或使用 Python 命令：

```bash
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 1 --script_prv sglatrack --config_prv deit_distilled
```

---

## 📊 修正後的配置總覽

```yaml
DATA:
  TRAIN:
    DATASETS_NAME: [COCO17, UAV123]
    SAMPLE_PER_EPOCH: 30000     # ✓ 已修正
  VAL:
    DATASETS_NAME: [UAV123]      # ✓ 已修正（不再是 GOT10K）
    SAMPLE_PER_EPOCH: 5000

TRAIN:
  EPOCH: 100
  BATCH_SIZE: 32
  LR: 0.0001
  VAL_EPOCH_INTERVAL: 5          # ✓ 已修正（每 5 epoch 驗證）
  CE_START_EPOCH: 20             # ✓ 已新增
  CE_WARM_EPOCH: 80              # ✓ 已新增
  FREEZE_LAYERS: [0]             # ✓ 已新增
```

---

## 🎯 預期結果

### 訓練會看到的訊息

```
UAV123: Found 72/91 sequences with annotations  ← ✓ 正常
Loading validation dataset: UAV123               ← ✓ 不再嘗試載入 GOT10K
[train: 1, 50/937] ...                          ← ✓ 937 batches (30000/32)
[train: 1, 100/937] ...
...
[train: 5, 937/937] ...
Validating...                                    ← ✓ 每 5 epoch 驗證一次
```

### 不會再出現的錯誤

- ❌ `FileNotFoundError: .../got10k/train/list.txt`
- ❌ `bash: ./train_uav123_finetune.sh: No such file or directory`
- ✓ SAMPLE_PER_EPOCH 正確設定為 30000

---

## 💡 重點提醒

1. **VAL dataset 必須是您有的資料集**
   - ✓ UAV123（您有）
   - ❌ GOT10K（您沒有）

2. **SAMPLE_PER_EPOCH 根據資料集大小調整**
   - COCO (118K) + UAV123 (82K) = 200K
   - 30000 ≈ 15% 採樣率
   - 每 epoch 937 batches

3. **訓練腳本位置**
   - 已複製到根目錄
   - 可以直接執行 `./train_uav123_finetune.sh`

---

**更新日期**: 2026-03-02  
**問題狀態**: ✅ 全部修正完成
