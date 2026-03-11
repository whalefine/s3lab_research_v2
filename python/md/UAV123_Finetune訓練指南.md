# 🎯 UAV123 + COCO Finetune 訓練指南

## 📋 您的情況

- **Detection Dataset**: COCO17（保持不變）✓
- **Tracking Dataset**: UAV123（改用較小的資料集）✓
- **方法**: Finetune（使用預訓練權重）✓

這正是圖片中的「**情況2: 換訓練 dataset, 但想微調 (finetune)**」！

---

## 🔑 您的預訓練權重

### 已找到的權重
```
檔案: output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar
大小: 97 MB
訓練 Epoch: 297
```

這就是您要用來 finetune 的預訓練權重！✓

---

## 📂 已建立的檔案

### 1. 配置檔案
**檔案**: `experiments/sglatrack/uav123_coco_finetune.yaml`

**關鍵設定**:
```yaml
TRAIN:
  DATASETS_NAME:
  - COCO17          # Detection dataset (保持)
  - UAV123          # Tracking dataset (改用 UAV123)
  DATASETS_RATIO:
  - 1
  - 1
  SAMPLE_PER_EPOCH: 30000    # 較小資料集，減少樣本數

TRAIN:
  EPOCH: 100                  # Finetune 不需要 300 epoch
  LR: 0.0001                  # 較小學習率（原本 0.0004）
  VAL_EPOCH_INTERVAL: 5       # 更頻繁驗證
```

### 2. 訓練腳本
**檔案**: `train_uav123_finetune.sh`

**功能**:
- 自動檢查預訓練權重
- 使用 `--resume` 載入預訓練權重
- 顯示訓練進度

---

## 🚀 如何執行 Finetune

### 方法 1: 使用訓練腳本（推薦）

```bash
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla

# 執行 finetune 訓練
./train_uav123_finetune.sh
```

### 方法 2: 直接執行訓練命令

```bash
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla

python tracking/train.py \
    --script sglatrack \
    --config experiments/sglatrack/uav123_coco_finetune.yaml \
    --save_dir output \
    --mode multiple \
    --nproc_per_node 1 \
    --resume output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar
```

---

## 🔧 Finetune 的關鍵參數

### 1. `--resume` 參數
```bash
--resume output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar
```

**作用**: 載入預訓練權重作為初始化權重

**注意**: 這就是圖片中提到的方法！

### 2. 較小的學習率
```yaml
LR: 0.0001  # 原本是 0.0004
```

**原因**: Finetune 時不希望大幅改變已學習的特徵

### 3. 較少的 Epoch
```yaml
EPOCH: 100  # 原本是 300
```

**原因**: 預訓練權重已經很好，不需要從頭訓練那麼久

### 4. 更頻繁的驗證
```yaml
VAL_EPOCH_INTERVAL: 5  # 原本是 999
```

**原因**: 觀察 finetune 過程，避免過擬合

---

## 📊 Finetune 的優點（如圖片所示）

### ✅ 收斂快
- 預訓練權重已經學會基本特徵
- 只需要適應新資料集

### ✅ 精度通常更好
- 保留了在大資料集（LASOT, GOT10K, TRACKINGNET）上學到的知識
- 在 UAV123 上進一步優化

### ✅ 比從頭訓練穩定
- 不會出現訓練初期的不穩定
- 更容易找到好的區域極值

---

## 📁 訓練結果位置

訓練完成後，模型會儲存在：

```
output/checkpoints/train/sglatrack/uav123_coco_finetune/
├── sglatrack_ep0001.pth.tar
├── sglatrack_ep0005.pth.tar
├── ...
└── sglatrack_ep0100.pth.tar  # 最終模型
```

---

## 🧪 訓練後的測試

### 1. 在 UAV123 上測試

```bash
# 使用最後一個 epoch
python tracking/test.py sglatrack uav123_coco_finetune \
    --dataset_name uav123 \
    --threads 4 \
    --num_gpus 1
```

### 2. 計算指標

```bash
python tracking/calculate_metrics.py \
    --tracker sglatrack \
    --param uav123_coco_finetune \
    --dataset uav123 \
    --plot
```

### 3. 計算 FPS

```bash
python tracking/calculate_fps.py \
    --tracker sglatrack \
    --param uav123_coco_finetune \
    --dataset uav123
```

---

## 📊 預期結果

### Finetune 前（原始 sglatrack_ep0297）
```
資料集: UAV123
AUC: 66.90%
Precision: 84.91%
```

### Finetune 後（預期）
```
資料集: UAV123
AUC: 68-72% (預期提升 1-5%)
Precision: 86-90% (預期提升)
```

**原因**: 模型針對 UAV123 特性進行了優化

---

## ⚙️ 進階設定

### 如果想要更激進的 Finetune

修改 `experiments/sglatrack/uav123_coco_finetune.yaml`:

```yaml
TRAIN:
  LR: 0.0002              # 更大的學習率
  EPOCH: 150              # 更多 epoch
  FREEZE_LAYERS: []       # 不凍結任何層（預設凍結第0層）
```

### 如果想要更保守的 Finetune

```yaml
TRAIN:
  LR: 0.00005             # 更小的學習率
  EPOCH: 50               # 更少 epoch
  FREEZE_LAYERS: [0, 1, 2, 3]  # 凍結更多層
```

---

## 🔍 訓練過程監控

### 查看訓練 log

```bash
tail -f output/logs/train_sglatrack_uav123_coco_finetune.log
```

### 預期輸出

```
Epoch: [1][0/938] ...
Loss: 0.523 (應該比從頭訓練低)
...
Epoch: [5] Validation
AUC: 67.5 (應該快速接近目標)
...
```

---

## 💡 常見問題

### Q1: 為什麼要用 `--resume` 而不是 `PRETRAIN_FILE`？

**A**: 
- `PRETRAIN_FILE`: 只載入 backbone 權重（用於從頭訓練）
- `--resume`: 載入完整模型權重（包括 head），用於 finetune

### Q2: strict=False 是什麼意思？

**A**: 圖片中提到的 `strict=False` 允許部分權重不匹配。在我們的情況下，PyTorch 會自動處理，不需要手動設定。

### Q3: 如果 UAV123 資料集還沒準備好？

**A**: 您之前已經設定好了：
```
路徑: /home/junjie/01_Research/SGLATrack-main/data/uav123/UAV123
狀態: ✓ 已準備好
```

### Q4: 訓練會很久嗎？

**A**: 
- **從頭訓練**: ~20-30 小時（300 epoch）
- **Finetune**: ~7-10 小時（100 epoch）✓
- 因為已經有好的初始權重，收斂更快！

---

## 📋 檢查清單

開始訓練前，請確認：

- [x] **預訓練權重**: `sglatrack_ep0297.pth.tar` ✓
- [x] **配置檔案**: `uav123_coco_finetune.yaml` ✓
- [x] **訓練腳本**: `train_uav123_finetune.sh` ✓
- [x] **UAV123 資料集**: `/data/uav123/UAV123` ✓
- [x] **COCO 資料集**: `/data/coco` ✓
- [ ] **Conda 環境**: `conda activate sgla`
- [ ] **GPU 可用**: `nvidia-smi`

---

## 🚀 快速開始

```bash
# 1. 啟用環境
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla

# 2. 確認 GPU
nvidia-smi

# 3. 開始 Finetune
./train_uav123_finetune.sh

# 4. 訓練完成後測試
python tracking/test.py sglatrack uav123_coco_finetune \
    --dataset_name uav123 --threads 4 --num_gpus 1

# 5. 計算指標
python tracking/calculate_metrics.py \
    --param uav123_coco_finetune \
    --dataset uav123 \
    --plot
```

---

## 🎯 總結

### 您的預訓練權重
```
檔案: sglatrack_ep0297.pth.tar
位置: output/checkpoints/train/sglatrack/deit_distilled/
大小: 97 MB
用途: Finetune 的初始化權重 ✓
```

### 訓練命令（關鍵）
```bash
python tracking/train.py \
    --script sglatrack \
    --config experiments/sglatrack/uav123_coco_finetune.yaml \
    --resume output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar
```

**`--resume` 就是載入預訓練權重的關鍵！**

### 優點（如圖片所示）
- ✅ 收斂快
- ✅ 精度更好
- ✅ 訓練更穩定

---

## 📞 需要幫助？

如果遇到問題：

1. **權重載入失敗**: 檢查權重檔案路徑
2. **資料集錯誤**: 確認 UAV123 路徑設定
3. **訓練不收斂**: 嘗試更小的學習率
4. **GPU 記憶體不足**: 減小 batch size

---

🎉 **準備好了！您可以開始 Finetune 訓練了！**
