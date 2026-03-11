# Python 命令方式執行訓練

## 📝 除了 `./train_uav123_finetune.sh`，如何用 Python 命令執行？

---

## 方法 1: 直接使用 python 命令（推薦）

### 基本命令

```bash
python tracking/train.py \
    --script sglatrack \
    --config uav123_coco_finetune \
    --save_dir output \
    --mode multiple \
    --nproc_per_node 1 \
    --script_prv sglatrack \
    --config_prv deit_distilled
```

### 單行版本（方便複製）

```bash
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 1 --script_prv sglatrack --config_prv deit_distilled
```

---

## 方法 2: 使用多 GPU 訓練

### 使用 2 個 GPU

```bash
python tracking/train.py \
    --script sglatrack \
    --config uav123_coco_finetune \
    --save_dir output \
    --mode multiple \
    --nproc_per_node 2 \
    --script_prv sglatrack \
    --config_prv deit_distilled
```

### 使用 4 個 GPU

```bash
python tracking/train.py \
    --script sglatrack \
    --config uav123_coco_finetune \
    --save_dir output \
    --mode multiple \
    --nproc_per_node 4 \
    --script_prv sglatrack \
    --config_prv deit_distilled
```

---

## 方法 3: 單 GPU 模式

```bash
python tracking/train.py \
    --script sglatrack \
    --config uav123_coco_finetune \
    --save_dir output \
    --mode single \
    --script_prv sglatrack \
    --config_prv deit_distilled
```

---

## 方法 4: 使用 torchrun（PyTorch 新版推薦）

```bash
torchrun --nproc_per_node=1 \
    tracking/train.py \
    --script sglatrack \
    --config uav123_coco_finetune \
    --save_dir output \
    --mode multiple \
    --script_prv sglatrack \
    --config_prv deit_distilled
```

---

## 📖 參數說明

### 必要參數

| 參數 | 說明 | 範例值 |
|------|------|--------|
| `--script` | 訓練腳本名稱 | `sglatrack` |
| `--config` | 配置檔案名稱（不含 .yaml） | `uav123_coco_finetune` |
| `--save_dir` | 儲存目錄 | `output` |
| `--mode` | 訓練模式 | `single` 或 `multiple` |

### Finetune 專用參數（重要！）

| 參數 | 說明 | 範例值 |
|------|------|--------|
| `--script_prv` | 預訓練模型的腳本名稱 | `sglatrack` |
| `--config_prv` | 預訓練模型的配置名稱 | `deit_distilled` |

**如果不加這兩個參數，就不會載入預訓練權重！**

### 多 GPU 參數

| 參數 | 說明 | 範例值 |
|------|------|--------|
| `--nproc_per_node` | 每個節點使用的 GPU 數量 | `1`, `2`, `4` |

### 其他可選參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--use_lmdb` | 使用 LMDB 格式資料集 | `0` |
| `--use_wandb` | 使用 W&B 記錄 | `0` |
| `--seed` | 隨機種子 | `42` |

---

## 🎯 常用場景

### 場景 1: 快速測試（單 GPU，少 epoch）

先修改 `experiments/sglatrack/uav123_coco_finetune.yaml`：
```yaml
TRAIN:
  EPOCH: 10  # 改成 10 用於測試
```

然後執行：
```bash
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode single --script_prv sglatrack --config_prv deit_distilled
```

### 場景 2: 正式訓練（多 GPU，完整 epoch）

```bash
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 2 --script_prv sglatrack --config_prv deit_distilled
```

### 場景 3: 從頭訓練（不使用預訓練權重）

```bash
# 不加 --script_prv 和 --config_prv
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 1
```

### 場景 4: 繼續之前中斷的訓練

假設之前訓練到 epoch 50 中斷了：

```bash
# 程式會自動從 uav123_coco_finetune/ 中找到最新的 checkpoint 繼續
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 1
```

**注意**：這種情況不需要 `--script_prv`，因為是繼續同一個訓練，不是從別的模型 finetune。

---

## 🔧 背景執行（長時間訓練）

### 使用 nohup

```bash
nohup python tracking/train.py \
    --script sglatrack \
    --config uav123_coco_finetune \
    --save_dir output \
    --mode multiple \
    --nproc_per_node 1 \
    --script_prv sglatrack \
    --config_prv deit_distilled \
    > train_uav123.log 2>&1 &
```

查看訓練進度：
```bash
tail -f train_uav123.log
```

### 使用 tmux（推薦）

```bash
# 建立新 session
tmux new -s uav123_train

# 在 tmux 中執行訓練
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 1 --script_prv sglatrack --config_prv deit_distilled

# 離開 tmux（訓練繼續在背景執行）
按 Ctrl+B，然後按 D

# 重新連接
tmux attach -t uav123_train

# 列出所有 session
tmux ls
```

### 使用 screen

```bash
# 建立新 screen
screen -S uav123_train

# 執行訓練
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 1 --script_prv sglatrack --config_prv deit_distilled

# 離開 screen
按 Ctrl+A，然後按 D

# 重新連接
screen -r uav123_train
```

---

## 📊 與 shell 腳本的比較

| 方式 | 優點 | 缺點 |
|------|------|------|
| **Shell 腳本** (`./train_uav123_finetune.sh`) | • 方便重複執行<br>• 可加入檢查邏輯<br>• 顯示友善訊息 | • 需要先寫腳本<br>• 修改參數要編輯檔案 |
| **Python 命令** | • 靈活調整參數<br>• 不需要額外檔案<br>• 直接執行 | • 命令較長<br>• 容易打錯參數 |

---

## 💡 快速參考

### 最簡單的方式（複製即用）

```bash
# 單 GPU finetune
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 1 --script_prv sglatrack --config_prv deit_distilled

# 2 GPU finetune
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 2 --script_prv sglatrack --config_prv deit_distilled

# 從頭訓練（不用預訓練權重）
python tracking/train.py --script sglatrack --config uav123_coco_finetune --save_dir output --mode multiple --nproc_per_node 1
```

---

**更新日期**: 2026-03-02  
**建議**: 開發階段用 Python 命令方便調整，正式訓練用 Shell 腳本更穩定
