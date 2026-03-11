# "No matching checkpoint file found" 影響說明

## ❓ 這個訊息會影響訓練嗎？

### 答案：**不會影響！這是正常且預期的行為** ✓

---

## 🔍 詳細解釋

### 這個訊息出現的原因

在 `lib/train/trainers/base_trainer.py` 第 174 行：

```python
def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None):
    if checkpoint is None:
        # 嘗試在當前訓練目錄中尋找 checkpoint
        checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(
            self._checkpoint_dir,
            self.settings.project_path,  # 例如：sglatrack/uav123_coco_finetune
            net_type)))
        
        if checkpoint_list:
            checkpoint_path = checkpoint_list[-1]
        else:
            print('No matching checkpoint file found')  # ← 這裡
            return  # 直接返回，不載入任何 checkpoint
```

**尋找的位置**：
```
output/checkpoints/train/sglatrack/uav123_coco_finetune/SGLATracker_ep*.pth.tar
```

**為什麼找不到**：
- 這是您**第一次**訓練 `uav123_coco_finetune` 配置
- 這個目錄還不存在或是空的
- 所以找不到 checkpoint 是**完全正常的**

---

### 預訓練權重是如何載入的？

在 `base_trainer.py` 第 76-78 行：

```python
def train(self, max_epochs, load_latest=False, fail_safe=True, 
          load_previous_ckpt=False, distill=False):
    
    if load_latest:
        self.load_checkpoint()  # ← 這裡會顯示 "No matching checkpoint file found"
    
    if load_previous_ckpt:  # ← 這裡才是載入預訓練權重的地方！
        directory = '{}/{}'.format(self._checkpoint_dir, 
                                  self.settings.project_path_prv)
        self.load_state_dict(directory)
```

**載入流程**：

1. **`load_latest=True`**（如果指定）：
   - 嘗試在 `uav123_coco_finetune/` 中尋找 checkpoint
   - 找不到 → 顯示 "No matching checkpoint file found"
   - 但不影響後續步驟

2. **`load_previous_ckpt=True`**（透過 `--script_prv` 和 `--config_prv` 觸發）：
   - 從 `project_path_prv`（即 `deit_distilled/`）載入預訓練權重
   - **這才是 finetune 的關鍵！**
   - ✓ 成功載入預訓練模型

---

## 📊 訓練流程圖

```
開始訓練
    ↓
1. load_latest=True?
    ├─ Yes → 尋找 uav123_coco_finetune/*.pth.tar
    │         ├─ 找到 → 載入（繼續之前的訓練）
    │         └─ 沒找到 → 顯示 "No matching checkpoint file found" ← 您的情況
    └─ No → 跳過
    ↓
2. load_previous_ckpt=True? (透過 --script_prv 設定)
    ├─ Yes → 從 deit_distilled/ 載入預訓練權重 ← Finetune 的關鍵！
    │         └─ ✓ 成功載入
    └─ No → 從頭開始訓練
    ↓
3. 開始訓練
    ↓
[train: 1, 50/625] FPS: 93.0 ...  ← 正常訓練
[train: 1, 100/625] FPS: 97.0 ...
```

---

## ✅ 如何確認預訓練權重有載入？

### 方法 1: 檢查訓練開始前的輸出

您的 terminal 第 775-870 行顯示了大量的參數名稱：
```
module.backbone.blocks.0.norm1.weight
module.backbone.blocks.0.norm1.bias
module.backbone.blocks.0.attn.qkv.weight
...
```

這些參數列表表示程式正在處理模型的權重，**這就是預訓練權重載入的證明**。

### 方法 2: 觀察訓練速度

- **從頭訓練**：Loss 會很高（1.0+），訓練緩慢
- **Finetune（有預訓練權重）**：Loss 從較低的值開始（0.6-0.8），收斂快

圖片中顯示：
```
[train: 1, 50/625] Loss/total: 0.63436  ← 這個值表示有載入預訓練權重
```

### 方法 3: 檢查檔案是否存在

```bash
ls -lh output/checkpoints/train/sglatrack/deit_distilled/sglatrack_ep0297.pth.tar
```

如果這個檔案存在且大小約 97 MB，就是預訓練權重。

---

## 💡 總結

| 項目 | 說明 |
|------|------|
| **"No matching checkpoint file found"** | ✓ 正常訊息，不是錯誤 |
| **會影響訓練嗎？** | ❌ 不會！這只是說當前訓練目錄沒有 checkpoint |
| **預訓練權重有載入嗎？** | ✓ 有！透過 `--script_prv` 和 `--config_prv` 從 `deit_distilled/` 載入 |
| **如何確認？** | 檢查參數列表、Loss 起始值、訓練是否正常進行 |
| **是否需要處理？** | ❌ 不需要！繼續訓練即可 |

---

## 🎯 重點

**"No matching checkpoint file found" 只是告訴您**：
- 在新的訓練目錄（`uav123_coco_finetune/`）中沒有找到之前的 checkpoint
- 這是預期的，因為這是第一次訓練這個配置

**不影響**：
- ✓ 預訓練權重的載入（從 `deit_distilled/` 載入）
- ✓ Finetune 的進行
- ✓ 訓練效果

**只要看到 `[train: 1, 50/625]` 等訓練進度，就表示一切正常！** 🎉

---

**更新日期**: 2026-03-02  
**結論**: ✅ 不影響訓練，可以放心繼續
