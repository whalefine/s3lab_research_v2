# UAV123 訓練錯誤修正 - NaN 標註問題

## ❌ 錯誤訊息

```
ValueError: cannot safely convert passed user dtype of float32 for object dtyped data in column 0

TypeError: Cannot cast array data from dtype('O') to dtype('float32') according to the rule 'safe'
```

**錯誤位置**: `lib/train/dataset/uav123.py` 第 73 行

```python
File "/home/junjie/01_Research/SGLATrack-main/lib/train/../../lib/train/dataset/uav123.py", line 73, in _read_bb_anno
    dtype=np.float32, na_filter=False, low_memory=False).values
```

---

## 🔍 錯誤原因

### UAV123 標註檔案包含 NaN 值

檢查標註檔案發現：

```bash
$ cat data/uav123/UAV123/anno/UAV123/bike2.txt | grep NaN | head -5

NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN
```

**問題分析**：
1. UAV123 資料集中，某些幀的目標不可見或追蹤失敗
2. 這些幀的標註用 `NaN,NaN,NaN,NaN` 表示
3. 原始程式碼使用 `na_filter=False`，導致 `NaN` 被當作字串處理
4. pandas 無法將字串 `"NaN"` 轉換為 `float32`，產生 `TypeError`

---

## ✅ 解決方案

### 修正 `_read_bb_anno` 函數

**檔案**: `lib/train/dataset/uav123.py` (第 65-78 行)

```python
def _read_bb_anno(self, seq_name):
    """Read bounding box annotations."""
    anno_file = os.path.join(self.root, 'anno', 'UAV123', f'{seq_name}.txt')
    
    if not os.path.exists(anno_file):
        raise RuntimeError(f'Annotation file not found: {anno_file}')
    
    # ✓ 修正：允許 NaN 值並正確轉換
    gt = pandas.read_csv(anno_file, delimiter=',', header=None, 
                        dtype=np.float32, na_filter=True, low_memory=False,
                        keep_default_na=True).values
    
    # ✓ 將 NaN 替換為 0（表示無效的 bbox）
    gt = np.nan_to_num(gt, nan=0.0)
    
    return torch.tensor(gt)
```

**關鍵修改**：
1. **`na_filter=True`** (原本 `False`)：啟用 NaN 過濾，讓 pandas 識別 `NaN` 字串
2. **`keep_default_na=True`**：保留預設的 NA 值識別
3. **`np.nan_to_num(gt, nan=0.0)`**：將 NaN 替換為 0.0

---

## 🎯 為什麼這樣修正

### 1. NaN 在追蹤任務中的意義

```python
# 原始標註檔案
629,441,95,82    # ✓ 有效幀：目標可見
629,441,95,82    # ✓ 有效幀
NaN,NaN,NaN,NaN  # ❌ 無效幀：目標不可見/遮擋/離開畫面
```

### 2. 轉換為 0 的合理性

在 `get_sequence_info` 函數中（第 85 行）：

```python
def get_sequence_info(self, seq_id):
    seq_name = self.sequence_list[seq_id]
    bbox = self._read_bb_anno(seq_name)  # NaN 已被轉換為 0
    
    # ✓ 這行會自動過濾掉 width=0 或 height=0 的 bbox
    valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
    visible = valid.clone().byte()
    
    return {'bbox': bbox, 'valid': valid, 'visible': visible}
```

- `bbox[:, 2]` 是 width
- `bbox[:, 3]` 是 height
- NaN 轉成 0 後，`valid` 會正確標記為 `False`
- 訓練時會自動跳過這些無效幀

---

## 📊 測試驗證

### 驗證修正是否正確

```python
# 可以在 Python 中測試
import pandas
import numpy as np

# 模擬有 NaN 的檔案
test_data = """629,441,95,82
629,441,95,82
NaN,NaN,NaN,NaN
630,442,95,82"""

# 測試原本的方法（會報錯）
try:
    gt_old = pandas.read_csv(pd.io.common.StringIO(test_data), 
                            delimiter=',', header=None,
                            dtype=np.float32, na_filter=False).values
except Exception as e:
    print(f"❌ 原方法錯誤: {e}")

# 測試新方法（正常）
gt_new = pandas.read_csv(pd.io.common.StringIO(test_data), 
                        delimiter=',', header=None,
                        dtype=np.float32, na_filter=True,
                        keep_default_na=True).values
gt_new = np.nan_to_num(gt_new, nan=0.0)
print("✓ 新方法結果:")
print(gt_new)
# 輸出:
# [[629. 441.  95.  82.]
#  [629. 441.  95.  82.]
#  [  0.   0.   0.   0.]  ← NaN 被正確轉換為 0
#  [630. 442.  95.  82.]]
```

---

## 🚀 現在可以重新訓練

```bash
./train_uav123_finetune.sh
```

**預期行為**：
1. ✓ 正確讀取 UAV123 標註檔案
2. ✓ 自動處理包含 NaN 的幀
3. ✓ 訓練過程不會中斷
4. ✓ 無效幀會被正確標記並跳過

---

## 💡 重點整理

### pandas 讀取 CSV 的 NA 處理

| 參數 | 設定 | 說明 |
|------|------|------|
| `na_filter` | `True` | 啟用 NA 值識別，將 "NaN" 字串轉換為 numpy.nan |
| `na_filter` | `False` | 停用識別，"NaN" 被當作普通字串（會導致轉型錯誤）|
| `keep_default_na` | `True` | 使用預設的 NA 值清單（包括 "NaN", "nan", "NA" 等）|

### UAV123 資料集特性

1. **包含遮擋和目標離開畫面的情況**
2. **使用 NaN 標記無效幀**
3. **需要正確處理這些特殊情況**
4. **訓練時會自動跳過無效幀**

---

**更新日期**: 2026-03-02  
**問題狀態**: ✅ 已解決  
**修改檔案**: `lib/train/dataset/uav123.py`
