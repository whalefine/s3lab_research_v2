# ✅ 錯誤已解決：calculate_fps.py 導入錯誤

## 問題

您遇到的錯誤：
```
File "tracking/calculate_fps.py", line 14, in <module>
    from lib.test.evaluation import env_settings
ImportError: cannot import name 'env_settings' from 'lib.test.evaluation'
```

## 根本原因

有**兩個問題**同時存在：

### 問題 1: 腳本位置錯誤 ❌
- 腳本被移動到 `tracking/` 目錄
- 應該在專案根目錄

### 問題 2: 導入路徑錯誤 ❌
- 原始代碼: `from lib.test.evaluation import env_settings`
- 正確代碼: `from lib.test.evaluation.environment import env_settings`

---

## ✅ 解決方案

### 已修復：

1. **✅ 腳本已移回根目錄**
   - 從 `tracking/calculate_fps.py` → `calculate_fps.py`

2. **✅ 導入路徑已修正**
   - 從 `from lib.test.evaluation import env_settings`
   - 改為 `from lib.test.evaluation.environment import env_settings`

---

## 🎯 正確的使用方式

### 基本用法

```bash
# 1. 切換到專案根目錄
cd /home/junjie/01_Research/SGLATrack-main

# 2. 啟用 conda 環境
conda activate sgla

# 3. 計算 UAV123 的 FPS
python calculate_fps.py --dataset uav123
```

### 完整參數

```bash
python calculate_fps.py \
  --tracker sglatrack \
  --config deit_distilled \
  --dataset uav123
```

---

## 📊 測試結果

執行 `python calculate_fps.py --dataset uav123` 後的輸出：

```
找到 123 個序列的時間檔案
======================================================================

📊 sglatrack (deit_distilled) on uav123
======================================================================
總序列數:        123
總幀數:          112,578
總執行時間:      600.80 秒

平均 FPS:        187.38 (總幀數/總時間) ⭐ 推薦
平均 FPS:        192.75 (各序列平均)
最快 FPS:        314.74
最慢 FPS:        106.07
======================================================================

🏆 最快的 5 個序列:
  1. uav_wakeboard5      :  314.74 FPS
  2. uav_wakeboard6      :  296.96 FPS
  3. uav_wakeboard4      :  289.92 FPS
  4. uav_wakeboard9      :  271.39 FPS
  5. uav_wakeboard3      :  220.59 FPS

🐌 最慢的 5 個序列:
  1. uav_car1_s          :  106.07 FPS
  2. uav_bird1_3         :  106.08 FPS
  3. uav_car11           :  114.72 FPS
  4. uav_truck1          :  125.70 FPS
  5. uav_car4_s          :  128.93 FPS
```

---

## 🎉 現在可以正常使用了！

### 快速命令

```bash
# 計算 UAV123 的 FPS
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla
python calculate_fps.py --dataset uav123
```

### 結果
- ✅ 平均 FPS: **187.38**
- ✅ 處理了 **123 個序列**
- ✅ 總共 **112,578 幀**

---

## ⚠️ 重要提醒

### ✅ 正確的命令格式

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--tracker` | 追蹤器名稱 | sglatrack |
| `--config` | 配置名稱 | deit_distilled |
| `--dataset` | 資料集名稱 | uav123 |

### ❌ 常見錯誤

1. **錯誤**: `python tracking/calculate_fps.py`
   - **正確**: `python calculate_fps.py` (在根目錄執行)

2. **錯誤**: `--param deit_distilled`
   - **正確**: `--config deit_distilled`

3. **錯誤**: 在 `tracking/` 目錄中執行
   - **正確**: 在專案根目錄執行

---

## 📁 文件位置檢查

```bash
# 確認腳本在正確位置
ls calculate_fps.py
# 應該顯示: calculate_fps.py (在根目錄)

# 不應該有這個文件
ls tracking/calculate_fps.py
# 應該顯示: No such file or directory
```

---

## 🔧 技術細節

### 修復的內容

**原始代碼** (錯誤):
```python
from lib.test.evaluation import env_settings
```

**修正後** (正確):
```python
from lib.test.evaluation.environment import env_settings
```

### 為什麼會這樣？

查看 `lib/test/evaluation/__init__.py`:
```python
from .data import Sequence
from .tracker import Tracker, trackerlist
from .datasets import get_dataset
from .environment import create_default_local_file_ITP_test
```

可以看到 `__init__.py` **沒有導出** `env_settings`，所以必須直接從 `environment` 模組導入。

---

## 📊 使用範例

### 計算不同資料集的 FPS

```bash
# UAV123
python calculate_fps.py --dataset uav123

# UAV123_10fps (需要先測試)
python tracking/test.py sglatrack deit_distilled \
  --dataset_name uav123_10fps --threads 4 --num_gpus 1
python calculate_fps.py --dataset uav123_10fps
```

---

## 🎉 總結

### 問題
- ❌ 腳本位置錯誤
- ❌ 導入路徑錯誤

### 解決
- ✅ 腳本已移回根目錄
- ✅ 導入路徑已修正
- ✅ 測試成功運行

### 結果
- ✅ **平均 FPS: 187.38**
- ✅ 腳本現在可以正常使用了！
