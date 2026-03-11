# 🎯 UAV123_10fps 測試資料集設定指南

## 📊 現況分析

### ✅ 已完成的設定

您的系統**已經配置好** `uav123_10fps` 資料集！以下是已經存在的配置：

#### 1. 資料集路徑已設定
**檔案**: `lib/test/evaluation/local.py` (第 31 行)
```python
settings.uav123_10fps_path = '/home/junjie/01_Research/SGLATrack-main/data/uav123_10fps/UAV123_10fps'
```

#### 2. 資料集類別已註冊
**檔案**: `lib/test/evaluation/datasets.py` (第 12 行)
```python
uav123_10fps=DatasetInfo(module=pt % "uav123_10fps", class_name="UAV123_10fpsDataset", kwargs=dict()),
```

#### 3. 資料集載入器已實作
**檔案**: `lib/test/evaluation/uav123_10fpsdataset.py`
- 已經有完整的 `UAV123_10fpsDataset` 類別

#### 4. 資料集實體存在
**路徑**: `/home/junjie/01_Research/SGLATrack-main/data/uav123_10fps/UAV123_10fps/`
- ✅ `anno/` 目錄存在
- ✅ `data_seq/` 目錄存在

---

## 🚀 如何使用 UAV123_10fps 進行測試

### 方法 1: 執行測試（如果還沒測試）

```bash
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla

# 執行 UAV123_10fps 測試
python tracking/test.py sglatrack deit_distilled \
  --dataset_name uav123_10fps \
  --threads 4 \
  --num_gpus 1
```

### 方法 2: 計算 FPS（如果已經測試過）

```bash
# 計算 UAV123_10fps 的 FPS
python tracking/calculate_fps.py --dataset uav123_10fps
```

---

## ❓ 需要做任何更改嗎？

### 答案: **不需要！** ✅

您的系統已經完整支援 `uav123_10fps` 資料集，**無需做任何程式碼更改**。

所有必要的配置都已經存在：
- ✅ 路徑設定
- ✅ 資料集註冊
- ✅ 載入器實作
- ✅ 資料實體

---

## 🔍 驗證配置

讓我們驗證一下設定是否正確：

### 檢查 1: 資料集路徑
```bash
ls -la /home/junjie/01_Research/SGLATrack-main/data/uav123_10fps/UAV123_10fps/
# 應該看到: anno/ 和 data_seq/ 目錄
```

### 檢查 2: 測試資料集是否可以載入
```bash
cd /home/junjie/01_Research/SGLATrack-main
python -c "from lib.test.evaluation import get_dataset; ds = get_dataset('uav123_10fps'); print(f'成功載入 {len(ds)} 個序列')"
```

### 檢查 3: 確認測試結果目錄
```bash
ls output/test/tracking_results/sglatrack/deit_distilled/
# 如果已經測試過，應該會看到 uav123_10fps/ 目錄
```

---

## 📝 完整測試流程

### 步驟 1: 執行測試（如果還沒測試）

```bash
# 切換到專案目錄
cd /home/junjie/01_Research/SGLATrack-main

# 啟用環境
conda activate sgla

# 執行測試
python tracking/test.py sglatrack deit_distilled \
  --dataset_name uav123_10fps \
  --threads 4 \
  --num_gpus 1
```

**預期輸出**:
```
Tracker: sglatrack deit_distilled None, Sequence: uav_bike1
FPS: XXX.XX
Tracker: sglatrack deit_distilled None, Sequence: uav_bike2
...
```

### 步驟 2: 計算 FPS

測試完成後，執行：

```bash
python tracking/calculate_fps.py --dataset uav123_10fps
```

**預期輸出**:
```
找到 XXX 個序列的時間檔案
======================================================================

📊 sglatrack (deit_distilled) on uav123_10fps
======================================================================
總序列數:        XXX
總幀數:          XX,XXX
總執行時間:      XXX.XX 秒

平均 FPS:        XXX.XX (總幀數/總時間) ⭐ 推薦
...
```

---

## 🎯 UAV123 vs UAV123_10fps 差異

| 特性 | UAV123 | UAV123_10fps |
|------|--------|--------------|
| 幀率 | 30 FPS | 10 FPS |
| 序列數 | 123 | 123 (相同) |
| 總幀數 | ~112,000 | ~37,000 (約 1/3) |
| 用途 | 標準測試 | 低幀率場景測試 |
| 資料集名稱 | `uav123` | `uav123_10fps` |

---

## 💡 常見問題

### Q1: 我需要修改程式碼嗎？
**A**: 不需要！所有設定都已經完成。

### Q2: 如何測試 UAV123_10fps？
**A**: 使用命令：
```bash
python tracking/test.py sglatrack deit_distilled --dataset_name uav123_10fps --threads 4 --num_gpus 1
```

### Q3: 如何計算 UAV123_10fps 的 FPS？
**A**: 測試完成後，使用：
```bash
python tracking/calculate_fps.py --dataset uav123_10fps
```

### Q4: UAV123_10fps 和 UAV123 有什麼不同？
**A**: 
- UAV123: 原始 30 FPS 資料集（123 個序列）
- UAV123_10fps: 降採樣到 10 FPS 的版本（相同序列，但幀數較少）

### Q5: 我可以同時測試兩個資料集嗎？
**A**: 可以！依序執行：
```bash
# 測試 UAV123
python tracking/test.py sglatrack deit_distilled --dataset_name uav123 --threads 4 --num_gpus 1

# 測試 UAV123_10fps
python tracking/test.py sglatrack deit_distilled --dataset_name uav123_10fps --threads 4 --num_gpus 1

# 計算兩者的 FPS
python tracking/calculate_fps.py --dataset uav123
python tracking/calculate_fps.py --dataset uav123_10fps
```

---

## ✅ 總結

### 需要修改的地方：**無** 🎉

您的系統已經完整支援 `uav123_10fps`，可以直接使用：

1. **測試**: `python tracking/test.py sglatrack deit_distilled --dataset_name uav123_10fps --threads 4 --num_gpus 1`
2. **計算 FPS**: `python tracking/calculate_fps.py --dataset uav123_10fps`

### 已有的配置
- ✅ 資料集路徑: `/home/junjie/01_Research/SGLATrack-main/data/uav123_10fps/UAV123_10fps`
- ✅ 資料集註冊: `datasets.py` 中已註冊 `uav123_10fps`
- ✅ 載入器: `uav123_10fpsdataset.py` 已實作
- ✅ 資料存在: `anno/` 和 `data_seq/` 目錄完整

---

## 🚀 快速開始

```bash
# 立即測試 UAV123_10fps
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla
python tracking/test.py sglatrack deit_distilled --dataset_name uav123_10fps --threads 4 --num_gpus 1
```

測試完成後：
```bash
python tracking/calculate_fps.py --dataset uav123_10fps
```

就這麼簡單！🎉
