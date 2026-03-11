# 📊 計算 AUC 和 Precision 指南

## ✅ 論文提供的工具

是的！論文**已提供** Python 工具來計算 AUC 和 Precision！

---

## 🚀 快速開始

### 基本用法

```bash
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla

# 計算 UAV123 的 AUC 和 Precision
python tracking/calculate_metrics.py --dataset uav123
```

### 完整參數

```bash
python tracking/calculate_metrics.py \
  --tracker sglatrack \
  --param deit_distilled \
  --dataset uav123 \
  --plot
```

---

## 📊 測試結果示例

執行後會輸出：

```
================================================================================
計算追蹤指標
================================================================================
追蹤器: sglatrack
配置: deit_distilled
資料集: uav123
================================================================================

載入資料集: uav123...
✓ 已載入 123 個序列

計算指標中...

Computed results over 123 / 123 sequences

Reporting results over 123 / 123 sequences

uav123                        | AUC        | OP50       | OP75       | Precision    | Norm Precision    | FPS        |
sglatrack_deit_distilled      | 66.90      | 82.23      | 60.15      | 84.91        | 82.56             | 205.85     |


================================================================================
指標說明:
================================================================================
AUC (Area Under Curve)    : Success Plot 下的面積 (越高越好, 0-100)
OP50                      : Overlap 閾值 0.5 時的成功率 (%)
OP75                      : Overlap 閾值 0.75 時的成功率 (%)
Precision                 : 距離閾值 20 pixels 時的精確度 (%)
Norm Precision            : 歸一化距離閾值時的精確度 (%)
================================================================================
```

---

## 📈 指標解釋

### 1. AUC (Area Under Curve) - 66.90
**定義**: Success Plot (重疊成功率曲線) 下的面積
- **範圍**: 0-100 (越高越好)
- **意義**: 衡量追蹤器在各種重疊閾值下的整體表現
- **您的結果**: 66.90% - 表示在不同重疊閾值下的平均成功率

### 2. OP50 (Overlap Precision at 0.5) - 82.23%
**定義**: 重疊閾值為 0.5 時的成功率
- **意義**: 預測框與真實框的 IoU ≥ 0.5 的幀數百分比
- **您的結果**: 82.23% - 超過 82% 的幀達到了 50% 重疊

### 3. OP75 (Overlap Precision at 0.75) - 60.15%
**定義**: 重疊閾值為 0.75 時的成功率
- **意義**: 預測框與真實框的 IoU ≥ 0.75 的幀數百分比
- **您的結果**: 60.15% - 約 60% 的幀達到了 75% 重疊（更嚴格的標準）

### 4. Precision - 84.91%
**定義**: 距離閾值為 20 pixels 時的精確度
- **意義**: 預測框中心與真實框中心距離 ≤ 20 像素的幀數百分比
- **您的結果**: 84.91% - 約 85% 的幀中心定位誤差小於 20 像素

### 5. Norm Precision - 82.56%
**定義**: 歸一化距離精確度
- **意義**: 考慮目標大小的相對距離精確度
- **您的結果**: 82.56%

### 6. FPS - 205.85
**定義**: 平均處理速度
- **意義**: 每秒處理的幀數
- **您的結果**: 205.85 FPS（從測試時的 *_time.txt 檔案計算）

---

## 🎯 不同資料集的計算

### 計算 UAV123
```bash
python tracking/calculate_metrics.py --dataset uav123
```

### 計算 UAV123_10fps
```bash
python tracking/calculate_metrics.py --dataset uav123_10fps
```

### 計算其他資料集
```bash
# OTB100
python tracking/calculate_metrics.py --dataset otb

# LaSOT
python tracking/calculate_metrics.py --dataset lasot

# GOT-10k
python tracking/calculate_metrics.py --dataset got10k_test
```

---

## 📂 生成圖表

### 使用 --plot 參數生成 PDF 圖表

```bash
python tracking/calculate_metrics.py --dataset uav123 --plot
```

這會在 `output/test/result_plots/uav123/` 目錄下生成：
- `success_plot.pdf` - Success Plot (AUC 曲線)
- `precision_plot.pdf` - Precision Plot
- `norm_precision_plot.pdf` - Normalized Precision Plot

---

## 🔧 論文原始工具

### 方法 1: 使用我建立的簡化腳本（推薦）
```bash
python tracking/calculate_metrics.py --dataset uav123
```

### 方法 2: 使用論文原始的 analysis_results.py

編輯 `tracking/analysis_results.py`:
```python
import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'uav123'  # 修改這裡

trackers.extend(trackerlist(name='sglatrack', parameter_name='deit_distilled', 
                            dataset_name=dataset_name, run_ids=None, 
                            display_name='sglatrack_deit'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=True, 
             plot_types=('success', 'norm_prec', 'prec'))
```

然後執行：
```bash
python tracking/analysis_results.py
```

---

## 📋 參數說明

| 參數 | 預設值 | 說明 | 範例 |
|------|--------|------|------|
| `--tracker` | sglatrack | 追蹤器名稱 | --tracker sglatrack |
| `--param` | deit_distilled | 參數配置名稱 | --param deit_distilled |
| `--dataset` | uav123 | 資料集名稱 | --dataset uav123 |
| `--plot` | False | 是否生成圖表 | --plot |

---

## 💡 使用範例

### 1. 基本計算（僅顯示數值）
```bash
python tracking/calculate_metrics.py --dataset uav123
```

### 2. 計算並生成圖表
```bash
python tracking/calculate_metrics.py --dataset uav123 --plot
```

### 3. 計算 UAV123_10fps
```bash
python tracking/calculate_metrics.py --dataset uav123_10fps
```

### 4. 比較兩個資料集
```bash
# 計算 UAV123
python tracking/calculate_metrics.py --dataset uav123

# 計算 UAV123_10fps
python tracking/calculate_metrics.py --dataset uav123_10fps
```

---

## 📊 您的測試結果總結

### UAV123 資料集

| 指標 | 數值 | 說明 |
|------|------|------|
| **AUC** | 66.90% | 整體追蹤成功率 |
| **OP50** | 82.23% | 重疊 ≥ 50% 的成功率 |
| **OP75** | 60.15% | 重疊 ≥ 75% 的成功率 |
| **Precision** | 84.91% | 中心距離 ≤ 20px 的精確度 |
| **Norm Precision** | 82.56% | 歸一化精確度 |
| **FPS** | 205.85 | 處理速度 |

---

## 🔍 與論文對比

論文中報告的結果通常包含：
- **AUC**: 主要指標，用於 Success Plot
- **Precision**: 用於 Precision Plot
- **FPS**: 速度指標

您現在可以用相同的工具計算出您自己的結果！

---

## ⚠️ 注意事項

### 1. 必須先執行測試
在計算指標之前，必須先執行測試生成結果檔案：
```bash
python tracking/test.py sglatrack deit_distilled \
  --dataset_name uav123 \
  --threads 4 \
  --num_gpus 1
```

### 2. 結果檔案位置
指標計算依賴於測試結果，位於：
```
output/test/tracking_results/sglatrack/deit_distilled/uav123/*.txt
```

### 3. 快取機制
計算結果會快取到：
```
output/test/result_plots/uav123/eval_data.pkl
```

如果想重新計算，可以刪除這個檔案。

---

## 🎉 總結

### ✅ 論文提供的工具
- **檔案**: `lib/test/analysis/plot_results.py`
- **函數**: `print_results()`, `plot_results()`
- **功能**: 計算 AUC、Precision、OP50、OP75 等

### ✅ 我建立的簡化腳本
- **檔案**: `tracking/calculate_metrics.py`
- **功能**: 更簡單易用的命令列介面
- **優點**: 參數清晰、輸出格式化、支援多資料集

### 🚀 立即使用
```bash
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla
python tracking/calculate_metrics.py --dataset uav123
```

### 📊 您的結果
- AUC: **66.90%**
- Precision: **84.91%**
- FPS: **205.85**

完美！✨
