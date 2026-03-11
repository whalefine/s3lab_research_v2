# 📊 FPS 計算結果

## ✅ 您的測試結果

### 🎯 平均 FPS: **187.38 FPS**

---

## 📈 詳細統計

### 總體資訊
- **總幀數**: 112,578 幀
- **測試序列數**: 123 個
- **資料集**: UAV123
- **配置**: deit_distilled

### 處理時間統計
| 指標 | 時間 |
|------|------|
| 平均處理時間 | 5.34 ms |
| 標準差 | 7.86 ms |
| 最快 | 0.30 ms |
| 最慢 | 1201.46 ms |

### FPS 統計
| 指標 | FPS |
|------|-----|
| **平均 FPS** | **187.38** |
| 最小 FPS | 0.83 |
| 最大 FPS | 3311.26 |

### 百分位數
| 百分位 | FPS | 時間 |
|--------|-----|------|
| 50th (中位數) | 203.79 | 4.91 ms |
| 95th | 129.45 | 7.73 ms |
| 99th | 78.77 | 12.70 ms |

---

## 🏆 表現分析

### ⚡ 最快的序列 (Top 5)
1. **wakeboard5** - 314.74 FPS
2. **wakeboard6** - 296.96 FPS
3. **wakeboard4** - 289.92 FPS
4. **wakeboard9** - 271.39 FPS
5. **wakeboard3** - 220.59 FPS

### 🐌 最慢的序列 (Bottom 5)
1. **car4_s** - 128.93 FPS
2. **truck1** - 125.70 FPS
3. **car11** - 114.72 FPS
4. **bird1_3** - 106.08 FPS
5. **car1_s** - 106.07 FPS

---

## 🎯 與論文對比

根據 SGLATrack 論文，報告的速度是在 **RTX 2080Ti** 上測試的。

您的結果顯示平均 **187.38 FPS** 的追蹤速度，這是一個非常好的即時追蹤性能！

---

## 📋 如何使用 FPS 計算腳本

### 基本用法
```bash
conda activate sgla
python calculate_fps.py
```

### 指定參數
```bash
# 計算特定配置的 FPS
python calculate_fps.py --config coco_uav123 --dataset uav123

# 計算 UAV123_10fps 的 FPS
python calculate_fps.py --dataset uav123_10fps

# 指定自訂結果目錄
python calculate_fps.py --results_dir output/test/tracking_results/sglatrack/deit_distilled/uav123
```

### 參數說明
- `--tracker`: 追蹤器名稱 (預設: sglatrack)
- `--config`: 配置名稱 (預設: deit_distilled)
- `--dataset`: 資料集名稱 (預設: uav123)
- `--results_dir`: 自訂結果目錄路徑

---

## 📁 結果檔案

詳細的 FPS 統計已儲存到：
```
output/test/tracking_results/sglatrack/deit_distilled/uav123/fps_statistics.txt
```

該檔案包含：
- 總體統計資訊
- 每個序列的 FPS
- 百分位數分析
- 處理時間統計

---

## 🔍 其他測試資料集

如果您測試了其他資料集，也可以計算 FPS：

```bash
# UAV123_10fps
python calculate_fps.py --dataset uav123_10fps

# UAVDT
python calculate_fps.py --dataset uavdt

# 其他配置的結果
python calculate_fps.py --config coco_uav123 --dataset uav123
```

---

## 💡 理解 FPS 結果

### 為什麼有些幀特別慢？
- 第一幀通常較慢（初始化）
- 某些複雜場景可能需要更多計算時間
- GPU 預熱時間影響

### 中位數 vs 平均值
- **中位數 (203.79 FPS)**: 代表典型的追蹤速度
- **平均值 (187.38 FPS)**: 包含所有幀，包括初始化較慢的幀

### 95th 百分位數
**129.45 FPS** 表示 95% 的幀都能達到這個速度或更快，這是衡量穩定性能的好指標。

---

## 🚀 優化建議

如果想要提升 FPS:

1. **減少輸入解析度**
   - 修改 `TEST.TEMPLATE_SIZE` 和 `TEST.SEARCH_SIZE`

2. **使用混合精度**
   - 啟用 AMP (Automatic Mixed Precision)

3. **批次處理**
   - 如果測試多個序列，可以使用批次處理

4. **模型剪枝**
   - 減少模型參數量

---

## 📊 效能總結

✅ **即時追蹤**: 187.38 FPS 遠超即時要求 (30 FPS)
✅ **穩定性**: 95% 的幀達到 129+ FPS
✅ **適用場景**: 適合 UAV 即時追蹤應用

您的模型在 UAV123 測試集上表現出色！🎉

---

## 🛠️ 腳本位置

- **FPS 計算腳本**: `calculate_fps.py`
- **使用說明**: 本檔案
- **結果檔案**: `output/test/tracking_results/.../fps_statistics.txt`
