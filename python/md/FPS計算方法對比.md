# 📊 計算 FPS 的兩種方法

## 方法 1: 論文官方提供的速度測試 (推薦用於論文對比)

### 📝 來自 README
論文在第 142-147 行提到：

```
## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

python tracking/profile_model.py
```

### 🚀 使用方法

#### 基本用法
```bash
conda activate sgla

# 測試 deit_distilled 配置
python tracking/profile_model.py --config deit_distilled

# 測試其他配置
python tracking/profile_model.py --config coco_uav123
```

### 📊 輸出結果
這個腳本會測試：
- **FLOPs** (浮點運算次數)
- **參數量** (模型大小)
- **平均延遲** (毫秒)
- **理論 FPS** (基於合成數據)

**測試方式**:
- 使用隨機生成的輸入數據
- 預熱 200 次
- 測試 500 次取平均
- 純粹測試模型推理速度

**優點**:
- ✅ 官方標準方法
- ✅ 與論文報告的速度可直接對比
- ✅ 測試純推理速度（不包含 I/O）

**缺點**:
- ⚠️ 使用合成數據，不是真實場景
- ⚠️ 不包含圖像載入、後處理等開銷

---

## 方法 2: 我建立的實際測試 FPS 計算 (推薦用於實際應用)

### 🎯 特點
這個方法計算**實際測試時的真實 FPS**。

### 🚀 使用方法

```bash
conda activate sgla

# 基本用法（使用預設參數）
python calculate_fps.py

# 指定資料集
python calculate_fps.py --dataset uav123
python calculate_fps.py --dataset uav123_10fps

# 指定配置
python calculate_fps.py --config deit_distilled --dataset uav123
python calculate_fps.py --config coco_uav123 --dataset uav123

# 自訂結果目錄
python calculate_fps.py --results_dir output/test/tracking_results/sglatrack/deit_distilled/uav123
```

### 📊 輸出結果
- **平均 FPS**: 基於所有測試幀
- **中位數 FPS**: 排除極端值
- **百分位數**: 95th, 99th
- **每個序列的 FPS**
- **最快/最慢序列分析**

**優點**:
- ✅ 真實測試場景的 FPS
- ✅ 包含完整的追蹤流程
- ✅ 可分析不同序列的性能
- ✅ 統計資訊豐富

**缺點**:
- ⚠️ 包含 I/O 開銷（圖像載入等）
- ⚠️ 第一幀通常較慢（初始化）

---

## 🔍 兩種方法的對比

| 特性 | 官方 profile_model.py | 我的 calculate_fps.py |
|------|----------------------|----------------------|
| **數據來源** | 合成隨機數據 | 真實測試結果 |
| **測試內容** | 純推理速度 | 完整追蹤流程 |
| **結果用途** | 論文對比 | 實際應用評估 |
| **輸出** | FLOPs + 理論FPS | 實際FPS統計 |
| **需要資料集** | ❌ 不需要 | ✅ 需要測試結果 |
| **執行時間** | ~1 分鐘 | ~幾秒 |

---

## 📋 完整流程建議

### 步驟 1: 測試理論速度（論文方法）
```bash
conda activate sgla
python tracking/profile_model.py --config deit_distilled
```

**預期輸出**:
```
overall macs is XXX
overall params is XXX
testing speed ...
The average overall latency is XX.XX ms
FPS is XXX.XX fps
```

### 步驟 2: 執行實際測試
```bash
python tracking/test.py sglatrack deit_distilled \
  --dataset_name uav123 --threads 4 --num_gpus 1
```

### 步驟 3: 計算實際 FPS
```bash
python calculate_fps.py --config deit_distilled --dataset uav123
```

**您的結果**: 平均 **187.38 FPS**

---

## 🎯 您目前的結果

### 實際測試 FPS (方法 2)
- ✅ 已執行: `calculate_fps.py`
- ✅ 結果: **187.38 FPS** (UAV123)
- ✅ 基於: 112,578 幀的真實測試

### 理論 FPS (方法 1)
- ⏳ 尚未執行: `profile_model.py`
- 💡 建議執行以獲得理論上限

---

## 🚀 立即執行官方速度測試

現在就可以執行論文的官方速度測試：

```bash
conda activate sgla
python tracking/profile_model.py --config deit_distilled
```

這會測試：
- ✅ FLOPs (計算複雜度)
- ✅ 參數量
- ✅ 理論 FPS (在您的 GPU 上)

執行時間約 1-2 分鐘。

---

## 📊 結果對比

執行完兩個腳本後，您會有：

1. **理論 FPS** (profile_model.py)
   - 純推理速度上限
   - 用於論文對比

2. **實際 FPS** (calculate_fps.py) = **187.38**
   - 真實應用速度
   - 包含完整流程

通常：**理論 FPS > 實際 FPS**

這是正常的，因為實際應用包含：
- 圖像載入和預處理
- 邊界框後處理
- 結果保存等開銷

---

## 💡 建議

### 如果您要寫論文/報告：
- 使用**兩種方法**的結果
- 理論 FPS: 展示模型效率
- 實際 FPS: 展示實用性

### 如果您要實際部署：
- 主要關注**實際 FPS** (187.38)
- 這更能反映真實性能

---

## 📁 相關檔案

1. **論文官方**: `tracking/profile_model.py`
2. **我建立的**: `calculate_fps.py`
3. **結果文件**: `FPS計算結果.md`
4. **測試結果**: `output/test/tracking_results/.../fps_statistics.txt`

---

## 🎉 總結

✅ **論文有提供**: `profile_model.py` (理論速度)
✅ **我額外建立**: `calculate_fps.py` (實際速度)
✅ **您已有結果**: 187.38 FPS (實際)
💡 **建議**: 再執行 `profile_model.py` 獲得理論速度

兩種方法都很有用，建議都執行一次！
