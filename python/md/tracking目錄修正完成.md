# ✅ calculate_fps.py 已移至 tracking/ 目錄

## 📁 文件位置

```
SGLATrack-main/
├── tracking/
│   ├── calculate_fps.py  ✅ 新位置
│   ├── test.py
│   └── profile_model.py
└── ...
```

---

## 🔧 修正內容

### 1. 路徑調整
由於腳本從根目錄移到 `tracking/` 子目錄，需要調整專案路徑：

**修正的代碼** (第 10-12 行):
```python
# 將專案根目錄加入 Python 路徑（因為腳本在 tracking/ 目錄下）
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
```

**說明**: 
- `os.path.dirname(__file__)` → `tracking/` 目錄
- 加上 `..` → 回到專案根目錄
- 這樣才能正確導入 `lib` 模組

### 2. 導入語句保持不變
```python
from lib.test.evaluation.environment import env_settings
```

導入路徑無需修改，因為我們已經將專案根目錄加入 `sys.path`。

---

## 🚀 使用方式

### 基本用法

```bash
cd /home/junjie/01_Research/SGLATrack-main
conda activate sgla

# 計算 UAV123 的 FPS
python tracking/calculate_fps.py --dataset uav123
```

### 完整參數

```bash
python tracking/calculate_fps.py \
  --tracker sglatrack \
  --param deit_distilled \
  --dataset uav123
```

---

## 📊 測試結果

執行命令後的輸出：

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

## 🎯 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--tracker` | sglatrack | 追蹤器名稱 |
| `--param` | deit_distilled | 參數配置名稱 |
| `--dataset` | uav123 | 資料集名稱 |
| `--runid` | None | 可選的運行 ID |

---

## 💡 使用範例

### 1. 計算 UAV123 的 FPS
```bash
python tracking/calculate_fps.py --dataset uav123
```

### 2. 計算 UAV123_10fps 的 FPS
```bash
# 先執行測試
python tracking/test.py sglatrack deit_distilled \
  --dataset_name uav123_10fps --threads 4 --num_gpus 1

# 再計算 FPS
python tracking/calculate_fps.py --dataset uav123_10fps
```

### 3. 指定不同的追蹤器和參數
```bash
python tracking/calculate_fps.py \
  --tracker sglatrack \
  --param deit_distilled \
  --dataset uav123
```

---

## ✅ 修正驗證

### 測試結果
- ✅ 腳本已移至 `tracking/` 目錄
- ✅ 路徑設定已修正
- ✅ 導入模組正常運作
- ✅ 功能測試通過
- ✅ 輸出結果正確

### 執行確認
```bash
# 確認文件位置
ls tracking/calculate_fps.py
# 輸出: tracking/calculate_fps.py

# 測試執行
python tracking/calculate_fps.py --dataset uav123
# 輸出: 平均 FPS: 187.38
```

---

## 🔍 與其他 tracking/ 腳本的一致性

現在 `calculate_fps.py` 與其他 tracking 腳本的結構一致：

```bash
tracking/
├── test.py                    # 測試腳本
├── profile_model.py           # 模型性能分析
├── calculate_fps.py           # FPS 計算 ✅ 新增
└── ...
```

所有腳本都使用相同的路徑處理方式：
```python
prj_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(prj_path)
```

---

## 📝 快速命令參考

```bash
# 基本用法
python tracking/calculate_fps.py --dataset uav123

# 完整用法
python tracking/calculate_fps.py --tracker sglatrack --param deit_distilled --dataset uav123

# 查看幫助
python tracking/calculate_fps.py --help
```

---

## 🎉 完成總結

### 修改內容
1. ✅ 將 `calculate_fps.py` 從根目錄移至 `tracking/` 目錄
2. ✅ 調整專案路徑設定以適應新位置
3. ✅ 保持導入語句和功能不變
4. ✅ 刪除根目錄的舊檔案

### 測試結果
- ✅ 腳本可以正常執行
- ✅ 正確計算 FPS (187.38)
- ✅ 與其他 tracking 腳本結構一致

現在您可以使用 `python tracking/calculate_fps.py` 來計算 FPS 了！🎉
