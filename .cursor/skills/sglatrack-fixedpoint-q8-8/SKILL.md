---
name: sglatrack-fixedpoint-q8-8
description: >-
  以 fixed-point（Q8.8）方式改寫 SGLATrack vit 變體，透過大量插入
  to_fixed_point 截斷中間特徵、attention、MLP 與殘差路徑。當使用者要求
  fixed-point 量化、Q8.8 模擬，或要對齊 reference/rongxuan/02_FixedPoint
  的硬體友善流程時使用。
---

# SGLATrack 定點量化 Q8.8

使用此 skill 將 `vit_*.py` 轉成以 **Q8.8** 為核心的 fixed-point 版本。

主要參考：
- `reference/rongxuan/02_FixedPoint/My_model.py`
- `reference/rongxuan/02_FixedPoint/predict.py`

此 skill 採 **fixed-point first**，不要求先做 QAT/PTQ。

## 目標

對每個目標模型：
1. 保持原始架構與 `state_dict` 關鍵鍵名相容。
2. 在關鍵運算邊界插入 `to_fixed_point(..., 8, 8)`。
3. 以硬體友善方式量化權重與 activation。
4. 驗證 train/infer 可執行，並整理精度影響。

## 預設數值策略

- Activation 格式：**Q8.8**（`int_bits=8`, `frac_bits=8`）
- 權重格式（預設）：Conv/Linear 使用 **Q1.8**（除非使用者另行指定）
- 取整方式：`round`
- 飽和策略：`clamp` 到固定點範圍

統一使用以下 helper：

```python
def to_fixed_point(tensor, int_bits=8, frac_bits=8):
    scale = 2 ** frac_bits
    min_val = -2 ** (int_bits - 1)
    max_val = 2 ** (int_bits - 1) - 1 / scale
    tensor_fp = torch.round(tensor * scale) / scale
    return torch.clamp(tensor_fp, min_val, max_val)
```

## 必要流程

### 步驟 1：先複製指定檔案並改名（必做）

第一步固定執行：
- 使用者指定來源檔（例如 `vit_CARE_relu6.py`）
- 先複製成同目錄新檔，檔名必須是 **`原檔名_fixed.py`**
  - 例：`vit_CARE_relu6.py` -> `vit_CARE_relu6_fixed.py`
- 後續所有修改都在 `*_fixed.py` 上進行，不直接改原檔

### 步驟 2：讀原始結構並標出插點位置

檢查 `*_fixed.py` 並列出：
- Patch embed 輸出
- Token concat / positional embedding 加法
- 每個 transformer block 的邊界
- Attention 內部路徑（`q/k/v`、logits、normalizer、`attn @ v`、`proj`）
- MLP 內部路徑（`fc1`、act、`fc2`）
- Residual add 輸出
- 最終 norm / head 輸入

### 步驟 3：插入截斷點（必做）

至少要在以下位置後面加上 `to_fixed_point(..., 8, 8)`：
- patch embedding 與 positional add
- `norm1` 輸出
- `q`、`k`、`v` projection 輸出
- attention scaling/logits 與 normalized attention
- attention output projection
- 第一次 residual add
- `norm2` 輸出
- MLP 的 `fc1`、activation、`fc2`
- 第二次 residual add
- 最終 backbone 輸出（進入 prediction heads 前）

若某個位置刻意不插，需在程式附近補簡短註解說明原因。

### 步驟 4：Attention 專屬規範

- 優先使用硬體友善 normalizer（如 `softmax_taylor`、linear attention 型正規化），避免 exp softmax。
- 若有倒數/除法（`1 / denom`），必須保留 epsilon，並在以下位置截斷：
  - 倒數前
  - 倒數後
  - 倒數結果套用到數值後
- matmul 邊界附近要保持高密度截斷。

### 步驟 5：權重量化工具

提供類似 `reference/rongxuan/02_FixedPoint/predict.py` 的推論工具：
- 走訪所有 modules
- 用 `to_fixed_point` 量化 Conv2d / Linear 權重（有 bias 也要處理）
- 列印逐層量化格式，確保可追蹤

### 步驟 6：建立對應 yaml 並更新 sglatrack

建立新變體時固定執行：
- 模型檔使用步驟 1 產生的 `原檔名_fixed.py`
- 新增對應實驗 yaml（從來源 yaml 複製並修改 backbone type 指向 fixed 版本）
- 在 `python/lib/models/sglatrack/sglatrack.py` 註冊 import 與 branch（對應 fixed 版本 builder）

模型參數名稱需保持不變（`qkv`、`proj`、`mlp.fc1`、`mlp.fc2` 等），以維持 checkpoint 載入相容性。

### 步驟 7：驗證清單

- Forward pass shape 檢查（template/search）
- Backward pass 檢查（loss backward 可正常執行）
- 套用量化權重後的 inference 檢查
- 確認 attention + MLP + residual 路徑確實有呼叫截斷
- 回報哪些 tensor 用 Q8.8、哪些有自訂格式覆寫

## 實作樣板（範例）

```python
# block forward 內部範例
x_norm = self.norm1(x)
x_norm = to_fixed_point(x_norm, 8, 8)

attn_out = self.attn(x_norm)
attn_out = to_fixed_point(attn_out, 8, 8)

x = x + attn_out
x = to_fixed_point(x, 8, 8)

mlp_in = self.norm2(x)
mlp_in = to_fixed_point(mlp_in, 8, 8)
mlp_out = self.mlp(mlp_in)
mlp_out = to_fixed_point(mlp_out, 8, 8)

x = x + mlp_out
x = to_fixed_point(x, 8, 8)
```

## 常見錯誤（避免）

- 目標混用：宣稱 Q8.8，最終卻轉成純 INT8 格式。
- 只量化權重，未量化中間 activation。
- residual add 後漏插截斷。
- 改動層名稱導致 pretrained checkpoint 無法對應。
- bit 格式前後不一致，且未註明覆寫位置與理由。

## 每次轉換的交付項目

1. `原檔名_fixed.py`（含 Q8.8 截斷點）
2. 對應更新後的實驗 yaml
3. `sglatrack.py` 的 fixed 版本註冊內容
4. 簡短報告：
   - 已插入的截斷位置
   - 權重量化設定
   - smoke test 結果
   - 已知數值風險點（若有）
