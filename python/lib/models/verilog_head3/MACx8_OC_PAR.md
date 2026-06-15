# verilog_head2：Conv2D MAC×8（OC_PAR=8）實作說明

> 對應 RTL：`conv.v`（通用 conv 模組）+ `head_top.v`（`.OC_PAR(8)` 實例化）  
> 軟體對照：`run_backbone_numpy_shared_trunk.py` → `conv2d()` + bias + ReLU + `fp()`（Q8.8）

---

## 1. 目標

在同一個輸出空間座標 `(oh, ow)` 上，**一次計算 8 個 output channel（OC）**：

- 8 路權重／累加器 **平行**（同拍更新 `acc_r[0:7]`）
- 輸入 activation **共用**（同一 `mac_feat` 只讀一份 `x_i`）
- 對外寫 feature map 仍為 **每拍 1 個 `(oc, oh, ow)`**（SAT 階段用 `sat_lane` 串列 8 拍）

`head_top` 中 **conv1、conv2** 皆設 `OC_PAR = 8`。`tail` 的 `tail_unit` 仍為單 OC（與本設計無關）。

---

## 2. 參數（head_top 實例）

| 項目 | conv1 (`u_conv1`) | conv2 (`u_conv2`) |
|------|-------------------|-------------------|
| IN_CH | 32 | 48（C_SH1） |
| OUT_CH | 96 | 48 |
| K×K / PAD | 3×3 / 1 | 3×3 / 1 |
| OUT_H×OUT_W | 16×16 | 16×16 |
| **OC_PAR** | **8** | **8** |
| FEAT_PER_OC | 32×9 = **288** | 48×9 = **432** |
| ROM_PROFILE | 1 | 2 |

- `oc_base_r`：目前 8 路 MAC 對應的 **起始 OC**（0, 8, 16, …）
- `oc_last`：`(oc_base_r + OC_PAR >= OUT_CH)`
- 有效 lane：`oc_base_r + lane < OUT_CH`（`sat_lane_valid`）

---

## 3. 儲存體（8 路平行）

```verilog
reg signed [DATA_W-1:0] wgt_buf [0:OC_PAR-1][0:FEAT_PER_OC-1];  // 目錄內唯一 2D reg
reg signed [DATA_W-1:0] bias_r  [0:OC_PAR-1];
reg signed [ACC_W-1:0]  acc_r   [0:OC_PAR-1];
reg signed [ACC_W-1:0]  acc_sat_r [0:OC_PAR-1];
```

| 陣列 | 意義 |
|------|------|
| `wgt_buf[lane][feat]` | OC = `oc_base_r + lane` 的第 `feat` 個卷積權重 |
| `acc_r[lane]` | 該 OC 在當前 `(oh_r, ow_r)` 的累加和 |
| `bias_r[lane]` | 該 OC 的 bias（WPRE 自 ROM 載入） |

**組合邏輯（每拍 unroll，lane 0..7）：**

```verilog
mac_w_op[i_lane]  = wgt_buf[i_lane][mac_feat];
mac_prod[i_lane]  = mac_x_op * mac_w_op[i_lane];
acc_next[i_lane]  = acc_r[i_lane] + mac_prod[i_lane];
```

`mac_x_op`：8 路共用；padding 時為 0（`mac_xi_pad_r` 鎖存）。

---

## 4. FSM

```
S_IDLE → S_WPRE → S_MAC → S_SAT → … → S_DONE → S_IDLE
```

| 狀態 | 功能 |
|------|------|
| **S_WPRE** | 權重 ROM 2-phase 預取 → 填滿 `wgt_buf[0:7][*]`；再預取 8 個 bias → `bias_r[0:7]` |
| **S_MAC** | 固定 `(oh_r, ow_r)`、`oc_base_r..+7`，對 FEAT_PER_OC 個 tap 做 pipeline MAC |
| **S_SAT** | 定點縮放 + bias + 飽和 + ReLU，`sat_lane` 0→7 各輸出 1 個 OC |

空間與 OC 推進（SAT 結束且非 lane 內循環時）：

- `ow`++；行末 `oh`++、`ow`←0
- 行末且尚有 OC：`oc_base_r += 8`，回到 **S_WPRE**
- 否則回到 **S_MAC**（同一組 OC 的下一個 pixel）

---

## 5. WPRE：8 lane 填滿 wgt_buf

**ROM 時序**（`conv.v` 註解）：posedge **T** 送 `w_addr` / `b_addr` → **T+1** `w_i` / `rom_c12b_q` 有效 → 寫 buffer。

二相預取：

1. `wpre_phase=0`：`w_addr_r = (oc_base_r + wpre_lane) * FEAT_PER_OC + wpre_feat`
2. `wpre_phase=1`：`wgt_buf[wpre_lane][wpre_feat] <= w_i`

排程：**固定 `wpre_feat`，先掃 `wpre_lane` 0→7**，再 `wpre_feat++`。全部 feat 完成 → `wpre_done`；接著 bias 預取（`b_addr = oc_base_r + bpre_lane`，寫 `bias_r[bpre_lane]`）。

下一 pixel／下一組 OC 前，在 `wpre_sat_wrap` 清除 `wpre_done` / `bpre_done` 再重載。

---

## 6. MAC：1-phase pipeline

相對舊版 **2-phase**（phase0 讀、phase1 算），本版為 **1-phase**：

| 概念 | 說明 |
|------|------|
| `mac_fill` | 第一拍只發 feat0 讀址，不累加 |
| 其後每拍 | 用上一拍讀回的 `x_i` 做 MAC，同拍發 feat+1 讀址 |
| `mac_phase_o` | `(CS==S_MAC) && !mac_done && !mac_fill` → parent 本拍接 `x_i` |
| `mac_active_o` | `(CS==S_MAC)` → parent MAC 期間每拍可發 SRAM read |

讀址：

- 當前 MAC：`mac_feat` → `(ic, kh, kw)` → NCHW `x_addr_nxt`
- Pipeline 讀：`mac_feat+1`（`x_addr_mac_rd`）；fill 拍讀 feat0

**每個 (oh, ow, oc_base) 的 MAC 拍數** ≈ `1 + FEAT_PER_OC`（conv1：289；conv2：433）。**8 個 OC 共用** 同一串 x 讀取，不 ×8。

`mac_done` 時：`acc_sat_r[lane] <= acc_next[lane]`（僅有效 OC）。

---

## 7. SAT：8 拍串列輸出

- `sat_lane` 0..7：組合邏輯選 `acc_sat_r[sat_lane]`、`bias_r[sat_lane]`
- `>>> FRAC_W`（可選 `ROUND_Y`）→ +bias → saturate → ReLU
- `y_valid=1`：`y_oc = oc_base_r + sat_lane`，`y_oh/ow = oh_r/ow_r`
- FSM：`sat_lane < OC_PAR-1` 留 S_SAT；否則依 spatial / OC 邊界轉 WPRE / MAC / DONE

---

## 8. head_top 與 SRAM

### conv1 → Sram_tok1

- `c1_mac_active`：S_MAC 期間每拍可讀
- `c1_mac_phase`：非 fill 時 `tok1_q` → `c1_x_i_mac`
- 位址（token-major）：  
  `(LENS_Z + x_addr_mac[7:0]) * IN_CH + x_addr_mac[12:8]`
- 權重在 conv 內部 ROM + `wgt_buf`，不走 tok1

### conv2 → sh1（lo/hi bank）

- `c2_mac_active`：每拍發 sh1 讀
- `c2_mac_phase`：`sh1_rd_q` → `c2_x_i_mac`
- **stall**：`c2_busy && c2_mac_active && (c1_rows_ready < c2_need_rows)`，等 conv1 產出足夠行

### 輸出

- conv1 `y_valid` → 寫 sh1（依 `y_oc` 分 bank）
- conv2 `y_valid` → 寫 sh2 → tail

---

## 9. Parent 介面

| 信號 | 方向 | 說明 |
|------|------|------|
| `x_addr_mac_rd` | out | MAC 期 activation 讀址（含 pipeline） |
| `mac_active_o` | out | 在 S_MAC，可發 read |
| `mac_phase_o` | out | 本拍 `x_i` 有效 |
| `x_i` | in | parent 依 `mac_phase_o` 選通 SRAM Q |
| `stall` | in | conv2 用；停 FSM（`cs_en=0`） |

---

## 10. OC_PAR=1 vs OC_PAR=8

| 項目 | OC_PAR=1 | OC_PAR=8 |
|------|----------|----------|
| wgt_buf | 1×FEAT | **8×FEAT（2D）** |
| 每 pixel MAC 拍數 | 1+FEAT | **相同**（x 只讀一份） |
| WPRE ROM 讀 | ∝ FEAT | **∝ 8×FEAT**（每 feat 8 個 OC） |
| 每 pixel 輸出 | 1 拍 | **最多 8 拍（SAT）** |
| OC 步進 | +1 | **oc_base +8** |

**取捨**：以 on-chip `wgt_buf` 與較長 WPRE，換 MAC 階段 **activation 讀取不隨 OUT_CH 線性增加**。

---

## 11. 數值路徑（單 lane）

1. `acc_sat_r[lane]`（int32）
2. `>>> FRAC_W`（`ROUND_Y` 可開四捨五入）
3. `+ bias_r[lane]`
4. saturate → Q8.8
5. ReLU（`HAS_RELU=1`）
6. `y_data` / `y_oc = oc_base_r + lane`

---

## 12. Cycle 粗估（單組 8 OC、單一 pixel）

設 ROM/WPRE 為 2 cycle／read，MAC 為 1 cycle／feat（含 1 fill）：

| 階段 | conv1（288 feat） | conv2（432 feat） |
|------|-------------------|-------------------|
| WPRE 權重 | 288×8×2 = 4608 | 432×8×2 = 6912 |
| WPRE bias | 8×2 = 16 | 16 |
| MAC | 1+288 = 289 | 1+432 = 433 |
| SAT | ≤8 | ≤8 |

實際總 cycle 還含 FSM 轉換、conv1/conv2 重疊 stall 等；以上僅供數量級參考。

---

## 13. 一句話摘要

在每個 `(oh, ow)` 上，**WPRE** 載入 8 個 OC 的權重與 bias 到 `wgt_buf` / `bias_r`，**MAC** 用一份 input 的 pipeline 讀取同時更新 `acc_r[0:7]`，**SAT** 用 `sat_lane` 串列 8 拍輸出；parent 以 `mac_active_o` / `mac_phase_o` 對齊 SRAM 讀時序。

---

## 14. 相關檔案

| 檔案 | 說明 |
|------|------|
| `conv.v` | `OC_PAR` 參數化；預設 1，head 覆寫為 8 |
| `head_top.v` | `u_conv1` / `u_conv2` → `.OC_PAR(8)` |
| `tail.v` | 單 OC，無 ×8 |
