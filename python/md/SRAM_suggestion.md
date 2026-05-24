# RTL → SRAM 改造建議（backbone + head + 整合版 verilog2）

> **規格**：Q8.8（16-bit signed），全專案共用。
> **規則依據**：[.cursor/rules/verilog_rule.mdc](../../.cursor/rules/verilog_rule.mdc)、[.cursor/rules/numpy-trunk-to-verilog.mdc](../../.cursor/rules/numpy-trunk-to-verilog.mdc)
> **判斷門檻**（全文件通用）：
>
> | 等級 | depth | 處理建議 |
> |------|-------|----------|
> | **必改 SRAM** | depth ≥ 1024（≥ 16 Kb / 2 KB 起跳） | 使用 1R1W 或單埠 SRAM macro |
> | **建議 SRAM** | 512 ≤ depth < 1024 | 視製程 macro 表決定，通常 ≥ 256 word 即可省面積 |
> | **可留 reg** | depth < 256 | flip-flop register file 即可 |
>
> 容量計算：**bits = depth × width**，**KB = bits / 8 / 1024**。

---

## 工作流總覽

| 目錄 | 角色 | SRAM 改造作為 |
|------|------|--------------|
| [python/lib/models/verilog_backbone2/](../lib/models/verilog_backbone2/) | **分模組除錯**：backbone 獨立驗證 | Phase 1：先在此改 SRAM port，獨立通過對拍 |
| [python/lib/models/verilog_head2/](../lib/models/verilog_head2/) | **分模組除錯**：head 獨立驗證 | Phase 2：先在此改 SRAM port，獨立通過對拍 |
| [python/lib/models/verilog2/](../lib/models/verilog2/) | **整合版**：backbone + head 端到端 | Phase 3：同步上述改動，加入 sglatrack_top 共享 SRAM mux |

> **本文件結構**：第一、二部分為 verilog_backbone2 / verilog_head2 的獨立分析（debug 階段參考），**第三部分為最終整合方案（verilog2，採用 Unified Shared SRAM Pool）**。

---

# 第一部分：Backbone（`verilog_backbone2/`）— 分模組除錯版

> **目錄**：[python/lib/models/verilog_backbone2/](../lib/models/verilog_backbone2/)
> **常數**：`EMBED_DIM=32`、`NUM_HEADS=4`、`HEAD_DIM=8`、`N_TOKENS=320`、`MLP_DIM=128`
> **注意**：此版為**分模組除錯目錄**，假設每個 transformer_block 各自有獨立 hardware copy（×7）。整合到 verilog2 後改為**單一 instance reuse 7 次**（見第三部分）。

## 1.1 各檔案需改成 SRAM 的 `reg` 訊號

### A. [backbone_top.v](../lib/models/verilog_backbone2/backbone_top.v) — **必改 SRAM**

| 訊號 (行號) | 宣告 | depth × width | 容量 |
|-------------|------|---------------|------|
| `tok_buf` [backbone_top.v:83](../lib/models/verilog_backbone2/backbone_top.v#L83) | `reg signed [15:0] tok_buf [0:N_TOKENS*EMBED_DIM-1]` | 10240 × 16 | **20 KB** |
| `out_buf` [backbone_top.v:165](../lib/models/verilog_backbone2/backbone_top.v#L165) | `reg signed [15:0] out_buf [0:N_TOKENS*EMBED_DIM-1]` | 10240 × 16 | **20 KB** |

### B. [transformer_block.v](../lib/models/verilog_backbone2/transformer_block.v) — **必改 SRAM**

| 訊號 (行號) | 宣告 | depth × width | 容量 |
|-------------|------|---------------|------|
| `x_buf` [transformer_block.v:63](../lib/models/verilog_backbone2/transformer_block.v#L63) | `[0:TOK_FLAT-1]` | 10240 × 16 | **20 KB** |
| `tmp_buf` [transformer_block.v:64](../lib/models/verilog_backbone2/transformer_block.v#L64) | `[0:TOK_FLAT-1]` | 10240 × 16 | **20 KB** |

### C. [care_attention.v](../lib/models/verilog_backbone2/care_attention.v) — **大量必改／建議 SRAM**

| 訊號 (行號) | 宣告 | depth × width | 容量 | 等級 |
|-------------|------|---------------|------|------|
| `x_in_buf` [care_attention.v:97](../lib/models/verilog_backbone2/care_attention.v#L97) | `[0:X_ELEMS-1]` | 10240 × 16 | **20 KB** | 必改 |
| `q_buf` [care_attention.v:98](../lib/models/verilog_backbone2/care_attention.v#L98) | `[0:HD_ELEMS-1]` | 10240 × 16 | **20 KB** | 必改 |
| `k_buf` [care_attention.v:99](../lib/models/verilog_backbone2/care_attention.v#L99) | `[0:HD_ELEMS-1]` | 10240 × 16 | **20 KB** | 必改 |
| `v_buf` [care_attention.v:100](../lib/models/verilog_backbone2/care_attention.v#L100) | `[0:HD_ELEMS-1]` | 10240 × 16 | **20 KB** | 必改 |
| `qkm_buf` [care_attention.v:102](../lib/models/verilog_backbone2/care_attention.v#L102) | `[0:QKM_ELEMS-1]` | 1280 × 16 | **2.5 KB** | 建議 |
| `zr_buf` [care_attention.v:103](../lib/models/verilog_backbone2/care_attention.v#L103) | `[0:QKM_ELEMS-1]` | 1280 × 16 | **2.5 KB** | 建議 |
| `ao_buf` [care_attention.v:105](../lib/models/verilog_backbone2/care_attention.v#L105) | `[0:X_ELEMS-1]` | 10240 × 16 | **20 KB** | 必改 |

**可留 reg：** `km_buf` (32)、`kv_buf` (256)。

### D. [mlp.v](../lib/models/verilog_backbone2/mlp.v) — **必改 SRAM**

| 訊號 (行號) | 宣告 | depth × width | 容量 |
|-------------|------|---------------|------|
| `x_in_buf` [mlp.v:71](../lib/models/verilog_backbone2/mlp.v#L71) | `[0:X_ELEMS-1]` | 10240 × 16 | **20 KB** |

**可留 reg：** `fc1_buf` (128 entries, 256 B)。

### E. 可全部留 reg 的檔案

- [linear.v](../lib/models/verilog_backbone2/linear.v)：`x_buf / wgt_buf` 32 entries each（64 B）
- [linear_wide.v](../lib/models/verilog_backbone2/linear_wide.v)：`x_buf / wgt_buf` 128 entries each（256 B；邊緣）
- [layer_norm.v](../lib/models/verilog_backbone2/layer_norm.v)：`feat_buf` 32 entries
- [residual.v](../lib/models/verilog_backbone2/residual.v)、[recip_nr.v](../lib/models/verilog_backbone2/recip_nr.v) 等：純 datapath，無陣列

### F. Testbench 不列入

- [TEST_backbone.v](../lib/models/verilog_backbone2/TEST_backbone.v) 的 `GOLD_BB / TEMPL_MEM / SRCH_MEM`：`$readmemb` 用，不進晶片。

---

# 第二部分：Head（`verilog_head2/`）— 分模組除錯版

> **目錄**：[python/lib/models/verilog_head2/](../lib/models/verilog_head2/)
> **常數**：`EMBED_DIM=32`、`FEAT_SZ=16`（→ MAP_LEN=256、MAP2_LEN=512）、`OC_CONV1=96`、`OC_CONV2=48`、`IN_CH_CONV1=32`、`IN_CH_CONV2=96`、`K=3 (PAD=1)`、tail `IN_CH=48`

## 2.1 各檔案需改成 SRAM 的 `reg` 訊號

### A. [conv.v](../lib/models/verilog_head2/conv.v) — **conv2 建議 SRAM**

| 訊號 (行號) | 實例 | depth × width | 容量 | 等級 |
|-------------|------|---------------|------|------|
| `wgt_buf` [conv.v:132](../lib/models/verilog_head2/conv.v#L132) | conv1 (IN_CH=32) | 288 × 16 | 576 B | 邊緣 |
| `wgt_buf` [conv.v:132](../lib/models/verilog_head2/conv.v#L132) | **conv2** (IN_CH=96) | **864 × 16** | **1.7 KB** | **建議 SRAM** |

### B. [cal_bbox.v](../lib/models/verilog_head2/cal_bbox.v) — **建議 SRAM**

| 訊號 (行號) | depth × width | 容量 | 等級 |
|-------------|---------------|------|------|
| `size_buf` [cal_bbox.v:106](../lib/models/verilog_head2/cal_bbox.v#L106) | 512 × 16 | **1 KB** | 建議 |
| `off_buf` [cal_bbox.v:107](../lib/models/verilog_head2/cal_bbox.v#L107) | 512 × 16 | **1 KB** | 建議 |

### C. [tail.v](../lib/models/verilog_head2/tail.v) — 可留 reg

`wgt_buf [0:47]` = 96 B，flip-flop OK。

### D. [sigmoid_lut.v](../lib/models/verilog_head2/sigmoid_lut.v) — 無需改

65-entry case-ROM，合成為組合邏輯，不佔 flip-flop。

### E. 重要警示：Feature memory 缺失

> **verilog_head2 內 feature map activation 完全沒有在可合成 RTL 中宣告。**
> 目前 `conv1 → sh1_buf (48 KB) → conv2 → sh2_buf (24 KB) → tail` 的 spatial feature map 緩衝**完全位於 [TEST_head.v](../lib/models/verilog_head2/TEST_head.v) testbench**，由 `$readmemb` 載入 golden。
>
> 必須補上：
> - **conv1 輸入 feature SRAM**：8192 × 16 = **16 KB**
> - **conv1 輸出 / conv2 輸入 feature SRAM**：24576 × 16 = **48 KB**
> - **conv2 輸出 / tail 輸入 feature SRAM**：12288 × 16 = **24 KB**

> **注意**：verilog2 的 [head_top.v:73-75](../lib/models/verilog2/head_top.v#L73) **已經把這三條 feature buffer 宣告在 synthesizable RTL**（不在 TB），所以此問題只存在於 verilog_head2 除錯目錄。

---

# 第三部分：整合版 `verilog2/` — Unified Shared SRAM Pool（**最終方案**）

> **目錄**：[python/lib/models/verilog2/](../lib/models/verilog2/)
> **設計取捨**：**捨棄 Power Gating 細粒度，換取最大 SRAM 共用**（見 §3.5）。

## 3.1 verilog2 重點結構發現

### (1) Backbone 是「**1 個 transformer_block instance 跑 7 次**」

從 [backbone_top.v:108](../lib/models/verilog2/backbone_top.v#L108)：

```verilog
transformer_block #(...) u_tb (...);    // ← 只有 1 個 instance
```

→ **不是** 7 個獨立 hardware copy，而是 1 個硬體 + `tok_buf` 回放 7 次。
→ 所以 `transformer_block / care_attention / mlp` 內的 reg 陣列**只算 1 份**，不像第一部分要 ×7。

### (2) Head 的 feature memory **已經在 RTL 內**

從 [head_top.v:73-75](../lib/models/verilog2/head_top.v#L73)：

```verilog
reg [DATA_W-1:0] x_buf  [0:IN_LEN_HEAD-1];  // 8192  → 16 KB
reg [DATA_W-1:0] sh1_buf [0:C1_LEN-1];      // 24576 → 48 KB
reg [DATA_W-1:0] sh2_buf [0:C2_LEN-1];      // 12288 → 24 KB
```

→ 比 verilog_head2 完整，無需另外補。

### (3) `sglatrack_top.v` 已有時間互斥 FSM

從 [sglatrack_top.v:84-99](../lib/models/verilog2/sglatrack_top.v#L84)：

```
S_IDLE → S_BACKBONE → S_HEAD → S_DONE
              ↑              ↑
        Head 完全 idle    Backbone 完全 idle
```

→ **共用 SRAM 的天然舞台**。

## 3.2 verilog2 全部 reg 陣列清單（depth ≥ 256）

### Backbone 路徑（1 份 hardware）

| 模組 (行號) | 訊號 | depth × width | 容量 |
|------------|------|---------------|------|
| [backbone_top.v:83](../lib/models/verilog2/backbone_top.v#L83) | `tok_buf` | 10240 × 16 | **20 KB** |
| [backbone_top.v:165](../lib/models/verilog2/backbone_top.v#L165) | `out_buf` | 10240 × 16 | **20 KB** |
| [transformer_block.v:63](../lib/models/verilog2/transformer_block.v#L63) | `x_buf` | 10240 × 16 | **20 KB** |
| [transformer_block.v:64](../lib/models/verilog2/transformer_block.v#L64) | `tmp_buf` | 10240 × 16 | **20 KB** |
| [care_attention.v:97](../lib/models/verilog2/care_attention.v#L97) | `x_in_buf` | 10240 × 16 | **20 KB** |
| [care_attention.v:98](../lib/models/verilog2/care_attention.v#L98) | `q_buf` | 10240 × 16 | **20 KB** |
| [care_attention.v:99](../lib/models/verilog2/care_attention.v#L99) | `k_buf` | 10240 × 16 | **20 KB** |
| [care_attention.v:100](../lib/models/verilog2/care_attention.v#L100) | `v_buf` | 10240 × 16 | **20 KB** |
| [care_attention.v:105](../lib/models/verilog2/care_attention.v#L105) | `ao_buf` | 10240 × 16 | **20 KB** |
| [care_attention.v:102](../lib/models/verilog2/care_attention.v#L102) | `qkm_buf` | 1280 × 16 | **2.5 KB** |
| [care_attention.v:103](../lib/models/verilog2/care_attention.v#L103) | `zr_buf` | 1280 × 16 | **2.5 KB** |
| [mlp.v:71](../lib/models/verilog2/mlp.v#L71) | `x_in_buf` | 10240 × 16 | **20 KB** |

> **重要**：上述 9 個 10240×16 buf **大多是同一 activation 在不同 module 邊界的重複 reg**（例如 `transformer_block.x_buf` ≈ `care_attention.x_in_buf` ≈ `mlp.x_in_buf`）。實際晶片**不需要這麼多份**，可摺疊成少數實體 SRAM。

### Head 路徑（1 份 hardware）

| 模組 (行號) | 訊號 | depth × width | 容量 |
|------------|------|---------------|------|
| [head_top.v:73](../lib/models/verilog2/head_top.v#L73) | `x_buf` | 8192 × 16 | **16 KB** |
| [head_top.v:74](../lib/models/verilog2/head_top.v#L74) | `sh1_buf` | 24576 × 16 | **48 KB** |
| [head_top.v:75](../lib/models/verilog2/head_top.v#L75) | `sh2_buf` | 12288 × 16 | **24 KB** |
| [conv.v:132](../lib/models/verilog2/conv.v#L132) (u_conv2) | `wgt_buf` | 864 × 16 | **1.7 KB** |
| [cal_bbox.v:106](../lib/models/verilog2/cal_bbox.v#L106) | `size_buf` | 512 × 16 | **1 KB** |
| [cal_bbox.v:107](../lib/models/verilog2/cal_bbox.v#L107) | `off_buf` | 512 × 16 | **1 KB** |

**可留 reg：** conv1 `wgt_buf` (288)、tail `wgt_buf` (48)、km_buf (32)、kv_buf (256)、fc1_buf (128)、layer_norm `feat_buf` (32)。

## 3.3 Unified Shared SRAM Pool — **總共 7 顆**

關鍵洞察：**真正同時存在的 activation 只有少數幾個**。把 12 條 backbone reg 陣列摺疊成 5~6 顆實體 SRAM，再與 head 共用。

### 共用對應表

| # | Macro 名稱 | macro 規格 | **Backbone phase** 用途 | **Head phase** 用途 |
|---|-----------|-----------|-----------------------|---------------------|
| **1** | `Sram_tok1` | 12288 × 16 SP | `tok_buf` ping（block 間 activation，使用 10240）| `x_buf`（conv1 input，使用 8192，addr offset 2048 跳 template）|
| **2** | `Sram_tok2` | 12288 × 16 SP | `tok_buf` pong / `out_buf`（使用 10240）| `sh2_buf`（conv2 out / tail in，使用 12288 ✓）|
| **3** | `Sram_q` | 12288 × 16 SP | `q_buf` / `ao_buf` 共用（使用 10240）| `sh1_buf` 前半（使用 12288 ✓）|
| **4** | `Sram_k` | 12288 × 16 SP | `k_buf`（使用 10240）| `sh1_buf` 後半（使用 12288 ✓）|
| **5** | `Sram_v` | 12288 × 16 SP | `v_buf`（使用 10240）| （idle, spare）|
| **6** | `Sram_qkm` | 1280 × 16 SP | `qkm_buf` / `zr_buf` 共用 | `cal_bbox.size_buf + off_buf`（使用 1024）|
| **7** | `Sram_c2w` | **1056 × 16 SP**（由 864 升上） | （idle, OFF）| `conv2` weight prefetch（實際使用 864，剩 192 不寫不讀）|

> **macro 統一升到 12288 × 16**：backbone 只用 10240（浪費 16%），但 head 的 sh1/sh2 剛好整除，sh1 = 2 × 12288 完美對齊。
> **SRAM #7 由 864 升到 1056**：864 低於 SHC-SPMBSRAM compiler 最小深度 1056（Mux 4 起跳），compiler 無法生產。升到 1056 後可直接編譯（詳見 §3.8）。

### 7 顆 SRAM 總容量

| # | 規格 | Compiler 設定 | 容量 |
|---|------|--------------|------|
| 1~5 | 12288 × 16 SP | SHC-SPMBSRAM Mux 16 | 5 × 24 KB = **120 KB** |
| 6 | 1280 × 16 SP | SHC-SPMBSRAM Mux 4 | **2.5 KB** |
| 7 | **1056 × 16 SP** | SHC-SPMBSRAM Mux 4 | **2.06 KB** |
| **合計** | **7 顆** | | **~ 124.6 KB** |

> 較原規劃多 ~0.4 KB（SRAM #7 由 864→1056，多 192 entries × 16-bit = 0.375 KB）。代價極小，換取「全部 macro 都能由 PDK compiler 直接生產」。

## 3.4 Phase-by-phase 時序檢查（確認單埠 SRAM 不衝突）

### Backbone 1 個 block 內

| FSM state | SRAM 1 | SRAM 2 | SRAM 3 (Q/AO) | SRAM 4 (K) | SRAM 5 (V) | SRAM 6 (QKM) |
|-----------|--------|--------|---------------|------------|------------|--------------|
| LOAD_X | R | – | – | – | – | – |
| QKV (linear) | R | – | W | W | W | – |
| K_MEAN | – | – | – | R | – | W |
| QK_MEAN | – | – | R | – | – | R+W |
| KV | – | – | – | R | R | – |
| ATTN | – | – | R/W (Q→AO) | – | – | R |
| PROJ | – | W | R | – | – | – |

→ **每顆 SRAM 任何拍只有讀 OR 寫**，單埠 macro 完全夠用。
→ block 切換時 SRAM 1/2 角色互換（ping-pong）。

### Head 4 個 stage

| Stage | SRAM 1 | SRAM 2 | SRAM 3 (sh1A) | SRAM 4 (sh1B) | SRAM 5 | SRAM 6 (bbox) | SRAM 7 (c2wgt) |
|-------|--------|--------|---------------|---------------|--------|---------------|----------------|
| FILL | W (x_buf) | – | – | – | – | – | – |
| CONV1 | R | – | W | W | – | – | – |
| CONV2 | – | W (sh2) | R | R | – | – | R |
| TAIL | – | R | – | – | – | W | – |
| BBOX | – | – | – | – | – | R | – |

→ 同樣**單埠 macro 不衝突**。
→ SRAM 5 在 head phase 完全閒置（可做 spare 或乾脆關電）。

## 3.5 工程取捨：放棄 Power Gating 細粒度

### 取捨對照

| 項目 | 影響 |
|------|------|
| **SRAM 顆數** | 從 ~14 顆 → **7 顆**（減少 50%）|
| **總 SRAM 容量** | ~ 235 KB → **~ 124 KB**（減少 47%）|
| **die area** | SRAM 部分縮小 ~50% |
| **leakage 靜態功耗** | 仍可改善（總 bit cell 變少），但**無法 layer-by-layer 變化** |
| **Power State Table** | **不能再寫 layer-by-layer ON/OFF**，論文章節需重新框架（見 §3.6）|
| **Control 複雜度** | 每顆共用 SRAM 入口加 2-to-1 mux + mode FSM |
| **APR floorplan** | SRAM 必須擺在 backbone & head 都繞得到處（通常擺中央）|
| **驗證成本** | Backbone mode、Head mode 各對拍一次 |

### 為什麼這樣取捨合理

1. **UAV tracking 推論模式**：一次跑完一張影像 → 整顆 idle → 等下一張。**inference 中 power gating 收益有限**（單張影像跑時間短）。
2. **真正 idle 時間在 chip-level**（兩張影像間），整顆 die 可以一起 shutdown，這時 SRAM 共用不共用都 OFF。
3. **die area 與成本**遠比 inference 階段的 leakage 重要。

## 3.6 論文章節（Chapter 3）應對策略

### 衝突點

[thesis_rule2.md](../../.cursor/rules/thesis_rule2.md) §3 明文要求 Chapter 3 必須有 **Power State Table**：

> *探討 Power Gating 時，必須以表格呈現硬體在執行每一層時，各個 Power domain (pgen1 ~ pgenX) 的 On/Off 狀態。*

捨棄 power gating 後，這張表會變成「**幾乎全 ON**」，沒有看點。

### 推薦策略：保留 Logic Power Gating + 新增 Memory Sharing Table

| 項目 | 做法 |
|------|------|
| **SRAM** | 全部共用、全 ON（接受面積優化）|
| **Logic（datapath / FSM）** | 仍做 power gating：backbone 跑時 head logic OFF，反之亦然 |
| **Chapter 3 §3.X** | **Power State Table** 改寫「邏輯 domain」的 ON/OFF |
| **Chapter 3 §3.Y**（新增）| **Memory Sharing Table** 展示 7 顆 SRAM 在 backbone/head 雙重用途 |

#### Power State Table 改寫範例

| 階段 | pgen_bb_logic | pgen_head_logic | SRAM_pool（共用）|
|------|:---:|:---:|:---:|
| Backbone Block 0~6 | **ON** | OFF | Always-ON |
| Head Conv1~bbox | OFF | **ON** | Always-ON |

#### 論文敘述範本（英文）

> *"To maximize SRAM area efficiency, the proposed architecture employs a unified SRAM pool shared across backbone and head phases. Power gating is applied at the logic domain level only, as shown in Table 3.X. The shared SRAM domain remains powered throughout inference but achieves lower total leakage through reduced macro count (~7 vs ~14 in a non-shared design). Memory sharing schedule is detailed in Table 3.Y."*

## 3.7 sglatrack_top 重構草圖

詳細 RTL 草圖見：[python/md/sglatrack_top_shared_sram_draft.v](sglatrack_top_shared_sram_draft.v)

內容包含：

- 7 顆 `sram_sp` 行為模型實例（APR 時換成 PDK macro）
- backbone-side / head-side SRAM port 線宣告（各 6 組 quintet）
- `running_mode` 驅動的 mux 路由（每顆 SRAM 一段 4 行）
- `backbone_top` / `head_top` 實例化（含新 SRAM port 介面契約）
- FSM 三段式（state register / next-state / output + `running_mode` 切換）

### 重構依賴清單（檔頭已標註）

| 模組 | 必改項 |
|------|--------|
| `backbone_top.v` | 刪 `tok_buf / out_buf` reg；新增 6 組 SRAM port；下傳 SRAM 3~6 ports |
| `transformer_block.v` | 刪 `x_buf / tmp_buf`；改用上層傳入 SRAM port |
| `care_attention.v` | 刪 `x_in_buf / q_buf / k_buf / v_buf / ao_buf / qkm_buf / zr_buf`|
| `mlp.v` | 刪 `x_in_buf`；改用上層 port |
| `head_top.v` | 刪 `x_buf / sh1_buf / sh2_buf`；新增 6 組 port；**刪除 S_FILL**（資料已在 SRAM 1）；sh1 split：`addr[13]==0 → SRAM 3、==1 → SRAM 4`|
| `cal_bbox.v` | 刪 `size_buf / off_buf`；改用 SRAM 6（addr[9] 區分 size/off）|
| `conv.v` | conv2 instance 的 `wgt_buf` 改用 SRAM 7 |

## 3.8 PDK Compiler 對齊檢查（SHC-SPMBSRAM）

依據 PDK Table 2.4「SHC-SPMBSRAM Compiler Range Information」：

| Mux Option | Word Depth 範圍 | Word Width 範圍 |
|:----------:|----------------|----------------|
| 4 | 1056~2048, 2112~4096 (step 32) | 8 ~ 144 |
| 8 | 2112~4096, 4224~8192 (step 64) | 4 ~ 72 |
| 16 | 4224~8192, 8448~16384 (step 128) | 4 ~ 39 |

### 7 顆 SRAM 逐一驗證

| # | 原規劃 | Compiler 驗證 | **修正後規格** | 備註 |
|---|--------|--------------|---------------|------|
| 1~5 | 12288 × 16 | Mux 16：12288 = 8448 + 30×128 ✓<br>Width 16 ∈ [4, 39] ✓ | **12288 × 16** ✓ | 不用拆 |
| 6 | 1280 × 16 | Mux 4：1280 = 1056 + 7×32 ✓<br>Width 16 ∈ [8, 144] ✓ | **1280 × 16** ✓ | 不用拆 |
| 7 | **864 × 16** | ❌ 864 < 1056（Mux 4 最小值）<br>**任何 Mux 都拒絕生產** | **1056 × 16**（升上）| 多浪費 192 entries (18%) |

### SRAM #7 為什麼必須升到 1056

864 同時 **小於三個 Mux 選項的最小深度門檻**（1056 / 2112 / 4224），落在「沒人接的死角」。

**物理原因**：SRAM macro 的 peripheral（row decoder、sense amp、write driver）有固定面積開銷。當深度太小時，peripheral / bit cell 比例失衡，compiler 廠商乾脆設下限避免浪費。

**選擇方案 A：升到 1056 × 16，Mux 4 編譯**
- 多 0.4 KB 容量代價（可忽略）
- macro 規格標準、可直接由 SHC-SPMBSRAM compiler 產出
- conv2 weight prefetch 實際只用前 864 entries（addr 0~863），addr 864~1055 不寫不讀（X 狀態不影響功能）

**RTL 改動最小**：[conv.v](../lib/models/verilog2/conv.v) 內 `wpre_feat` 計數器與 weight ROM 對應位址維持 0..863，**SRAM #7 的 address port 寬度從 10 bit 加大到 11 bit**（ceil(log2(1056)) = 11），但 controller 只用低 10 bit 即可。

### 24576 為什麼必須拆 2 顆

24576 > 16384（Mux 16 最大深度上限），**沒有任何 Mux option 在 16-bit width 下能提供 24576 深度的單 macro**。

**拆法**：24576 = 12288 + 12288，對應 SRAM 3（sh1 part A）+ SRAM 4（sh1 part B）。

**Head 端 RTL 解碼**：
```verilog
wire        sh1_sel        = sh1_addr[13];        // 0 -> SRAM 3, 1 -> SRAM 4
wire [12:0] sh1_local_addr = sh1_addr[12:0];
```

### Compiler 對齊後總結

| 指標 | 數值 |
|------|------|
| SRAM macro 顆數 | **7 顆**（全部可由 SHC-SPMBSRAM 產出） |
| 總容量 | **~ 124.6 KB** |
| 最大單顆 macro | 12288 × 16 = 24 KB（Mux 16）|
| 最小單顆 macro | 1056 × 16 = 2.06 KB（Mux 4）|
| 規格相容性 | ✅ 100% 符合 SHC-SPMBSRAM Table 2.4 |

## 3.9 SHC-SPMBSRAM 編譯指令清單

**指令格式**：`Sram_<name> <depth> <width> <CM_mux> s`
- `<CM_mux>`：column mux 選項（4 / 8 / 16），由 depth 決定
- `s`：segment option（Small / SGE）

### 7 條指令（一鍵複製）

```
# ============================================================================
# Verilog2 Unified Shared SRAM Pool - 7 SHC-SPMBSRAM macros
# Format: Sram_<name> <depth> <width> <CM_mux> s
# All single-port (segment=s), Q8.8 16-bit data path
# ============================================================================
Sram_tok1   12288 16 16 s
Sram_tok2   12288 16 16 s
Sram_q      12288 16 16 s
Sram_k      12288 16 16 s
Sram_v      12288 16 16 s
Sram_qkm    1280  16 4  s
Sram_c2w    1056  16 4  s
```

### 指令對應角色（Backbone + Head 共用對照）

| 指令 | Backbone phase 用途 | Head phase 用途 |
|------|--------------------|----------------|
| `Sram_tok1` | `tok_buf` ping（用 10240）| conv1 `x_buf`（用 8192）|
| `Sram_tok2` | `tok_buf` pong / `out_buf`（用 10240）| `sh2_buf`（用 12288 ✓）|
| `Sram_q` | `q_buf` / `ao_buf` 共用（用 10240）| `sh1_buf` lo（用 12288 ✓）|
| `Sram_k` | `k_buf`（用 10240）| `sh1_buf` hi（用 12288 ✓）|
| `Sram_v` | `v_buf`（用 10240）| idle / spare |
| `Sram_qkm` | `qkm_buf` / `zr_buf` 共用（用 1280）| `cal_bbox.size_buf+off_buf`（用 1024）|
| `Sram_c2w` | idle | conv2 weight prefetch（用 864）|

### CM_mux 選擇邏輯

各 macro 為何用該 Mux：

| 指令 | depth | 為何選此 Mux | 範圍驗證 |
|------|-------|-------------|---------|
| `Sram_tok1/2/q/k/v` | 12288 | **Mux 16**：12288 > 8192（Mux 8 上限），只能 Mux 16 | depth ∈ [4224, 16384] ✓ |
| `Sram_qkm` | 1280 | **Mux 4**：1280 < 2112（Mux 8 下限），只能 Mux 4 | depth ∈ [1056, 4096] ✓ |
| `Sram_c2w` | 1056 | **Mux 4**：Mux 4 最小深度（864 升上）| depth ∈ [1056, 4096] ✓ |

> 註：Mux 16 的 width 上限為 39，剛好涵蓋 16 bit（Q8.8）；Mux 4/8 的 width 上限更寬（72/144），對 16-bit 都沒問題。

---

# 第四部分：SRAM port 設計通用注意事項

不論 backbone 或 head，每顆 SRAM macro 在 RTL 內須遵守 [verilog_rule.mdc §7](../../.cursor/rules/verilog_rule.mdc)：

1. **讀取契約三行註解** 寫在模組頭：
   - 位址何時有效（`posedge clk`）
   - 讀出資料何時有效（位址後第 1 拍）
   - macro `CLK` 是 `clk` 還是 `~clk`
2. **禁止同拍 write & read 同一位址**（除非 macro 規格明訂 write-through）。
3. **FSM 顯式區分** `ADDR` 與 `USE` 狀態，避免「送位址同拍就送進 MAC」的 off-by-one。
4. **`stream` / `stream_r` 預取慣例**（§7.2）：weight ROM 已驗證可行；token / feature SRAM 也應沿用此模式。
5. **golden 對拍**：每顆 SRAM 對應的 activation stream 須在模組頭標出 golden 檔，例如：
   ```
   // Golden: vit_care_relu6_numpy_trunk_dim32_out/Activation/backbone_after_norm_backbone_out_bi.txt
   ```

---

# 第五部分：後續工作順序

依使用者工作流（**先 verilog_backbone2、verilog_head2，再同步 verilog2**）：

## Phase 1：verilog_backbone2 改 SRAM port（分模組除錯）

1. 在 [backbone_top.v](../lib/models/verilog_backbone2/backbone_top.v) 拆掉 `tok_buf / out_buf`，改為 2 組 SRAM port（SRAM 1、2）
2. 在 [transformer_block.v](../lib/models/verilog_backbone2/transformer_block.v) 拆掉 `x_buf / tmp_buf`，改為對上層的 SRAM port pass-through
3. 在 [care_attention.v](../lib/models/verilog_backbone2/care_attention.v) 拆掉 `q/k/v/ao/qkm/zr_buf`，改為 4 組大型 + 1 組中型 SRAM port
4. 在 [mlp.v](../lib/models/verilog_backbone2/mlp.v) 拆掉 `x_in_buf`，共用上層的 token SRAM
5. 用行為模型 `sram_sp` 跑 testbench，對 `Activation/*_bi.txt` 通過 backbone 對拍
6. 保留 reg 版本作 reference build（`` `ifdef USE_REG_BUF ``），便於 regression

## Phase 2：verilog_head2 改 SRAM port（分模組除錯）

1. 在 [head_top.v](../lib/models/verilog_head2/) **補上** `x_buf / sh1_buf / sh2_buf` 對應的 3+1 組 SRAM port（verilog_head2 目前缺，但 verilog2 已有可參考）
2. [conv.v](../lib/models/verilog_head2/conv.v) 的 conv2 `wgt_buf` 改 SRAM port
3. [cal_bbox.v](../lib/models/verilog_head2/cal_bbox.v) 的 `size_buf / off_buf` 合併為 1 顆 SRAM（addr[9] 區分）
4. testbench 對 head bbox 通過對拍

## Phase 3：同步到 verilog2 + 加 sglatrack_top mux

1. 把 Phase 1、2 改好的 RTL 同步到 [verilog2/](../lib/models/verilog2/)
2. 用 [sglatrack_top_shared_sram_draft.v](sglatrack_top_shared_sram_draft.v) 草圖替換現有 [sglatrack_top.v](../lib/models/verilog2/sglatrack_top.v)
3. 重新整合 7 顆 SRAM + mux + `running_mode` FSM
4. 對端到端 TEST.v 通過對拍

## Phase 4：APR 前置作業

1. 把行為模型 `sram_sp` 替換成 PDK SRAM compiler 產出的真實 macro
2. 確認所有 SRAM 與 ROM 的 `CLK` 邊沿一致性（本專案 weight ROM 多為 `CLK(~clk)`）
3. 規劃 logic-only power gating domain（呼應論文 Power State Table）
4. 撰寫 Memory Sharing Table（呼應論文 §3.Y 新增章節）

---

> 若需要進一步輸出 **backbone_top.v / head_top.v 的 SRAM port 重構 patch、Memory Sharing Table Markdown、論文 §3 章節範本**，後續再請指示。
