這是一份為你量身打造的**「升級版 `thesis_rule.md`」**。我已經將你們實驗室高分論文的排版與結構鐵律，結合了你專屬的**軟硬體技術敘事（CARE + ReLU6 + Q8.8 等防幻覺機制）**以及**隱藏版表格潛規則**，全部融合進去。

你可以直接複製以下 Markdown 內容，取代你原本的 `thesis_rule.md` 檔案。未來在使用 AI 協助撰寫或潤飾內文時，只要 AI 讀取到這份文件，就絕對不會發生結構偏離或技術幻覺！

---

```markdown
--------------------------------------------------------------------------------

#### description: Thesis writing rules for structure, chapter scope, references, technical narrative alignment, and formatting. alwaysApply: false
### 論文撰寫規則與實驗室黃金模板（僅在提到論文撰寫語境時啟用）

#### 啟用條件
*  只有在使用者明確提到「要寫論文 / 正在寫論文 / 論文章節 / 摘要 / 引言 / related work / 方法 / 實驗 / 結論」等論文撰寫語境時，才套用本規則。
*  若對話內容不是論文撰寫任務，忽略本規則。

---

### 壹、 論文標題與摘要規範

#### 論文標題（固定，不可自行改寫）

- 中文標題：應用於 UAV 視覺追蹤之 16 奈米 Softmax-Free Vision Transformer 硬體加速器
- 英文標題：A 16 nm Softmax-Free Vision Transformer Hardware Accelerator for UAV Visual Tracking
- 中英文標題需一一對應，涉及封面、摘要首頁、題目欄時必須使用完整標題。內文需縮寫時，僅可以「本文」或「本研究」指代。

#### 摘要寫法模板（強制四段式）

- 摘要固定四段式，每段對應一個主題，缺一不可：
  1.  **第一段（背景）：** 應用情境背景（邊緣運算 / UAV 視覺追蹤 / 問題陳述）。
  2.  **第二段（方法）：** 本文提出的方法（宣告使用 Softmax-Free CARE 架構與 ReLU6，並說明設計動機）。
  3.  **第三段（硬體）：** 硬體實作（在 16nm 製程實作、使用 Q8.8 量化與 Power Gating 技術）。
  4.  **第四段（結果）：** 關鍵量化結果（必須包含：工作頻率 MHz、功耗 mW、準確率/AUC %，三者必須出現）。
- **關鍵字格式** ：**關鍵字：** 詞1、詞2、詞3、詞4、詞5（黑體，頓號分隔）。
- **Keywords 格式** ：**Keywords**: term1, term2, term3（黑體，逗號分隔）。

---

### 貳、 本研究專屬技術敘事（防幻覺與對齊鐵律）

在撰寫方法、實驗與硬體章節時，**必須嚴格遵守以下技術設定，嚴禁 AI 幻覺出其他架構或傳統參數**：

- **模型選型鐵律：** 本研究的 Softmax-free 架構嚴格基於 **「CARE 架構 + ReLU6 + Q8.8 定點量化」**，具有 $O(N)$ 線性複雜度與無 `exp` 的硬體友善特性。**絕不可提及使用傳統 Softmax、Squaremax、SimA 或 MALA (ELU + RoPE)**。
- **軟硬體驗證管線（Python Golden Check）：** 在敘述軟硬體驗證時，權重來源必須強調使用 **NPY 格式**（而非 PTH），並輸出為 Binary/TXT 格式，作為硬體 RTL 逐級比對的 Golden Reference。
- **資料集與基準數據：** 追蹤效能比較以 UAV123、UAVTrack112 等資料集為主。論文中討論效能折衷時，應以真實數據為準：Baseline 原架構 Avg AUC 為 62.1%，而本研究採用的 CARE + ReLU6 (Fixed-point) Avg AUC 約為 58.4% ~ 59.14%。
- **硬體驗證收尾：** RTL 與驗證收斂後進行 APR（自動佈局繞線），並以 DRC、LVS 皆 clean 作為晶片實作指標。

---

### 參、 各章節寫作結構與潛規則

#### 總原則與排版限制

- 清楚區分「別人的工作（Ch1）」與「自己的工作（Ch2, Ch3）」：前者放背景與文獻，後者放方法、實作與結果，絕不可混寫。
- **小節標號限制：** 小節與圖號固定使用二層（如 1.1、1.2），**嚴禁出現三層標題（如 1.1.1）**。

#### Chapter 1（Introduction）

- **第一節 (1.1)：** 固定為應用情境介紹（如 UAV Visual Tracking Dataset）。**嚴禁在 1.1 節出現任何底層數學公式（如 Softmax 原公式）**。
- **漏斗式公式出場：** 將傳統 Softmax 包含的 `exp` 與浮點數除法公式，放置在 1.X 節專屬的「硬體痛點分析」小節（例如：Hardware-Friendly Softmax Approximations in Transformer）中，將其作為缺點靶子抨擊。
- **倒數第二節 (Motivation)：** 總結前人公式的硬體限制，帶出本研究解法，並**明確列出 3 點 Contributions**（1. 演算法改良；2. 系統/量化優化；3. 硬體實作結果）。
- **最後一節 (Thesis Organization)：** 逐章說明各章內容（一段式，每章一句），不可省略。

#### Chapter 2（Software Implementation）

- **第一節 (2.1) Architecture Overview：** 固定給出整體架構圖或流程圖，不可省略。
- **第二節 (2.2) Dataset Preprocessing（強制加入）：** 必須包含「資料前處理」說明，並**強制附上一張比較表**，比較不同插值法（如 Nearest neighbor, Bilinear, Bicubic）與影像尺寸對準確率的影響。
- **最後一節 (2.X) Software Result：** 呈現軟體實作的結果（如 AUC 數據等）。若有 Confusion matrix 必須放在這一節。

#### Chapter 3（Hardware Implementation）

- **硬體實作兩大必備神表（強制加入）：** 在描述硬體架構與管理時，必須附上以下兩張表：
  1.  **Memory Usage Table（記憶體使用量表）：** 以表格列出每一層的 Input data、Psum、Feature map 分配到的 SRAM 大小與總 bits 數。
  2.  **Power State Table（電源狀態表）：** 探討 Power Gating 時，必須以表格呈現硬體在執行每一層時，各個 Power domain (pgen1 ~ pgenX) 的 `On/Off` 狀態。
- **倒數第二節 (Hardware Implementation Analysis and Comparison)：** 必須包含硬體實作結果總表（頻率、功耗、面積），以及一張名為 **"Comparison with prior methods"** 的比較表。
- **最後一節 (Summary)：** 一段式總結本章重點。

#### Chapter 4（Conclusion and Future Works）

- 收斂成結論與未來工作。常見拆法：4.1 Conclusion、4.2 Future Works。

---

### 肆、 標號、圖片與公式規範

- **圖號格式** ：Figure X.Y Description [ref].（例：Figure 1.3 Flowchart of tracking algorithm.）
- **表號格式** ：Table X.Y Description.（例：Table 3.10 Comparison with prior methods.）
- 圖表編號按章節重新計數（Figure 1.x 在 Ch1，Figure 2.x 在 Ch2）。
- 引用他人圖片時需在圖說末標注 `[ref]`。
- **數學公式：** 需用可編輯方程式（LaTeX 或 Word 方程式），絕對不可使用截圖。
```
