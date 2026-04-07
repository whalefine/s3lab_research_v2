> From: https://chatgpt.com/g/g-p-696735f1876c819183a609dbc7d6da09-xun-zhao-lun-wen-yan-jiu-ti-mu/c/69d16499-6fac-83a4-8b3f-2aed0633e707

# you asked

幫我看這篇論文用什麼方法解決什麼問題?

---

# chatgpt response

這篇論文的核心其實很清楚：**用一個「硬體友善的函數」取代 Softmax**，並且設計對應的 VLSI 架構。

我幫你整理成你在做 ViT / hardware 時會用到的重點👇

---

# 🔥 一、這篇論文在解決什麼問題？

## ✅ 問題本質：Softmax 太難做硬體

Softmax 定義為：
$$
\text{Softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

👉 問題在這兩個地方：
1. **exponential (e^x)** → 很難硬體實現  
2. **division (除法)** → 也很貴（latency + area）

📌 論文明確指出：
- LUT 方法 → 要很大記憶體
- log-sum-exp → 還是很複雜
- approximation → 會掉精度

👉 結論：
> Softmax 是 Transformer 硬體化的最大ile0

---

# 🚀 二、論文提出什麼方法？

## ⭐ 核心方法：Squaremax（取代 Softmax）

### 👉 定義（最重要）
$$
y_i = \frac{\text{ReLU}(x_i)^p}{\sum_j \text{ReLU}(x_j)^p}
$$

其中：
- 論文選 **p = 2（Square）**
- 所以叫 **Squaremax**

👉 實際形式：
$$
y_i = \frac{\text{ReLU}(x_i)^2}{\sum_j \text{ReLU}(x_j)^2}
$$

---

## 🔍 直觀理解（很重要）

Softmax：
- 用 **exp 放大差異**

Squaremax：
- 用 **平方放大差異**

👉 效果：
- 都會讓「大值更大、小值更小」
- 但 Squaremax **不用 exp**

---

## 🧠 為什麼用 ReLU + square？

論文設計是有目的的：

### 1️⃣ ReLU
$$
\text{ReLU}(x) = \max(0,x)
$$

👉 保證：
- output ≥ 0
- 避免負值造成 weird weighting

---

### 2️⃣ square
$$
x^2
$$

👉 提供：
- 非線性（像 exp）
- 放大大值

---

## 📊 和 Softmax 性質對齊

論文強調 Squaremax 保留關鍵性質：

| 性質 | Softmax | Squaremax |
|------|--------|----------|
| 輸出總和=1 | ✅ | ✅ |
| 非負 | ✅ | ✅ |
| 單調 | ✅ | ✅ |
| 可微 | ✅ | 幾乎（0除外） |

👉 所以可以直接替代

---

# ⚡ 三、第二個關鍵：消除 division

Softmax 還有一個問題：

$$
\frac{1}{\sum}
$$

👉 論文做了這個：

## 👉 Division → Multiplication + Shift

$$
\frac{x}{D} \approx x \cdot \frac{1}{m} \cdot 2^{-n}
$$

### 作法：
1. 將 denominator 表示成：
   $$
   D = m \cdot 2^n
   $$

2. $2^{-n}$ → 用 **shift**
3. $1/m$ → 用 **小 decoder（8 entries）**

👉 完全不用：
- divider
- LUT

📌 重點：
> 整個設計只用 multiplier + siteturn0file0

---

# 🏗️ 四、整體演算法流程（硬體版）

## Step 1
對每個 $x_i$：
1. ReLU
2. square
3. 累加（sum）

---

## Step 2
對每個元素：
1. 乘上 reciprocal
2. shift
3. 得到 normalized output

---

👉 簡化版：

$$
\text{Squaremax} = \text{Normalize}( \text{ReLU}(x)^2 )
$$

---

# 📈 五、效果如何？（很關鍵）

## ✅ Accuracy（Transformer）

在 DeiT / Swin：

- **不輸 Softmax**
- 甚至 **更好**

👉 example：
- DeiTeciteturn0file0

---

## ✅ Hardware（重點）

| 指標 | 改善 |
|------|------|
| 不用 exp | ✅ |
| 不用 LUT | ✅ |
| 不用 division | ✅ |
| Area | ↑↑ |
| Power | ↓↓ |
| Throughput | ↑ |

👉 結論：
> 非常適合 ASIC / FPGA / ViT accelerator

---

# 🎯 六、你做 ViT / Tracking 的重點結論

這篇論文本質就是：

## 👉 Softmax-free attention 的一種

### 可以寫成：

$$
\text{Attn}(Q,K,V) = \frac{\text{ReLU}(QK^T)^2}{\sum} V
$$

---

# 🧠 七、對你研究的價值（超重要）

你現在在做：
👉 UAV tracking + softmax-free + hardware friendly

這篇論文提供：

## ✅ 優點
- 完全移除 exp（超關鍵）
- O(n²) 還在，但運算變便宜
- 很適合 fixed-point（Q format）

---

## ⚠️ 可能缺點
- 沒有 gating（不像 ReGLA / CARE）
- attention 表達力可能較弱

👉 所以你之前 AUC 掉是合理的

---

# 🏁 八、一句話總結

👉 這篇論文：

> 用 **ReLU² + normalization（Squaremax）** 取代 Softmax，  
> 並透過 **division→shift+multiplier**，實現一個**完全硬體友善的 attention weighting 方法**。

---

# 🚀 如果你下一步要做（我可以幫你）

你可以接著做：

1. 👉 **Squaremax + gating（改良版）**
2. 👉 **Squaremax + linear attention（O(n)）**
3. 👉 **Tracking-friendly normalization（避免 AUC 掉）**

如果你要，我可以幫你設計一個：
👉「可以投稿的 softmax-free tracking attention（硬體版）」

---

# you asked

此論文的attention怎麼寫的?

---

# chatgpt response

這篇論文其實**沒有直接寫出完整的 attention 公式（像 Transformer paper 那樣）**，但它的核心就是：

👉 **把 attention 裡的 softmax 換成 Squaremax**

所以我們可以幫你「還原」它在 attention 中實際長怎樣👇

---

# 🔥 一、標準 Attention（先對照）

標準 Transformer：

$$
\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

---

# 🚀 二、這篇論文的核心替換

論文把：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

改成：

$$
\text{Squaremax}(x_i) = \frac{\text{ReLU}(x_i)^2}{\sum_j \text{ReLU}(x_j)^2}
le0

---

# 🧠 三、套進 Attention（最重要）

令：

$$
A = \frac{QK^T}{\sqrt{d}}
$$

---

## ✅ 那 Attention 就變成：

$$
\text{Attn}(Q,K,V)
=
\left(
\frac{\text{ReLU}(A)^2}{\sum_j \text{ReLU}(A)^2}
\right)
V
$$

---

# ✨ 四、展開寫成「矩陣版本」（你做硬體會用這個）

### Step 1：算 attention score
$$
S = \frac{QK^T}{\sqrt{d}}
$$

---

### Step 2：nonlinearity（取代 exp）
$$
S' = \text{ReLU}(S)^2
$$

---

### Step 3：row-wise normalization
$$
\hat{S}_{i,j} = \frac{S'_{i,j}}{\sum_j S'_{i,j}}
$$

---

### Step 4：乘 V
$$
\text{Attn} = \hat{S} V
$$

---

# ⚡ 五、對照差異（重點整理）

| 步驟 | Softmax Attention | Squaremax Attention |
|------|-----------------|------------------|
| 非線性 | exp | ReLU² |
| normalization | 除法 | 除法（但硬體轉換） |
| 硬體成本 | 高 | 低 |
| LUT | 常需要 | 不需要 |

---

# 🔧 六、硬體版本（更貼近 RTL）

實際上他們不是直接做：

$$
\frac{1}{\sum}
$$

而是做👇

---

### 👉 normalization 改寫

$$
\hat{S}_{i,j} = S'_{i,j} \cdot \text{reciprocal}\left(\sum_j S'_{i,j}\right)
$$

其中 reciprocal 是：

- decoder（3-bit）
- shift
- multiplier

👉 完全沒有rn0file0

---

# 🎯 七、你做 ViT / Tracking 要注意的點

這個 attention 的本質：

## ✅ 優點
- 完全 **softmax-free**
- 沒有 exp → 超適合硬體
- 可直接套在 ViT

---

## ⚠️ 缺點（你 AUC 掉的原因）

### 1️⃣ 沒有 exponent sharpness
- exp → 非常尖銳
- square → 比較平

👉 attention 不夠 focus

---

### 2️⃣ ReLU 會砍掉負值
- 原本 attention 是 signed
- 現在變 sparse

---

# 🚀 八、如果你要做改良（超重要）

你現在做 tracking，我會建議你這樣改：

## 👉 改良版（推薦）

$$
\text{Attn} =
\frac{(\text{ReLU}(QK^T)+\epsilon)^2}{\sum}
V
$$

或

$$
=
\frac{(\text{ReLU}(QK^T)^2 + \alpha QK^T)}{\sum}V
$$

👉 保留一些負值資訊（很重要）

---

# 🏁 一句話總結

👉 這篇論文的 attention 本質是：

> 把  
> $$
> \exp(\cdot)
> $$  
> 換成  
> $$
> \text{ReLU}(\cdot)^2
> $$  
> 並做 normalization，形成一個 **完全不需要 exponential 的 attention 機制**

---

如果你要，我可以幫你把這個：

👉 **直接改寫成 Verilog-friendly pipeline（含 Q-format）**  
👉 或設計「tracking 不掉 AUC 的改良版」