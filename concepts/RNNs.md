![[Pasted image 20260411180749.png]]

👉 Task: predict next word step-by-step
A cute teddy bear is reading
> **RNN processes words one by one and remembers previous words**
## 🔹 Step 1: Input = "A"

### One-hot:
A → [1,0,0,0,0,0]
### Hidden state:
$$
h_1 = f(W_x x_1 + W_h h_0)h1​=f(Wx​x1​+Wh​h0​)
$$

👉 Since no previous state:

h₀ = [0, 0]

---

## 🔹 Step 2: Compute hidden state

Assume:

W_x =  
[0.2, 0.5]  
[0.1, 0.3]  
...

Then:

h₁ ≈ [0.2, 0.9]

👉 This is similar to embedding in [[Foundations#Word2Vec]]

---

## 🔹 Step 3: Predict next word
```math
||{"id":1521350095464}||

s
```
```math
||{"id":320953565113}||

sdlafja;

```
```math
||{"id":404846674536}||

```
$$
$$
$$
y1=softmax(Wy⋅h1)y_1 = \text{softmax}(W_y \cdot h_1)y1​=softmax(Wy​⋅h1​)
$$
Output:

[0.2, 0.4, 0.1, 0.1, 0.1, 0.1]

👉 Predicts:

"A" → "cute" ✅

---

# 🔁 4. Now IMPORTANT Difference Starts

## Next input = "cute"

But now:

👉 RNN uses **previous memory**

---

## 🔹 Step 4: Hidden state update
h2​=f(Wx​x2​+Wh​h1​)
👉 Now:

- x2​ = "cute"
- h1​ = memory from "A"

---

### Calculation (intuition)

h₂ = (current word info) + (past memory)

So:

h₂ ≈ [0.6, 1.2]

---

## 🔹 Step 5: Predict next word

y2=softmax(Wy⋅h2)y_2 = \text{softmax}(W_y \cdot h_2)y2​=softmax(Wy​⋅h2​)

👉 Output:

cute → teddy ✅

---

# 🔁 Continue…

|Step|Input|Hidden State|Prediction|
|---|---|---|---|
|1|A|h₁|cute|
|2|cute|h₂|teddy|
|3|teddy|h₃|bear|
|4|bear|h₄|is|
|5|is|h₅|reading|

---

# ⚙️ 5. Full RNN Formula

At each step:

### Hidden state:

$$
ht=tanh⁡(Wxxt+Whht−1)h_t = \tanh(W_x x_t + W_h h_{t-1})ht​=tanh(Wx​xt​+Wh​ht−1​)
$$

### Output:

$$
yt=softmax(Wyht)y_t = \text{softmax}(W_y h_t)yt​=softmax(Wy​ht​)

$$


# 💡 6. Key Difference from Word2Vec

|Word2Vec|RNN|
|---|---|
|One word at a time|Sequence|
|No memory|Has memory|
|Static embedding|Dynamic context|
|Predict context|Predict next word|

---

# 🔥 7. Biggest Insight

👉 Word2Vec:

"A" → "cute"

👉 RNN:

"A cute teddy" → "bear"

👉 RNN understands **context**

---

# 🧠 8. Intuition (VERY IMPORTANT)

👉 RNN is like reading a sentence:

- You remember previous words
- That memory helps predict next word

---

# 🧾 Obsidian Note (Perfect)

# RNN (Recurrent Neural Network)  
  
## Idea  
Processes sequence one word at a time with memory  
  
## Formula  
$$
h_t = tanh(Wx xt + Wh h(t-1))  
  ,  y_t = softmax(Wy ht)  

$$

## Key Feature  
Uses previous hidden state (memory)  
  
## Example  
A → cute → teddy → bear  
  
## Insight  
RNN captures context, unlike Word2Vec

---

# 🚀 Final Understanding

👉 RNN =  
**Word2Vec + memory of previous words**

---

# ⚠️ Important (Next Step)

RNN has problems:

- forgets long sentences
- vanishing gradients

👉 That’s why we use:

- LSTM
- GRU
- Transformers (LLMs)
