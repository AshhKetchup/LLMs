---
tags: [llm, nlp, transformer, self-attention, positional-encoding, multi-head-attention]
source_count: 3
last_updated: 2026-04-14
source: "Lec 15 | Self-Attention + Lec 16 | Transformer Architecture + Lec 17 | Positional Encoding — Introduction to LLMs playlist"
---

# Transformer

Proposed by Vaswani et al. ("Attention Is All You Need", 2017). The Transformer replaces all recurrent connections with **self-attention**, enabling full parallel processing and better long-range dependency capture.

---

## Motivation: Why Remove Recurrence?

Vanilla attention (in seq2seq) already gives every decoder state direct access to all encoder states. This raises a key question: **if we have attention shortcuts, why do we still need the recurrent chain?**

The recurrent chain is the reason sequences can't be parallelized — computing $h_t$ requires $h_{t-1}$. If we remove recurrence and use only attention, we can process all tokens simultaneously on a GPU.

**Three remaining problems from RNNs/attention:**
1. Sequential access — can't parallelize
2. Position information lost — tokens have no inherent order without recurrence
3. Self-attention is linear — need nonlinearity

The Transformer addresses all three.

---

## Self-Attention

In self-attention, every token attends to all other tokens in the **same sequence** (no encoder/decoder split needed).

> **Analogy:** Like a group meeting where everyone can talk to everyone else simultaneously, instead of passing notes one-by-one down a chain.

From each token's hidden state $h$, generate three vectors via learned projections:
- **Query** $q = W_Q h$
- **Key** $k = W_K h$
- **Value** $v = W_V h$

The same matrices $W_Q, W_K, W_V$ are shared across all token positions.

**Scaled dot-product attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

The $\sqrt{d_k}$ scaling prevents the dot products from growing large (which would saturate softmax and kill gradients) as the key dimension increases.

**Complexity:** $O(n^2 d)$ — every token attends to every other token. Quadratic in sequence length.

**Self-attention is linear in the hidden states** — it's a linear transformation (weighted average of values, with data-dependent weights). This is why the feed-forward layer is needed.

---

## Multi-Head Attention

Running a single attention head captures one type of relationship between tokens. Multi-head attention runs $H$ attention heads in parallel, each with its own $W_Q^h, W_K^h, W_V^h$ matrices:

$$\text{head}_h = \text{Attention}(Q W_Q^h,\; K W_K^h,\; V W_V^h)$$

$$\text{MultiHead}(Q, K, V) = \text{concat}(\text{head}_1, \ldots, \text{head}_H) \cdot W_O$$

$W_O$ is an output projection that maps the concatenated $H \times d_v$ vector back to dimension $d$.

**Intuition (from CNNs):** Multiple convolutional filters each capture different visual features (edges, textures). Multiple attention heads capture different semantic relationships simultaneously — one head might track syntactic dependencies, another coreference, another positional patterns. We don't specify which head does what; this emerges from training.

> **Analogy:** Like looking at a painting through different filters simultaneously — one for color, one for shapes, one for texture. Each head sees the same text but focuses on different relationships.

The original Transformer paper used $H = 8$ heads.

---

## Positional Encoding

Self-attention is **permutation invariant** — swapping "the man eats cheese" and "the cheese eats man" would produce the same self-attention output. Position information must be injected explicitly.

**Method:** add a positional embedding $P_\text{pos}$ to the token embedding before feeding to the first block:

$$\text{input}_i = E_i + P_i$$

> **Analogy:** Like adding seat numbers to identical concert tickets so you know who sits where — without them, everyone looks the same.

(Summing, not concatenating — keeps dimensionality constant; no strong theoretical justification for sum vs. concat.)

Positional encoding is only added once at the input layer, not at every block.

### Sinusoidal Positional Encoding (Vaswani et al.)

Motivated by binary positional encoding: in binary, each bit alternates at a different frequency (LSB fastest, MSB slowest). Use continuous sinusoids instead:

$$P_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_\text{model}}}\right)$$
$$P_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_\text{model}}}\right)$$

- $\text{pos}$ = token position in sequence
- $i$ = element index within the positional vector
- Alternating sin/cos: even indices use sin, odd use cos

**Properties:**
- Low $i$ (early elements) → high frequency — differentiates nearby positions
- High $i$ (later elements) → low frequency — differentiates distant positions
- Nearby positions differ mostly in high-frequency bands; distant positions differ in both high and low bands
- Fixed (not learned); works for sequences longer than training sequences

### Learned Positional Embeddings
Treat position vectors as trainable parameters, one per position. Same vector shared across sequences at the same position. Simple but limited to training sequence length.

### RoPE (Rotary Positional Encoding)

Used in: LLaMA 1/2, PaLM, GPT-NeoX. Key innovation: **encode absolute position via rotation, recover relative position in the attention dot product**.

**Idea:** The dot product of query and key should only depend on the **relative** position $m - n$, not the absolute positions $m$ and $n$ separately.

Achieved by multiplying query and key vectors by a rotation matrix $R_{\theta_m}$ (depending on absolute position $m$):

$$f_q(x_m, m) = W_Q x_m \cdot e^{im\theta}$$
$$f_k(x_n, n) = W_K x_n \cdot e^{in\theta}$$

When you compute $\langle f_q, f_k \rangle$, the absolute positions cancel and you get a function of $m - n$ only.

In $d$ dimensions, the rotation is applied pairwise across every 2 elements of the vector using block-diagonal rotation matrices with frequencies $\theta_i = 10000^{-2i/d}$.

**Advantages over sinusoidal:**
- Multiplicative (not additive) — integrates naturally into the QK dot product
- Preserves relative distance information exactly
- More efficient computation (decomposed into element-wise operations)
- Empirically faster convergence and better performance

---

## Position-Wise Feed-Forward Network

Self-attention is linear — a weighted average of values. To add nonlinearity, each position's output passes through a small 2-layer MLP independently:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

- $W_1, W_2$ are **shared across all token positions** (position-wise)
- Nonlinearity: ReLU in the original paper; GeLU in most modern variants
- Typically $d_\text{ff} = 4 \times d_\text{model}$

**Role:** While self-attention passes messages between tokens (reads from a distributed "memory"), the FFN processes each token's representation independently — adding nonlinear transformation capacity.

---

## Add & Norm (Residual + Layer Normalization)

After every sub-layer (self-attention and FFN), apply:

$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

**Residual connection** (`Add`): $x + F(x)$
- Prevents vanishing gradients by providing a direct gradient path to earlier layers
- Ensures the model doesn't forget its input through complex operations

**Layer Normalization** (`Norm`): for each token vector independently, normalize elements to have mean 0, variance 1, then apply learned scale $\gamma$ and shift $\beta$:

$$\hat{x}_j = \frac{x_j - \mu}{\sigma}, \quad \text{LayerNorm}(x) = \gamma \hat{x} + \beta$$

Layer norm vs. Batch norm:
- Batch norm: normalize across the batch (problematic for variable-length sequences)
- Layer norm: normalize across the feature dimension of a single token (sequence-length agnostic) — preferred for language models

---

## Encoder-Decoder Architecture

The full Transformer (for seq2seq tasks like translation):

**Encoder** (N stacked blocks):
- Each block: Multi-head self-attention → Add & Norm → FFN → Add & Norm
- All tokens attend to all other tokens (bidirectional)

**Decoder** (N stacked blocks):
- Each block: **Masked** multi-head self-attention → Add & Norm → **Cross-attention** → Add & Norm → FFN → Add & Norm

**Masked self-attention** in the decoder: at position $t$, can only attend to positions $\leq t$ (not future tokens). Implemented by setting future attention scores to $-\infty$ before softmax.

**Cross-attention**: query from decoder, keys and values from the **last encoder layer** (not each encoder layer separately — every decoder layer attends to the same final encoder output). This replaces the "decoder attending to all encoder states" from seq2seq attention.

**Training:** all decoder positions computed in parallel using teacher forcing. Loss is the sum of cross-entropies at each position, back-propagated end-to-end.

---

## Complexity and Parallelism

| | RNN | Transformer |
|--|-----|-------------|
| Sequential operations | $O(n)$ | $O(1)$ |
| Computation per layer | $O(n d^2)$ | $O(n^2 d)$ |
| Max dependency path length | $O(n)$ | $O(1)$ |
| Parallelizable | No | Yes |

The quadratic $O(n^2)$ attention cost is the main scaling challenge for very long sequences (motivating sparse attention, linear attention variants, etc.).

---

## Summary of Components

| Component | Purpose |
|-----------|---------|
| Token embedding | Map vocabulary IDs to dense vectors |
| Positional encoding | Inject sequence order (sinusoidal or learned) |
| Multi-head self-attention | Exchange information across all token pairs |
| Feed-forward (position-wise) | Nonlinear per-token transformation |
| Add & Norm (residual + LayerNorm) | Stable gradient flow, prevent covariate shift |
| Masked attention (decoder) | Prevent attending to future tokens |
| Cross-attention (decoder) | Connect decoder to encoder representations |

---

## Related

- [[Seq2Seq and Attention]] — motivation, encoder-decoder, vanilla attention
- [[RNNs]] — the predecessor; residual connections also appear there
- [[Embeddings]] — token embeddings as Transformer input
- [[Pre-training Strategies]] — BERT, GPT, T5 built on the Transformer backbone
