---
tags: [llm, nlp, seq2seq, attention, encoder-decoder, beam-search, decoding]
source_count: 3
last_updated: 2026-04-14
source: "Lec 12 | Seq2Seq + Lec 13 | Decoding Strategies + Lec 14 | Attention — Introduction to LLMs playlist"
---

# Seq2Seq and Attention

Sequence-to-sequence models map an input sequence to an output sequence of (potentially) different length. They underpin machine translation, summarization, dialogue, and other generation tasks.

---

## Seq2Seq Architecture

Uses **two RNNs** (or LSTMs/GRUs):

- **Encoder** — scans the input sequence, producing a series of hidden states. The final hidden state $h_\text{enc}$ is passed to the decoder
- **Decoder** — generates the output sequence token by token, starting from $h_\text{enc}$

> **Analogy:** Like a translator who reads the whole French sentence first (encoder), then starts speaking in English (decoder).

This is a **conditional language model**: the probability factored as:

$$P(y_1, \ldots, y_T | x) = \prod_{t=1}^{T} P(y_t | y_1, \ldots, y_{t-1}, x)$$

**Encoder hidden state weights:** $W_h, W_e$ (shared across encoder steps)
**Decoder hidden state weights:** $W_h', W_e', U$ (U is the output projection; shared across decoder steps)
**Training:** end-to-end backpropagation through time (BPTT) through both encoder and decoder.

### Teacher Forcing
During training: instead of feeding the decoder's predicted output as the next input, feed the **ground truth** token. This stabilizes training (the decoder doesn't spiral due to early errors) but means the model sees a different input distribution at inference time.

---

## Decoding Strategies

At inference time, the decoder must select a token from the output distribution at each step.

### Greedy Decoding
Always pick the highest-probability token.
- Fast, deterministic
- No backtracking — a wrong early choice is never corrected

### Exhaustive Search
Explore all $|V|^T$ possible output sequences. Impractical for any realistic vocabulary.

### Beam Search
Keep the top-$k$ hypotheses (beams) at each step:
1. Start with the `<start>` token
2. At each step, expand each of the $k$ hypotheses by generating the top-$k$ next tokens → $k^2$ candidates
3. Keep only the $k$ highest-scoring candidates
4. Continue until all beams hit `<EOS>` or max length is reached

**Score** = sum of log probabilities. **Normalization by length** is essential: longer sequences accumulate more negative log-probs, so without normalization short sequences are always preferred.

### Sampling Strategies (for diversity)

| Strategy | Idea |
|----------|------|
| **Random sampling** | Sample directly from full distribution — too noisy |
| **Top-$k$ sampling** | Sample only from top-$k$ tokens; renormalize | Good if distribution is peaked |
| **Top-$p$ (nucleus)** | Sample from smallest set of tokens whose cumulative probability ≥ $p$; renormalize | Adapts to distribution shape |
| **Temperature sampling** | Replace softmax with $\text{softmax}(x/\tau)$. Low $\tau$ → peakier (more deterministic); high $\tau$ → flatter (more diverse) | Doesn't truncate distribution |

Greedy/beam/exhaustive = **deterministic** (good for translation, QA). Sampling strategies = **stochastic** (good for creative text, dialogue).

---

## The Bottleneck Problem

In a vanilla seq2seq model, the **entire source sequence** must be compressed into the encoder's final hidden state before the decoder can begin. This creates a bottleneck:
- Long sequences strain the single vector's capacity
- Vanishing gradients mean early encoder states are weakly represented in the final state
- The decoder has no direct access to early encoder states

> **Analogy:** Like the translator trying to cram an entire French novel into one Post-It note before starting to speak English — critical information gets lost.

**Solution:** Attention.

---

## Attention Mechanism

Core idea: at each decoder step, instead of relying only on the final encoder state, give the decoder a **direct connection to all encoder states**.

### Vanilla (Additive / Dot-Product) Attention

At decoder step $t$ with hidden state $s_t$:

1. **Attention scores** — measure similarity between decoder state (query) and each encoder state (keys):

$$e_{t,i} = s_t \cdot h_i \quad \text{for each encoder state } h_i$$

2. **Attention distribution** — normalize with softmax:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}$$

3. **Attention vector** — weighted sum of encoder states (values):

$$a_t = \sum_i \alpha_{t,i} \cdot h_i$$

4. **Decoder input** — concatenate attention vector with decoder hidden state, then project:

$$\text{output}_t = U \cdot [s_t; a_t]$$

> **Analogy:** Now the translator can glance back at specific words in the French text while forming each English word — no more relying on that single Post-It note.

**No new parameters are introduced** — just dot products, softmax, and weighted sums using the existing encoder/decoder weights.

### Why Attention Helps

| Problem | Fixed by attention? |
|---------|---------------------|
| Bottleneck | Yes — decoder can access all encoder states directly |
| Vanishing gradients | Partially — shortcuts exist from decoder to early encoder states |
| Interpretability | Yes — $\alpha_{t,i}$ forms an **alignment matrix** showing which source tokens the decoder attends to when generating each target token |

The attention matrix also provides word alignment for free — in statistical MT this required a separate alignment model.

### Variants

| Name | Formula |
|------|---------|
| Dot product | $s \cdot h$ |
| Bilinear | $s^T W h$ |
| Additive (Bahdanau) | $v^T \tanh(W_1 s + W_2 h)$ |
| Scaled dot product (Transformer) | $\frac{s \cdot h}{\sqrt{d_k}}$ — prevents dot products from growing large with vector size |

---

## From Attention to Self-Attention

In vanilla attention, the **query** comes from the decoder and the **keys/values** come from the encoder. But attention is a generic operation: you can apply it where both query and keys/values come from the **same sequence**.

**Key observation:** if a decoder state can attend to all encoder states directly, do we still need the recurrent connections at all? Removing them would allow full parallelism.

This question motivates **self-attention** and the Transformer — see [[Transformer]].

---

## Related

- [[RNNs]] — encoder and decoder building blocks
- [[Transformer]] — replaces recurrence with full self-attention
- [[Embeddings]] — token representations fed into the encoder
- [[Foundations]] — language modeling task setup
