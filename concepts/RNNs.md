---
tags: [llm, nlp, rnn, lstm, gru, sequence-models, neural-language-model]
source_count: 2
last_updated: 2026-04-14
source: "Lec 10 | Neural Language Models (CNN/RNN) + Lec 11 | Advanced RNNs (LSTM/GRU) — Introduction to LLMs playlist"
---

# RNNs (Recurrent Neural Networks)

RNNs are the foundational sequence model in NLP. Unlike feedforward networks or CNNs, they process tokens one at a time while maintaining a **hidden state** — a compressed memory of everything seen so far.

---

## Why Not CNN for Language?

CNNs were the first neural language model approach. The setup:
1. Pick a fixed window of k words
2. Flatten into a 1D matrix
3. Apply a convolutional filter W
4. Project through linear layer → softmax over vocabulary

**Problems with CNN language models:**
1. **Fixed window** — can't see beyond the context window; long-range dependencies are lost
2. **No message passing** — when W multiplies the flattened embedding matrix, each word's embedding only interacts with its own column of W. There is no exchange of information between word positions
3. **W grows with window size** — larger context = more parameters

---

## Vanilla RNN

At each time step $t$, the RNN computes:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

$$y_t = \text{softmax}(W_y h_t)$$

- $x_t$ = token embedding at step $t$
- $h_{t-1}$ = hidden state from previous step (the "memory")
- $W_x, W_h, W_y$ = **shared weights across all time steps** (key insight — model size doesn't grow with sequence length)

> **Analogy:** Like reading a detective novel — each new sentence updates your mental model of whodunit. The hidden state is your running theory.

**Advantages over CNN:**
- Can handle sequences of any length
- Same weight matrices applied at every step (symmetry — fair chance for every word to influence the hidden state)
- Model size is constant regardless of context length

**Disadvantages:**
- Sequential — must compute $h_{t-1}$ before $h_t$; no parallelism
- Vanishing/exploding gradients with long sequences

---

## Backpropagation Through Time (BPTT)

Training RNNs requires differentiating through the recurrence. The gradient of the loss w.r.t. W requires summing across all time steps:

$$\frac{\partial L}{\partial W} = \sum_{t} \frac{\partial L}{\partial h_T} \prod_{k=t}^{T} \frac{\partial h_k}{\partial h_{k-1}} \cdot \frac{\partial h_t}{\partial W}$$

Each $\frac{\partial h_k}{\partial h_{k-1}}$ is a Jacobian matrix. If its largest eigenvalue is < 1, repeated multiplication makes gradients vanish; if > 1, they explode.

- **Vanishing gradient**: words far back in the sequence contribute almost nothing to the loss → the model forgets distant context
- **Exploding gradient**: addressed with **gradient clipping** (cap gradient norm at a threshold)
- **Truncated BPTT**: a practical fix — stop backpropagation at some fixed depth instead of propagating all the way to $t=0$

> **Analogy:** Vanishing gradient is like a game of telephone — by the time the message passes through 20 people, the original meaning is completely garbled.

---

## RNN Variants

### Bidirectional RNN
- Two RNNs: one left-to-right, one right-to-left
- At each position, concatenate both hidden states: $\tilde{h}_t = [h_t^{\rightarrow}; h_t^{\leftarrow}]$
- Useful when the full sequence is available (e.g., POS tagging, NER) and future context matters

### Multi-layer RNN
- Stack RNN layers where the input to layer $l$ is the hidden state output of layer $l-1$
- Different from bidirectional: depth (more abstraction) vs. direction (more context)

---

## LSTM (Long Short-Term Memory)

Proposed by Hochreiter & Schmidhuber (1997); widely adopted from ~2013 with large datasets.

LSTM adds an explicit **cell state** $c_t$ alongside the hidden state $h_t$. The cell state is the long-term memory; the hidden state is the current output.

Three gates control information flow (all vectors, values in $[0, 1]$ via sigmoid):

| Gate | Symbol | Function |
|------|--------|----------|
| Forget gate | $f_t$ | How much of $c_{t-1}$ to keep |
| Input gate | $i_t$ | How much of the new candidate to write to $c_t$ |
| Output gate | $o_t$ | How much of $c_t$ to expose as $h_t$ |

**Gate equations** (all conditioned on $h_{t-1}$ and $x_t$):
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$

**Cell content** (what we want to write, before gating):
$$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$

**Cell state update** (the key formula):
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Hidden state**:
$$h_t = o_t \odot \tanh(c_t)$$

**Intuition for vanishing gradient fix:** If $f_t = \mathbf{1}$ and $i_t = \mathbf{0}$, the cell state is copied unchanged — information from far back can persist indefinitely. The gates learn when to preserve vs. discard.

> **Analogy:** The cell state is like a sticky note you carry through a conversation — at each turn you decide what to write down, what to erase, and what to keep unchanged.

**Cost:** 4× the parameters of a vanilla RNN. Needs large data to avoid overfitting.

---

## GRU (Gated Recurrent Unit)

Proposed by Cho et al. (2014). Simpler than LSTM — no separate cell state, only two gates:

| Gate | Symbol | Function |
|------|--------|----------|
| Update gate | $z_t$ | Balance between old $h_{t-1}$ and new candidate |
| Reset gate | $r_t$ | How much of $h_{t-1}$ to use when computing candidate |

$$z_t = \sigma(W_z x_t + U_z h_{t-1})$$
$$r_t = \sigma(W_r x_t + U_r h_{t-1})$$
$$\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}))$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

The update gate $z_t$ balances old vs. new using a single parameter: high $z_t$ → replace old state; low $z_t$ → preserve old state.

> **Analogy:** GRU is like LSTM but you merged the sticky note and your short-term memory into one thing — simpler, fewer moving parts, does roughly the same job.

**LSTM vs. GRU:**
- No clear winner; empirically similar
- GRU has fewer parameters → more data-efficient
- Rule of thumb: start with LSTM, switch to GRU if you need efficiency

---

## Residual / Skip Connections

Another approach to vanishing gradients (not specific to RNNs; also used in Transformers):

$$\text{output} = F(x) + x$$

The shortcut provides a direct gradient path from the output back to earlier layers, preventing gradient death. Also used in Highway Networks and DenseNets (which generalize this to all-to-all connections).

---

## Limitations Remaining After LSTM/GRU

1. **Cannot parallelize** — position $t$ depends on $t-1$; sequences must be processed serially
2. **Long-distance dependencies** — improved but not fully solved
3. **No global view** — hidden state is a bottleneck for the full sequence

These limitations motivated the **attention mechanism** and ultimately **Transformers**.

---

## Related

- [[Foundations]] — Word2Vec, language model task setup
- [[Embeddings]] — static embeddings consumed by RNNs as $x_t$
- [[Seq2Seq and Attention]] — encoder-decoder architecture, attention, beam search
- [[Transformer]] — replaces RNN with self-attention + parallel processing
