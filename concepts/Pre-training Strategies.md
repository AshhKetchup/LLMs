---
tags: [llm, nlp, pre-training, bert, gpt, elmo, t5, bart, transfer-learning]
source_count: 2
last_updated: 2026-04-14
source: "Lec 18 | ELMo + Lec 19 | BERT, GPT, T5 — Introduction to LLMs playlist"
---

# Pre-training Strategies

Pre-training captures general language knowledge from large unlabeled corpora. The pre-trained model is then **fine-tuned** on small labeled task datasets.

> **Analogy:** Pre-training is like giving a new employee a year to read every book in the library. Fine-tuning is then giving them a 2-day crash course on their specific role — they bring vast background knowledge to learn the job quickly.

> "The complete meaning of a word is always contextual." — J.R. Firth (revised)

Static embeddings (Word2Vec, GloVe) give one vector per word regardless of context. "bank" in "river bank" and "bank account" get the same vector. **Contextual embeddings** fix this.

---

## ELMo (Embeddings from Language Models)

Proposed 2018 (Peters et al., Allen AI). First widely-adopted contextual embedding approach; RNN-based, not Transformer.

**Architecture:** Two stacked BiLSTMs — one left-to-right, one right-to-left — trained with a language modeling objective (next-word prediction in each direction).

**Producing contextual embeddings:**
1. Pre-train the bidirectional LSTM on a large corpus (language model objective only — no task supervision)
2. Freeze the weights
3. For a downstream task, feed input tokens through the frozen LSTM → collect all hidden states from all layers for each token
4. The contextual embedding = **learned weighted combination** of all layer representations:

$$\text{ELMo}_k = \gamma \sum_{j=0}^{L} s_j \mathbf{h}_{k,j}$$

- $s_j$ = task-specific softmax weights over layers (learned per task)
- $\gamma$ = task-specific scaling factor
- This allows different tasks to emphasize different layers (lower layers = syntax, upper layers = semantics)

**Key innovation vs. Word2Vec:** The word "bat" in "bat flew out of the cave" and "bat flew out of the cave, he hit the ball with a bat" gets two different representations because the hidden state is a function of all preceding context.

**Limitation:** Still fundamentally uni-directional per LSTM (one left-to-right, one right-to-left — but not truly bidirectional within a single pass). Also uses character-level CNN for token embeddings to handle OOV words.

---

## Pre-training Paradigm Shift

ELMo pre-trains only the **embedding layer**. The Transformer-era insight: pre-train the **entire model** (all attention heads, FFN layers, etc.) and then fine-tune. Three architectures emerge:

| Type | Models | Self-attention | Use case |
|------|--------|---------------|----------|
| Encoder-only | BERT, RoBERTa | Bidirectional (unmasked) | Classification, NLU |
| Decoder-only | GPT-1/2/3, LLaMA | Causal (masked) | Generation, LLM |
| Encoder-decoder | T5, BART | Encoder: unmasked; Decoder: masked | Seq2seq, translation |

---

## BERT (Bidirectional Encoder Representations from Transformers)

Proposed by Google (Devlin et al., 2018).

**Architecture:** Encoder-only Transformer. Because there is no decoder, BERT can attend **bidirectionally** — every token attends to all other tokens.

### Pre-training Objectives

#### 1. Masked Language Modeling (MLM)
Self-supervised: no labeled data needed.

1. Sample 15% of tokens to process
2. Of those:
   - 80% → replace with `[MASK]` token
   - 10% → replace with a random token
   - 10% → keep unchanged
3. Predict the original token only at masked positions (loss computed only there)

Why not mask 50%? Too much context is destroyed. Why the 10% random + 10% unchanged? To prevent the model from only learning to predict `[MASK]` tokens and ignoring non-masked positions.

> **Example:** Like a fill-in-the-blank exercise — "The ___ flew over the river." The model learns by guessing what word was masked, using surrounding context.

#### 2. Next Sentence Prediction (NSP)
Binary classification: given sentence A and sentence B, predict whether B follows A in the original corpus.

Input format:
```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

Three embedding types are added:
- Token embeddings
- Positional embeddings (learned, not sinusoidal)
- Segment embeddings (A vs. B segment)

**[CLS] token:** Prepended to every input. Because it carries no lexical meaning, it's unbiased toward any specific token. Its final hidden state is used for sentence-level classification tasks. Attends to all other tokens via self-attention, so it summarizes the full input.

### Fine-tuning BERT

For a downstream task, add a small task-specific head (linear layer) on top of:
- `[CLS]` representation → sentence classification, NLI
- Per-token representations → NER, POS tagging, QA span extraction

Only 6% of compute vs. pre-training is needed for fine-tuning. End-to-end backprop updates the head and all BERT layers.

**Question answering example (SQuAD):** Input = `[CLS] Question [SEP] Passage [SEP]`. Add two linear heads (one for span start, one for span end) over all token positions. Train with cross-entropy on the correct start/end positions.

### BERT Variants

| Model | Key Difference |
|-------|---------------|
| BERT-base | 12 encoder layers, 12 heads, 110M params |
| BERT-large | 24 encoder layers, 16 heads, 340M params |
| RoBERTa | Removes NSP, dynamic masking, larger batches |
| DistilBERT | Knowledge distillation → 40% smaller, 60% faster |

---

## GPT (Generative Pre-trained Transformer)

Decoder-only Transformer. **Causal language modeling** (CLM) objective: predict the next token given all previous tokens. Attention is always left-to-right (masked self-attention).

**Training:** Large text corpora, next-token prediction, all positions in parallel during training (teacher forcing), cross-entropy loss summed across positions.

**Key insight (GPT-3):** A large enough decoder-only model pre-trained on enough data can perform tasks **without any fine-tuning**, just from natural language prompts (few-shot prompting). This revealed that scale enables emergent instruction-following.

---

## T5 (Text-to-Text Transfer Transformer)

Proposed by Google (Raffel et al., 2020). Encoder-decoder architecture.

**Core idea:** Every NLP task is reframed as a **text-to-text** problem. The model always produces text, regardless of the task:

| Task | Input | Output |
|------|-------|--------|
| Translation | `translate English to German: Hello` | `Hallo` |
| Sentiment | `cola sentence: The course is jumping well` | `not acceptable` |
| Similarity | `stsb sentence1: ... sentence2: ...` | `3.8` |
| Summarization | `summarize: ...` | summary text |

Note: even numeric outputs like `3.8` are generated as strings. The model learns this from pre-training on text-to-text examples.

**Pre-training:** Span corruption (similar to BERT's masking but applied to spans). Randomly mask consecutive spans of tokens, replacing each with a unique Sentinel token (`<extra_id_0>`, `<extra_id_1>`, etc.). The decoder only predicts the masked spans, not the full sequence.

Example:
- Input: `Thank you <extra_id_0> me to your <extra_id_1> week`
- Target: `<extra_id_0> for inviting <extra_id_1> party last <extra_id_2>`

**Training data:** Cleaned Common Crawl (C4 dataset) — removed non-English, non-terminal-punctuation lines, deduplicated, removed code/HTML.

---

## BART (Bidirectional and Auto-Regressive Transformers)

Proposed by Facebook AI (Lewis et al., 2019). Encoder-decoder; combines BERT-style bidirectional encoding with GPT-style autoregressive decoding.

**Pre-training:** Inject noise into input text in various ways, then train the decoder to reconstruct the original:

| Corruption method | Description |
|------------------|-------------|
| Token masking | Replace tokens with `[MASK]` (like BERT) |
| Token deletion | Delete tokens entirely (length changes) |
| Text infilling | Mask spans of varying length with a single `[MASK]` |
| Sentence permutation | Shuffle sentences randomly |
| Document rotation | Rotate to start from a random token |

The decoder always regenerates the **full original sequence** (not just the masked parts) — different from T5 which only generates the masked spans.

BART is particularly strong for summarization and generation tasks where the decoder needs to produce fluent, complete text.

---

## Summary

| Method | Architecture | Pre-train Obj | Strengths |
|--------|-------------|---------------|-----------|
| ELMo | BiLSTM | Language model | First contextual; good for NLU |
| BERT | Encoder-only | MLM + NSP | Strong NLU, classification |
| GPT | Decoder-only | Causal LM | Generation, few-shot prompting |
| T5 | Encoder-decoder | Span corruption | Unified text-to-text, versatile |
| BART | Encoder-decoder | Noise reconstruction | Summarization, generation |

---

## Related

- [[Transformer]] — the shared architecture backbone
- [[Embeddings]] — static embeddings that pre-training replaced
- [[Instruction Tuning]] — what happens after pre-training: teaching instruction-following
- [[Foundations]] — the original NLP tasks these models tackle
