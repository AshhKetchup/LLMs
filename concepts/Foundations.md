---
tags: [llm, nlp, word2vec, foundations, classification, generation]
source_count: 1
last_updated: 2026-04-15
source: "Lec 01–06 | NLP Tasks, Tokenization, Word Representations — Introduction to LLMs playlist"
---

# Foundations

This page covers the core NLP task taxonomy, tokenization, and the first generation of word representations (Word2Vec) that set the stage for everything that follows.

---

## NLP Task Taxonomy

### Classification (Single Output)

Input text → model → **one label**

- Sentiment analysis (`positive` / `negative` / `neutral`)
- Intent detection (`book_flight`, `cancel_order`, …)
- Language identification

### Multi-label / Sequence Classification (Multiple Outputs)

Input text → model → **one label per token**

- Named Entity Recognition (NER) — tag each token as `PERSON`, `ORG`, `LOC`, `O`, …
- Part-of-Speech (POS) tagging — tag each token as noun, verb, adjective, …

### Generation

Input text → model → **output text**

- Machine translation
- Question answering
- Summarization
- Text completion

---

## Tokenization

Tokenization breaks raw text into **tokens** — the atomic units fed to a model.

Three strategies:

| Strategy | Unit | Vocab size | OOV handling |
|----------|------|-----------|--------------|
| Word-based | Full words | Large (~50k+) | Unknown `<UNK>` token |
| Character-based | Individual characters | Tiny (~100) | Always handles OOV |
| Subword-based | Frequent subwords | Medium (~30k) | Decomposes rare words |

Subword tokenization (BPE, WordPiece, SentencePiece) dominates modern LLMs — it balances vocabulary size against OOV coverage.

→ See [[Tokenization]] for full detail on BPE, WordPiece, and SentencePiece.

---

## Token Representation

After tokenization, each token needs a numeric representation the model can compute over. The pipeline is:

```
raw text → tokenizer → token IDs → embedding lookup → dense vectors
```

Early approaches (one-hot, TF-IDF) were sparse and carried no semantic information. The goal of representation learning is to produce **dense vectors** that encode meaning.

→ See [[Embeddings]] for count-based vs. prediction-based representations.

---

## Word2Vec

Word2Vec (Mikolov et al., 2013) learns dense word vectors from raw text using a single principle:

> "A word is known by the company it keeps." — J.R. Firth

If two words consistently appear in similar contexts (similar surrounding words), the model assigns them similar vectors.

**Famous analogy test:**
$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

### Training Objectives

Word2Vec has two equivalent formulations:

#### Skip-Gram

Given a **center word**, predict its **surrounding words**.

```
Sentence: "I love machine learning"
Input:    "love"
Targets:  ["I", "machine"]
```

The model is a shallow neural net: center word (one-hot) → embedding matrix $W$ → hidden vector → output matrix $W'$ → softmax over vocabulary.

#### CBOW (Continuous Bag of Words)

Given the **surrounding words**, predict the **center word**.

```
Input:  ["I", "machine"]
Target: "love"
```

CBOW averages the context embeddings before the output layer. Generally faster to train; skip-gram tends to produce better embeddings for rare words.

### Negative Sampling

Training the full softmax over a 50k+ vocabulary is expensive. Negative sampling replaces it: for each positive (word, context) pair, sample $k$ random "negative" words and train the model to distinguish real context words from noise.

This makes training tractable and is the default in practice.

### Limitation — Static Embeddings

Word2Vec produces **one vector per word** regardless of context:

- "I need to go to the **bank** to deposit money."
- "We sat by the river **bank** at sunset."

Both uses of "bank" get the **same vector** — the model has no way to disambiguate polysemy.

This is the key limitation motivating contextual embeddings (ELMo, BERT).

---

## Attention (Overview)

Attention is a mechanism that lets a model **selectively focus** on different parts of the input when producing each output token, instead of compressing the entire input into a single fixed vector.

The core idea:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Query (Q)**: what the current position is looking for
- **Key (K)**: what each position offers
- **Value (V)**: the actual content passed forward if attended to

The dot product $QK^T$ scores compatibility; dividing by $\sqrt{d_k}$ prevents the values from growing too large in high-dimensional spaces; softmax normalizes into a probability distribution; the result is a weighted sum of values.

→ See [[Seq2Seq and Attention]] for the attention mechanism in encoder-decoder models.
→ See [[Transformer]] for self-attention and multi-head attention in full detail.

---

## Related

- [[Tokenization]] — BPE, WordPiece, SentencePiece
- [[Embeddings]] — one-hot problems, count vs. prediction, fastText, static vs. contextual
- [[GloVe]] — count + prediction hybrid
- [[RNNs]] — sequence models that consume token embeddings
- [[Transformer]] — self-attention replaces recurrence entirely
