---
tags: [llm, nlp, embeddings, word-vectors, representation]
source_count: 1
last_updated: 2026-04-14
source: "Lec 07 | Word Representation: Word2Vec & fastText — Introduction to LLMs playlist"
---

# Embeddings

A word embedding is a **dense, low-dimensional vector** that represents the meaning of a word. The key insight: words that appear in similar contexts should have similar vectors.

> **Analogy:** Word vectors are like coordinates on a city map — words with similar meanings live in the same neighborhood. "Cat" and "kitten" sit close together; "cat" and "democracy" are miles apart.

> "You shall know a word by the company it keeps." — J.R. Firth

---

## Why Not Just Use One-Hot Vectors?

One-hot encoding: a vocabulary of 50,000 words → 50,000-dimensional sparse vector, all zeros except one.

**Problems:**
- Dimensionality explosion
- No semantic information — "cat" and "kitten" are orthogonal, just as "cat" and "democracy"
- Dot product of any two different words = 0 (no similarity)

---

## Two Approaches to Word Representation

### 1. Count-Based Methods (co-occurrence matrices)

Build a word × word matrix where entry (i, j) = number of times words i and j appear in the same window.

Variants:
- **TF-IDF weighted** — downweights frequent, uninformative co-occurrences
- **PMI / PPMI** (Pointwise Mutual Information) — measures how much more often two words appear together than expected by chance

**Pros:** Fast to train, captures corpus-wide statistics well
**Cons:**
- Resulting matrix is huge and sparse
- Gives disproportionate weight to high-frequency pairs (stop-words dominate)
- Dense vector requires dimensionality reduction (SVD)

### 2. Prediction-Based Methods (Word2Vec)

Train a shallow neural network to **predict** context words from a target word (or vice versa). The weight matrix learned by the network becomes the embedding.

→ See [[Foundations#Word2Vec]] for skip-gram and CBOW details

**Pros:** Dense vectors, captures complex semantic analogies, generalizes well
**Cons:** Scales with corpus size (must scan every context window); doesn't use global co-occurrence statistics as directly

---

## fastText

Extension of Word2Vec by Facebook AI Research (Bojanowski et al., 2017).

**Key innovation:** Represents each word as a **bag of character n-grams**.

- "where" → `<wh`, `whe`, `her`, `ere`, `re>` + `<where>`
- The word embedding = sum of its n-gram embeddings

> **Example:** Even if you've never seen "unhappiness", fastText can embed it by combining what it knows about "un-", "happy", and "-ness" — like recognizing a new recipe from familiar ingredients.

**Advantages over Word2Vec:**
- Handles **morphologically rich languages** (Finnish, Turkish, Arabic) — suffixes/prefixes carry meaning
- Can embed **OOV words** — compose from known n-grams even if word was never seen
- Better on rare words — the n-grams may appear in other words

**Same training objectives** as Word2Vec (skip-gram or CBOW), just with subword units.

---

## Static vs Contextual Embeddings

All methods above produce **static embeddings**: one vector per word, regardless of context.

**Problem:** "bank" (river) and "bank" (financial) get the same vector.

| Method | Type | Context-aware? |
|--------|------|----------------|
| Word2Vec | Static | No |
| GloVe | Static | No |
| fastText | Static | No |
| ELMo | Contextual | Yes (LSTM-based) |
| BERT | Contextual | Yes (Transformer) |

→ See [[GloVe]] for the count+prediction hybrid
→ See [[Transformer]] and [[Pre-training Strategies]] for contextual embeddings

---

## Embedding Properties and Bias

Good embeddings capture semantic analogies:
- `king − man + woman ≈ queen`
- `Beijing − China + France ≈ Paris`

**Important warning:** These embeddings also encode societal biases from training data:
- `father − programmer + mother ≈ homemaker`
- `man − doctor + woman ≈ nurse`

This has been extensively studied and is a known problem with all static (and even contextual) embeddings.

---

## Temporal Embeddings (Fun Application)

Train separate embeddings on different decades (1950s–1960s, 1960s–1970s, ...).  
Compare nearest neighbors of a word across decades to track **semantic drift**:
- "broadcast" (1950s) → spreading seeds in agriculture
- "broadcast" (2000s) → radio/TV/media

---

## Related

- [[Foundations]] — Word2Vec, skip-gram, CBOW, analogy tests
- [[GloVe]] — global co-occurrence + prediction hybrid
- [[Tokenization]] — what gets embedded (subword tokens)
- [[RNNs]] — use embeddings as input sequences
- [[Transformer]] — self-attention over embedded token sequences
