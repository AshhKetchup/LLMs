---
tags: [llm, nlp, tokenization, bpe, wordpiece, sentencepiece]
source_count: 1
last_updated: 2026-04-14
source: "Lec 09 | Tokenization Strategies — Introduction to LLMs playlist"
---

# Tokenization

Tokenization is the process of splitting a running text into discrete units (tokens) that a model can process. It is a critical first step — the choice of tokenization strategy directly affects how a model handles unknown words, vocabulary size, and downstream task performance.

## Why Tokenization Matters

Before neural language models, simple delimiters (space, tab, punctuation, newline) were used to split text. These break down when the model encounters:
- Unknown tokens (OOV — out of vocabulary words)
- Compound words, morphological variations, emojis, code
- Languages without clear word boundaries (Chinese, Japanese, Thai)

The field of tokenization has been a core area in LLMs, neural machine translation, and neural language models in general.

---

## Three Main Strategies

### 1. Word-level Tokenization
Split on whitespace and punctuation. Each word = one token.

**Problems:**
- Large vocabulary (hundreds of thousands of entries)
- Unknown words (`<UNK>`) are a black box — model learns nothing about them
- "running", "ran", "runs" are three separate tokens despite shared root

---

### 2. Character-level Tokenization
Each character = one token.

> **Analogy:** Like spelling out every word letter-by-letter in Morse code — technically nothing is lost, but it takes forever and you lose the natural structure of words.

**Advantages:**
- No unknown tokens — any string is representable
- Small vocabulary (26 chars for English + symbols)

**Problems:**
- Sequences become very long — "hello" = 5 tokens
- Harder to learn meaningful representations (model must learn that h-e-l-l-o is a word)

---

### 3. Subword Tokenization (dominant approach)
Split words into frequently occurring subword units. Balances vocabulary size with handling of unknown words.

**Key insight:** Rare or unknown words are decomposed into known subword pieces. Common words remain intact.

#### Byte-Pair Encoding (BPE)
Originally a data compression algorithm, adapted for NLP (used by GPT-2, RoBERTa).

> **Analogy:** Like compressing a zip file — find repeated patterns in the text and replace them with shorthand symbols. "ing" appears a thousand times, so give it its own token instead of spelling it out each time.

**Algorithm:**
1. Start with character-level vocabulary
2. Count all character pairs in the training corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches desired size

**Example:**
- Corpus: "low", "lower", "lowest", "newer", "new"
- Frequent pair: `e` + `r` → `er`
- `er` + `s` → `ers` (if frequent enough)
- Result: ["low", "lower", "low##est", "new##er", "new"]

#### WordPiece (BERT, DistilBERT)
Similar to BPE but merges based on **likelihood** of the language model rather than raw frequency.
- Tokens prefixed with `##` indicate continuation: `playing` → `play`, `##ing`

#### SentencePiece (T5, LLaMA, multilingual models)
- Treats the raw text (including spaces) as input — no pre-tokenization needed
- Language-agnostic: works for Chinese, Japanese, etc.
- Implements BPE or Unigram Language Model under the hood
- Spaces are represented by a special symbol (`▁`)

---

## Vocabulary Size Trade-off

| Vocab Size | Pros | Cons |
|------------|------|------|
| Too small | Fewer parameters | More `<UNK>`, worse rare-word handling |
| Too large | Precise tokenization | Sparse embeddings, slow softmax |
| Optimal (~30k–50k) | Balance | Model-dependent |

---

## Tokenization in Practice

Modern LLMs (GPT, BERT, LLaMA, T5) all use subword tokenization:

| Model | Method | Vocab Size |
|-------|--------|-----------|
| GPT-2/3/4 | BPE | 50,257 |
| BERT | WordPiece | 30,522 |
| T5, LLaMA | SentencePiece (BPE) | 32,000 |

Tokenization happens **before** embeddings. The token IDs are looked up in an embedding table to get dense vectors.

---

## Related

- [[Foundations]] — NLP overview, where tokenization fits
- [[Embeddings]] — what happens after tokenization
- [[RNNs]] — neural models that consume token sequences
- [[Transformer]] — attention operates on token sequences
