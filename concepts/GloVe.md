---
tags: [llm, nlp, embeddings, glove, word-vectors]
source_count: 1
last_updated: 2026-04-14
source: "Lec 08 | Word Representation: GloVe — Introduction to LLMs playlist"
---

# GloVe (Global Vectors for Word Representation)

Proposed by Pennington, Socher, and Manning (Stanford, 2014). GloVe tries to **combine the best of count-based and prediction-based methods** into a single objective.

> "Utilize the best part of both worlds."

---

## Motivation: Count-Based vs Prediction-Based

| | Count-Based (TF-IDF, PMI) | Prediction-Based (Word2Vec) |
|--|--|--|
| **Speed** | Fast — scan corpus once, build matrix | Slow — scan every context window repeatedly |
| **Statistics** | Captures global co-occurrence faithfully | Doesn't directly use global statistics |
| **Performance** | Good at word similarity | Better on downstream tasks (sentiment, NER) |
| **Vectors** | Large sparse → needs SVD | Dense, small |
| **Analogies** | Poor | Good (`king − man + woman ≈ queen`) |
| **Rare words** | Handles poorly | Also struggles |

---

## Core Idea: Co-occurrence Ratio

Consider two target words: **ice** and **steam** (from thermodynamics).

For each context word, measure the probability that it appears with ice vs. steam:

| Context word | P(context \| ice) | P(context \| steam) | Ratio |
|---|---|---|---|
| solid | High | Low | **Large** |
| gas | Low | High | **Small** |
| water | High | High | ≈ 1 |
| fashion | Low | Low | ≈ 1 |

**Key insight from the ratio:**
- Large ratio → context word is specific to **ice** (e.g., "solid")
- Small ratio → context word is specific to **steam** (e.g., "gas")
- Ratio ≈ 1 → context word is either **irrelevant** (fashion) or **common to both** (water)

> **Example:** Noticing that "solid" comes up constantly near "ice" but almost never near "steam" — that asymmetry tells you something meaningful about both words.

This ratio captures more information than raw counts alone.

---

## Objective Function

GloVe models the **log of co-occurrence probability** using a dot product of embeddings:

$$
w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log(X_{ij})
$$

Where:
- $w_i$ = target word embedding for word $i$
- $\tilde{w}_j$ = context word embedding for word $j$
- $b_i, \tilde{b}_j$ = word-specific bias terms
- $X_{ij}$ = co-occurrence count of words $i$ and $j$

**Loss (weighted least squares):**

$$
J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

---

## The Weighting Function $f(X)$

Raw co-occurrence still gives disproportionate weight to very frequent pairs (stop word pairs).  
The weighting function $f(X_{ij})$ penalizes this:

$$
f(x) = \begin{cases} (x/x_{max})^\alpha & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}
$$

Properties of $f$:
1. **Continuous** and goes to 0 as $x \to 0$ (rare co-occurrences contribute little)
2. **Non-decreasing** — more co-occurrences = more weight
3. **Relatively small for large values** — very frequent pairs are down-weighted

Typical: $x_{max} = 100$, $\alpha = 3/4$

---

## Why Two Embeddings Per Word?

Like Word2Vec, GloVe learns **two embedding vectors** per word — one when the word is a target ($w_i$) and one when it's a context ($\tilde{w}_j$). At inference time, the final embedding is usually $w + \tilde{w}$ (element-wise sum).

---

## Properties

- **Static embedding** — one vector per word regardless of context
- **Fast to train** — builds co-occurrence matrix once, then optimizes
- **Scalable** — works well on large corpora
- **Good on small corpora** — unlike Word2Vec which needs large data
- Captures analogies: `king − man + woman ≈ queen`

---

## Analogy and Bias

Same properties as Word2Vec:
- Linear analogies work: `Beijing:China :: Delhi:India`
- Social biases are encoded: `father:programmer :: mother:homemaker`
- Semantic drift can be tracked across decades (train on time-sliced corpora)

---

## Related

- [[Embeddings]] — overview of static vs contextual embeddings, count vs prediction
- [[Foundations]] — Word2Vec (skip-gram, CBOW), the predecessor to GloVe
- [[Tokenization]] — what tokens get embedded
- [[RNNs]] — consume static embeddings as input
- [[Transformer]] — replaced static embeddings with contextual representations
