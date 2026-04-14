# LLM Notes

Personal notes from the **NPTEL course on Large Language Models** by [Prof. Soumen Chakrabarti](https://www.cse.iitb.ac.in/~soumen/) — IIT Bombay.
![[Pasted image 20260415013530.png]]
---

## What's inside

Structured, concept-linked notes covering the full arc from word vectors to alignment — written to be used, not just read.

| Topic | What it covers |
|-------|---------------|
| [Foundations](concepts/Foundations.md) | NLP tasks, Word2Vec, skip-gram, CBOW |
| [Tokenization](concepts/Tokenization.md) | BPE, WordPiece, SentencePiece, vocab tradeoffs |
| [Embeddings](concepts/Embeddings.md) | TF-IDF, PMI, fastText, static vs contextual |
| [GloVe](concepts/GloVe.md) | Co-occurrence ratios, weighted least-squares |
| [RNNs](concepts/RNNs.md) | Vanilla RNN, BPTT, LSTM, GRU |
| [Seq2Seq & Attention](concepts/Seq2Seq%20and%20Attention.md) | Encoder-decoder, beam search, attention |
| [Transformer](concepts/Transformer.md) | QKV self-attention, multi-head, RoPE, FFN |
| [Pre-training Strategies](concepts/Pre-training%20Strategies.md) | ELMo, BERT, GPT, T5, BART |
| [Instruction Tuning](concepts/Instruction%20Tuning.md) | Instruction datasets, Self-Instruct |
| [Prompting](concepts/Prompting.md) | CoT, few-shot, tree/graph of thoughts |
| [Alignment](concepts/Alignment.md) | RLHF, reward model, PPO, DPO |

---

## How to use

**Best experience: [Obsidian](https://obsidian.md)**

1. Clone the repo
2. Open the folder as an Obsidian vault
3. Start from [`index.md`](index.md) or open `LLM.canvas` for the visual map
4. Follow the concept dependency chain — each note links to the next

**Pair with:** [Andrej Karpathy's LLM Wiki](https://github.com/karpathy/LLM101n) for implementation-level intuition alongside the theory here.

**Claude users:** Drop any note into Claude and ask questions — the notes are written with enough structure and LaTeX math that Claude can reason over them well.

---

## Concept order

```
Foundations → Tokenization → Embeddings → GloVe
                                  ↓
                                RNNs → Seq2Seq & Attention → Transformer
                                                                   ↓
                                                        Pre-training Strategies
                                                                   ↓
                                                         Instruction Tuning
                                                                   ↓
                                                       Prompting · Alignment
```

---

*Course: [NPTEL — Large Language Models, IIT Bombay](https://nptel.ac.in)*
