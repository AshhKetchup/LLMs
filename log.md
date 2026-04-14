# LLM — Log

_Append-only. Format: `## [YYYY-MM-DD] <operation> | <title>`_

---

## [2026-04-15] update | concepts/Foundations.md
Rewrote page to match wiki conventions: added YAML frontmatter, LaTeX formulas (Word2Vec analogy, scaled dot-product attention), NLP task taxonomy table, tokenization strategy table, negative sampling explanation, static embedding limitation, and full attention overview (QKV formula). Removed emoji-prefixed rough notes.

## [2026-04-14] ingest | concepts/Foundations.md
Initial wiki setup. Page covers: NLP task types (classification, multi-classification, generation), Tokenization intro, token representation, Word2Vec (skip-gram & CBOW), polysemy limitation, links to Embeddings/Tokenization/Transformer/RNNs, and Attention overview stub.

## [2026-04-14] ingest | concepts/RNNs.md
Initial wiki setup. Page covers: RNN intuition (predict next word step-by-step), hidden state formula h_t = tanh(Wx*xt + Wh*h(t-1)), comparison table with Word2Vec, vanishing gradient problem, path to LSTM/GRU/Transformers.

## [2026-04-14] ingest | concepts/Tokenization.md
Rewrote stub from Lec 09 transcript. Covers: word/character/subword strategies, BPE algorithm with example, WordPiece (## prefix), SentencePiece (▁ space), vocab size trade-off table, model comparison table (GPT-2/BERT/T5/LLaMA).

## [2026-04-14] ingest | concepts/Embeddings.md
Wrote from Lec 07 transcript. Covers: one-hot problems, count-based (TF-IDF, PMI/PPMI) vs prediction-based (Word2Vec), fastText n-gram innovation (OOV, morphology), static vs contextual comparison table, analogy properties, societal bias warning, temporal semantic drift.

## [2026-04-14] ingest | concepts/GloVe.md
Rewrote stub from Lec 08 transcript. Covers: count vs prediction trade-off, co-occurrence ratio intuition (ice/steam), weighted least-squares objective (LaTeX), weighting function f(X) with 3 properties, two-embedding-per-word explanation.

## [2026-04-14] ingest | concepts/RNNs.md
Rewrote from Lec 10+11 transcripts. Covers: CNN language model problems, vanilla RNN formulas, BPTT + vanishing/exploding gradients, LSTM (3 gates, cell state), GRU (2 gates), bidirectional/multilayer variants, residual connections.

## [2026-04-14] ingest | concepts/Seq2Seq and Attention.md (new page)
Created from Lec 12+13+14 transcripts. Covers: encoder-decoder seq2seq, teacher forcing, greedy/beam/exhaustive decoding, top-k/top-p/temperature sampling, bottleneck problem, vanilla attention (QKV dot-product), attention alignment matrix.

## [2026-04-14] ingest | concepts/Transformer.md
Created from Lec 15+16+17 transcripts. Covers: self-attention (scaled dot-product), multi-head attention, sinusoidal positional encoding, RoPE, position-wise FFN, add & norm (residual + layer norm), masked attention, cross-attention, encoder-decoder architecture.

## [2026-04-14] ingest | concepts/Pre-training Strategies.md (new page)
Created from Lec 18+19+20 transcripts. Covers: ELMo (BiLSTM, contextual embeddings), BERT (MLM + NSP, [CLS] token, fine-tuning), GPT (causal LM, few-shot), T5 (text-to-text, span corruption, C4), BART (noise reconstruction).

## [2026-04-14] ingest | concepts/Instruction Tuning.md (new page)
Created from Lec 22 transcript. Covers: base model limitations, instruction dataset format, teacher forcing during training, scaling results (more tasks + bigger model = better), Self-Instruct automated data generation.

## [2026-04-14] ingest | concepts/Prompting.md (new page)
Created from Lec 23+24 transcripts. Covers: zero/one/few-shot prompting, chain-of-thought, self-consistency, tree of thoughts, graph of thoughts, prompt sensitivity and POSIX metric.

## [2026-04-14] ingest | concepts/Alignment.md (new page)
Created from Lec 25 transcript. Covers: RLHF three stages (SFT → reward model → PPO), pairwise preference training, KL penalty, DPO (direct preference optimization), RLAIF, reward hacking.

## [2026-04-14] setup | wiki infrastructure created
Created CLAUDE.md (schema), index.md, log.md, and LLM.canvas (visual index).
