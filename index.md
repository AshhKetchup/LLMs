# LLM — Index

_Last updated: 2026-04-15_

## Pages

| Page | Summary | Tags | Status |
|------|---------|------|--------|
| [Foundations](concepts/Foundations.md) | NLP tasks, Word2Vec (skip-gram, CBOW), tokenization intro, attention overview | llm, nlp, word2vec | ✅ |
| [Tokenization](concepts/Tokenization.md) | Word/character/subword strategies; BPE (GPT-2), WordPiece (BERT), SentencePiece (T5/LLaMA); vocab size tradeoffs | llm, tokenization, bpe | ✅ |
| [Embeddings](concepts/Embeddings.md) | One-hot problems; count-based (TF-IDF, PMI) vs prediction-based (Word2Vec); fastText n-grams; static vs contextual table | llm, embeddings, word-vectors | ✅ |
| [GloVe](concepts/GloVe.md) | Count+prediction hybrid; co-occurrence ratio intuition; weighted least-squares objective; weighting function f(X) | llm, embeddings, glove | ✅ |
| [RNNs](concepts/RNNs.md) | Vanilla RNN + BPTT; LSTM gates (forget/input/output), cell state; GRU update/reset gates; residual connections | llm, rnn, lstm, gru | ✅ |
| [Seq2Seq and Attention](concepts/Seq2Seq%20and%20Attention.md) | Encoder-decoder; teacher forcing; beam search + decoding strategies (top-k, top-p, temperature); attention mechanism; bottleneck problem | llm, seq2seq, attention | ✅ |
| [Transformer](concepts/Transformer.md) | Self-attention (QKV, scaled dot-product); multi-head attention; sinusoidal + RoPE positional encoding; FFN; add & norm; encoder-decoder architecture | llm, transformer, self-attention | ✅ |
| [Pre-training Strategies](concepts/Pre-training%20Strategies.md) | ELMo (contextual BiLSTM); BERT (MLM + NSP); GPT (causal LM); T5 (text-to-text, span corruption); BART (noise reconstruction) | llm, bert, gpt, t5, pre-training | ✅ |
| [Instruction Tuning](concepts/Instruction%20Tuning.md) | Teaching LLMs to follow instructions; instruction data format; training loss; scaling results; Self-Instruct automated data generation | llm, instruction-tuning, finetuning | ✅ |
| [Prompting](concepts/Prompting.md) | Zero/one/few-shot; chain-of-thought; self-consistency; tree/graph of thoughts; prompt sensitivity (POSIX metric) | llm, prompting, in-context-learning | ✅ |
| [Alignment](concepts/Alignment.md) | RLHF (SFT → reward model → PPO); DPO; RLAIF; KL penalty; reward hacking | llm, alignment, rlhf, dpo | ✅ |

## Concept Dependency Chain

```
Foundations → Tokenization → Embeddings → GloVe
                                ↓
                              RNNs → Seq2Seq & Attention → Transformer
                                                                ↓
                                                     Pre-training Strategies
                                                                ↓
                                                      Instruction Tuning
                                                                ↓
                                                    Prompting / Alignment
```

## Stubs to Fill (Optional — Advanced Topics)

- **PEFT** (LoRA, adapters, prompt tuning) — Lec 29 transcript available
- **Quantization** (INT8, GGUF, bitsandbytes) — Lec 30 transcript available
- **RAG** (Retrieval-Augmented Generation) — Lec 31+ transcripts available
- **Evaluation** (BLEU, ROUGE, BERTScore, LLM-as-judge) — covered later in playlist

## Cross-folder Links

- Transformer architecture underpins all LLM-based agents → [[../Agents/AI AGENTS]]
- Pre-training Strategies relates to fine-tuning pipelines → [[../Agents/Langchain and Langgraph]]
