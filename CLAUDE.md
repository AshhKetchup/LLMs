# LLM Wiki — Schema

This folder is an LLM-maintained wiki on **Large Language Models, NLP, and ML foundations**.

## Structure

```
LLM/
  CLAUDE.md          ← this schema
  index.md           ← content catalog
  log.md             ← ingest/query/lint history
  LLM.canvas         ← visual index of all concept pages
  cheat sheet llm.canvas  ← existing visual overview
  concepts/
    Foundations.md           ← NLP tasks, Word2Vec, Attention overview
    Tokenization.md          ← BPE, WordPiece, SentencePiece
    Embeddings.md            ← static embeddings, fastText, count vs prediction
    GloVe.md                 ← co-occurrence ratio, weighted least-squares
    RNNs.md                  ← vanilla RNN, BPTT, LSTM, GRU
    Seq2Seq and Attention.md ← encoder-decoder, beam search, attention mechanism
    Transformer.md           ← self-attention, multi-head, RoPE, FFN, add&norm
    Pre-training Strategies.md ← ELMo, BERT, GPT, T5, BART
    Instruction Tuning.md    ← instruction datasets, Self-Instruct
    Prompting.md             ← zero/few-shot, CoT, tree/graph of thoughts
    Alignment.md             ← RLHF, reward model, PPO, DPO
```

## Conventions

- YAML frontmatter on every page:
  ```yaml
  ---
  tags: [llm, nlp, transformers]
  source_count: 1
  last_updated: YYYY-MM-DD
  ---
  ```
- Internal links: `[[PageName]]` or `[[Folder/PageName]]`.
- Stubs (empty pages) are listed in `index.md` with a ⚠️ marker.
- Math: use `$$...$$` LaTeX blocks for formulas.

## Operations

### Ingest
1. Discuss key takeaways with the user.
2. Write or update page(s) in the right subfolder (`concepts/`, `systems/`, `maths/`).
3. Promote stubs to full pages when content is available.
4. Update `index.md` and append to `log.md`.

### Query
1. Read `index.md`; drill into relevant pages.
2. Synthesize with citations; offer to file valuable answers as pages.

### Lint
Flag: stubs without content, missing links between related concepts, contradictions between pages.

## Domain Coverage

| Area | Subfolder | Key pages |
|------|-----------|-----------|
| Foundations & Representations | concepts/ | Foundations, Tokenization, Embeddings, GloVe |
| Sequence Models | concepts/ | RNNs, Seq2Seq and Attention |
| Transformer Architecture | concepts/ | Transformer |
| Pre-training & Fine-tuning | concepts/ | Pre-training Strategies, Instruction Tuning |
| Prompting & Alignment | concepts/ | Prompting, Alignment |
| Math | maths/ | (to be populated) |
| Systems | systems/ | (to be populated) |

## Concept Dependency Graph (for ordering)

Foundations → Tokenization → Embeddings → GloVe → RNNs → Seq2Seq and Attention → Transformer → Pre-training Strategies → Instruction Tuning → Prompting / Alignment
