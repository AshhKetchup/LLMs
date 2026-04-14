---
tags: [llm, nlp, prompting, in-context-learning, chain-of-thought, few-shot]
source_count: 1
last_updated: 2026-04-14
source: "Lec 23 | Prompting + Lec 24 | Advanced Prompting — Introduction to LLMs playlist"
---

# Prompting

Pre-trained large language models can perform tasks without fine-tuning, using only **natural language instructions** written in the input (the prompt). This behavior emerges at scale — small models don't exhibit it.

---

## Why Prompting?

Classical paradigm: pre-train → fine-tune → predict. Fine-tuning requires labeled task data and compute.

GPT-3 (2020) showed: a 175B parameter model pre-trained on enough data can solve downstream tasks from **prompts alone** — no weight updates. This is called **in-context learning**.

> **Analogy:** Like showing a new hire 3 example emails before asking them to draft one themselves — they learn the pattern from context, without formal training.

- Model parameters are **frozen** at inference time
- Task performance comes from the prompt structure and examples
- Trade-off: accuracy typically lower than fine-tuned models, but no labeled data needed

---

## Zero-Shot, One-Shot, Few-Shot

| Type | Description | Example prompt structure |
|------|-------------|--------------------------|
| **Zero-shot** | Only task description, no examples | `Translate English to French: cheese =` |
| **One-shot** | One (instruction, input, output) example | Instruction + 1 example + target input |
| **Few-shot** | Several examples | Instruction + N examples + target input |

More examples generally improve accuracy, up to the context window limit. But there is no rule for the optimal number — task-dependent.

**Key concern:** there is no principled way to determine the optimal prompt. Prompt engineering is empirical and human-driven.

---

## Scaling Effect

As model size increases:
- Zero/few-shot accuracy increases for the same prompt
- Emergent behaviors appear only above certain size thresholds
- Larger models benefit more from instruction tuning and few-shot examples

Parameter counts across models: ELMo (93M) → BERT-base (110M) → GPT-3 (175B) → GPT-4 (unknown, estimated >> 175B).

---

## Chain-of-Thought (CoT) Prompting

Standard few-shot: provide (question, answer) pairs. For multi-step reasoning tasks, the model often produces wrong answers.

**Chain-of-thought:** include **step-by-step reasoning** in the examples:

```
Q: Roger has 5 tennis balls. He buys 2 cans with 3 balls each.
   How many balls does he have?
A: Roger started with 5. Two cans with 3 = 6 more. 5 + 6 = 11. Answer: 11.

Q: The cafeteria had 23 apples. They used 20 for lunch and bought 6 more.
   How many?
A: [model generates: 23 - 20 = 3 remaining. 3 + 6 = 9. Answer: 9.]
```

> **Analogy:** Like showing your work in a math exam — forcing the model to reason step-by-step reduces mistakes, just like it does for humans.

**Zero-shot CoT:** Just add "Let's think step by step." before the question — even without examples, this triggers step-by-step reasoning.

### Self-Consistency
Ask the same question multiple times (temperature > 0 → different chains). Aggregate answers by **majority vote**. More robust than a single chain.

### Tree of Thoughts (ToT)
Branch the reasoning at each step; maintain multiple hypotheses simultaneously. Three components:
- **Branching module** — decides when to split into multiple sub-paths
- **Scoring module** — evaluates whether a branch is promising
- **Backtracking module** — prunes dead ends and returns to earlier states

Example: the "Game of 24" — given four numbers (4, 9, 10, 13), find operations that equal 24. A linear chain often fails; tree search succeeds by exploring and pruning.

### Graph of Thoughts (GoT)
Extends tree-of-thought by allowing branches to **merge** — earlier thoughts can be reused and combined across branches. An aggregator module determines when to merge two reasoning paths.

---

## Prompt Sensitivity

Models are often highly sensitive to small wording changes in the prompt. The same question phrased differently can produce wildly different responses.

**POSIX (Prompt Sensitivity Index):** A metric for measuring this, capturing four dimensions:
1. **Response diversity** — how much the model's outputs vary across paraphrases
2. **Response distribution entropy** — entropy over distinct outputs across paraphrases (higher = more sensitive)
3. **Semantic coherence** — cosine similarity between response embeddings (lower similarity = higher sensitivity)
4. **Variance in confidence** — variance of log-likelihoods across paraphrases

A model with lower accuracy but lower sensitivity (POSIX) can be preferred over one with higher accuracy but high sensitivity — robustness matters for production use.

---

## Prompting vs. Fine-tuning vs. In-Context Learning

| | Fine-tuning | Prompting (0/few-shot) |
|--|-------------|----------------------|
| Weight updates | Yes | No |
| Labeled data required | Yes | No (few-shot: a few) |
| Best accuracy | Yes | Lower |
| Flexibility | Fixed to task | Flexible |
| Compute cost | High (one-time) | None |

When to use prompting: no labeled data, exploring tasks, rapid prototyping, or when fine-tuning isn't feasible.

When to fine-tune: have labeled data, need highest accuracy, deploying at scale.

---

## Related

- [[Pre-training Strategies]] — the base models being prompted
- [[Instruction Tuning]] — teaching models to follow instructions (enables better zero-shot prompting)
- [[Alignment]] — RLHF further shapes model responses beyond instruction tuning
