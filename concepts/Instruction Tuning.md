---
tags: [llm, nlp, instruction-tuning, finetuning, rlhf, prompting]
source_count: 1
last_updated: 2026-04-14
source: "Lec 22 | Instruction Fine-Tuning — Introduction to LLMs playlist"
---

# Instruction Tuning

Pre-trained base models (trained only on next-word prediction) fail to follow instructions. When asked "What is the national flower of India?", a base model might continue generating similar questions rather than answering, because it's only doing text continuation.

**Instruction tuning** fine-tunes the model on a dataset of (instruction, input, output) triples to teach it to understand and follow natural language instructions.

---

## Why Instruction Tuning?

**Base model limitation:** Next-word prediction doesn't guarantee instruction-following. The model has learned language patterns but not how to respond to directives.

> **Example:** Ask a base model "What is the capital of France?" and it might respond with another geography question instead of "Paris" — it learned to predict text that looks like yours, not to answer questions.

**Multitask fine-tuning problem:** Classical multitask learning trains on multiple tasks using examples. When a new unseen task appears at test time, the model has no idea what to do — there is no explicit description of what the task requires.

**Solution:** Describe every task by a natural language instruction. Now the model learns the *pattern* of following instructions, so it can generalize to new tasks described by new instructions at inference time.

> **Analogy:** Like teaching someone "when someone asks a question, answer it" instead of just showing them a thousand question-answer pairs — they learn the principle, not just the examples.

---

## Instruction Dataset Format

Each example has three components:
1. **Instruction** — natural language description of the task
2. **Input** (optional) — the actual data to process
3. **Output** — the desired response

Example (translation):
```
Instruction: Translate the sentence to Spanish.
Input: The new office building was built in less than three months.
Output: El nuevo edificio de oficinas se construyó en menos de tres meses.
```

Key property: **training and test tasks are disjoint**. Evaluation measures how well the model generalizes to tasks it never saw during instruction tuning.

---

## Diversity in Instruction Phrasing

A single task can be described in many ways. Training on multiple phrasings of the same task improves generalization:

```
# NLI task — four possible templates
1. "Based on the paragraph, can we conclude that..."
2. "If the premise is true, can we infer the following?"
3. "Read the following and determine if the hypothesis can be inferred..."
4. "Does the premise entail the hypothesis? Options: yes / no"
```

All four describe the same natural language inference task. Training on diverse templates helps the model learn the **concept** of entailment, not the specific wording.

---

## Training Loss

For decoder-only models:
- Feed the full instruction + input + output as a single sequence
- Apply **teacher forcing**: use ground truth output tokens, not model predictions
- Compute cross-entropy loss **only on the output tokens** (not the instruction/input prefix)
- Maximize $\sum_t \log P(o_t | i_1, \ldots, i_n, o_1, \ldots, o_{t-1})$

For encoder-decoder models:
- Feed instruction + input to the encoder
- Feed ground truth output to the decoder (teacher forcing)
- Loss computed on decoder output

---

## Scaling Results

Key empirical findings (from "Scaling Instruction-Finetuned Language Models", Wei et al.):

1. **More tasks → better generalization** — performance improves consistently as the number of fine-tuning tasks increases
2. **Larger models → better results** — instruction tuning benefits scale more for larger models
3. **Cheap relative to pre-training** — instruction fine-tuning uses ~0.06% of the compute required for pre-training

These findings generalize across encoder-decoder (T5) and decoder-only (GPT) models, and across different pre-training objectives.

---

## Automating Instruction Dataset Creation (Self-Instruct)

Human-written instruction datasets are expensive to scale. **Self-Instruct** (Wang et al., 2022) generates instruction data automatically:

1. Start with 175 human-written seed tasks
2. Sample a few seed tasks; prompt the LLM to generate new instructions
3. Classify each new instruction: classification task or not (using few-shot examples)
4. Generate input-output instances:
   - **Non-classification:** prompt → generate input, then output
   - **Classification (output-first approach):** generate output label first, then generate an input for that label (ensures balanced class distribution)
5. Filter: discard if ROUGE-L similarity to existing tasks > 0.7, or if instructions mention images/graphs (unimodal)
6. Add valid examples to the task pool; repeat

This bootstraps a large diverse instruction dataset with minimal human effort.

---

## Instruction Tuning vs. Prompting

| | Instruction Tuning | Prompting (In-context Learning) |
|--|----|----|
| Weight updates | Yes (fine-tuning) | No (frozen model) |
| Example format | Training data | Prompt examples |
| Cost | Compute for fine-tuning | No training cost |
| Flexibility | Fixed to trained behavior | Flexible at inference time |

Instruction tuning teaches the model **to follow instructions in general**. Prompting (covered in [[Prompting]]) uses the already instruction-tuned model's capabilities at inference time without updating weights.

---

## Related

- [[Pre-training Strategies]] — base models that are then instruction-tuned
- [[Transformer]] — the architecture underpinning all modern LLMs
- [[Alignment]] — RLHF builds on instruction tuning to align with human preferences
- [[Prompting]] — using instruction-tuned models at inference time
