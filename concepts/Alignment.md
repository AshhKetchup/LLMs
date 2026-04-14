---
tags: [llm, nlp, alignment, rlhf, reward-model, ppo, dpo]
source_count: 1
last_updated: 2026-04-14
source: "Lec 25 | Alignment & RLHF — Introduction to LLMs playlist"
---

# Alignment

After pre-training and instruction fine-tuning, LLMs can follow instructions but may still produce responses that are factually wrong, harmful, or inconsistent with human values. **Alignment** shapes the model's responses to match human preferences.

---

## The Problem

Instruction fine-tuning teaches *how* to respond. Alignment teaches *which response is better*.

Example of misalignment:
- Q: "What's the best way to lose weight quickly?"
- Instruction-tuned response: "Stop eating entirely for a few days"
- Aligned response: "Reduce carbs, increase protein/fiber, exercise regularly"

Both can be generated. Without alignment, the model doesn't know which is preferred.

**Goal:** Responses should be **helpful, harmless, and honest** (Anthropic's HHH criteria).

---

## Reinforcement Learning Framework

Alignment maps the language model to a **reinforcement learning** setup:

- **Policy** ($\pi_\theta$) = the LLM being trained. Outputs a probability distribution over vocabulary at each step.
- **State** ($s_t$) = the prompt + tokens generated so far
- **Action** ($a_t$) = the next token generated
- **Reward** = scalar indicating quality of the response

Key difference from standard RL: reward is **sparse** — received only at the end of the full response (end-of-sequence token), not at each step.

---

## RLHF (Reinforcement Learning from Human Feedback)

Three stages:

### Stage 1: Supervised Fine-Tuning (SFT)
Fine-tune a pre-trained LLM on high-quality (instruction, response) pairs using standard next-token prediction. This is instruction fine-tuning (see [[Instruction Tuning]]).

### Stage 2: Train a Reward Model
Instead of having humans rate individual responses numerically (subjective, hard to scale), collect **pairwise preference data**:
- Given prompt $x$, generate two responses $y_1, y_2$
- Humans judge: which is better?
- Train a reward model $r(x, y)$ to predict human preferences

> **Analogy:** The reward model is like a taste-tester who rates recipes. The chef (policy model) learns what flavors the tester prefers, then adjusts cooking accordingly.

The pairwise preference format is less subjective than assigning raw scores. It's also cheaper — you need fewer human annotations to train a good reward model.

**RLAIF (RL from AI Feedback):** Replace human annotators with another LLM to generate the preference labels. Much cheaper and faster; used in models like Claude.

### Stage 3: Policy Optimization (PPO)
Use the reward model to fine-tune the SFT model via **Proximal Policy Optimization**:

$$\max_{\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} \right]$$

- $r(x, y)$ = reward model score for the response
- $\pi_\text{ref}$ = reference policy (SFT model, frozen) — prevents the model from drifting too far from the instruction-tuned version
- $\beta$ = KL penalty coefficient — controls how much the policy can deviate from $\pi_\text{ref}$

> **Analogy:** The KL penalty is like telling the chef "improve your dishes but don't drift so far that you stop being a French restaurant."

The KL penalty is critical: without it, the policy can "game" the reward model by generating unusual text that scores high on the reward model but is nonsensical in practice (reward hacking).

---

## DPO (Direct Preference Optimization)

Simpler alternative to PPO-based RLHF. Instead of training a separate reward model and then doing RL, DPO directly optimizes on preference pairs:

$$\mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E} \left[ \log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right) \right]$$

Where $y_w$ = preferred response ("winner"), $y_l$ = dispreferred response ("loser").

DPO falls under **contrastive learning** — directly increase probability of preferred responses and decrease probability of dispreferred ones, relative to the reference policy.

> **Analogy:** Like telling the chef directly "prefer dish A over dish B" — skipping the taste-tester rating system entirely. Simpler and often just as effective.

**Advantages of DPO over PPO:**
- No separate reward model needed
- More stable training (no RL instability)
- Computationally cheaper

---

## Alignment Taxonomy

| Approach | Method | Online/Offline |
|----------|--------|---------------|
| **RLHF** | PPO with human preferences | Online (iterative) |
| **RLAIF** | PPO with AI-labeled preferences | Online |
| **DPO** | Direct preference optimization | Offline |
| **Constitutional AI** | Rule-based AI feedback (Anthropic) | Both |

---

## Limits of Alignment

1. **Reward hacking** — model finds ways to maximize reward without actually being helpful (addressed by KL penalty)
2. **Distribution shift** — aligned on a distribution of prompts; may fail on unusual inputs
3. **Value subjectivity** — "better" is human-defined; annotators disagree on edge cases
4. **Sycophancy** — models may learn to tell users what they want to hear rather than what is true

---

## Related

- [[Instruction Tuning]] — Stage 1 of RLHF; prerequisite
- [[Pre-training Strategies]] — the base model before alignment
- [[Prompting]] — eliciting aligned behavior at inference time
