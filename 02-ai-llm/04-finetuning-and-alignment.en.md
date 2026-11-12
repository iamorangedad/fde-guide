# 3.4 Fine-Tuning and Alignment: SFT, RLHF, PPO, GRPO, DPO

> 🌐 [中文](04-finetuning-and-alignment.md)

> [🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md) · [Previous: 3.3](03-pretraining-and-adaptation.en.md) · [Next: 3.5](05-neural-nets-and-transformers.en.md)

## Prerequisites

[3.1](01-business-and-model-choice.en.md)–[3.3](03-pretraining-and-adaptation.en.md). This section focuses on **concepts and method selection**; backpropagation and attention are in [3.5](05-neural-nets-and-transformers.en.md), and LoRA parameter counts and memory accounting are developed later in this section and in [3.7](07-training-and-inference-systems.en.md). The in-depth material on "when to fine-tune / LoRA" has been merged into this section and the worked-example scripts in `examples/`.

## Learning Objectives

- Choose among SFT / preference optimization / no training based on "is the knowledge external, is the behavior stable, is there preference data"
- Explain at which step of the pipeline RLHF (reward model + RL), PPO, GRPO, and DPO differ
- Complete the closed-form exercises on prompt masking and LoRA parameter counts; do not copy leaderboard scores from papers as your own results

## Definitions and Selection Table

**Prompting**: does not change the weights.  
**RAG**: does not change the weights; knowledge is attached externally.  
**SFT**: maximizes \(\log p_\theta(y\mid x)\) on paired data (the loss on \(x\) is often masked).  
**Full fine-tuning**: updates all (or nearly all) weights.  
**LoRA**: freezes \(W\) and trains only a low-rank update \(BA\) (Hu et al., 2021).  
**QLoRA**: stores the frozen base model in low-bit precision and trains the adapters in high precision (Dettmers et al., 2023).  
**RLHF**: first (or concurrently) collect preference data → learn a reward model → optimize the policy with RL, usually with a KL constraint (the InstructGPT narrative: Ouyang et al., 2022).  
**PPO**: a clipped policy-gradient algorithm, often used as the RL step in RLHF (Schulman et al., 2017).  
**GRPO**: a PPO variant proposed in DeepSeekMath that estimates the advantage by standardizing rewards within a group of samples for the same question, dropping the value network (Shao et al., 2024).  
**DPO**: rewrites the "reward model + RL" objective as a classification loss on the policy and optimizes directly on preference pairs (Rafailov et al., 2023).

| What you actually need | Consider first | Do not consider first |
|---|---|---|
| Policies that get updated + citations | RAG + ACL | SFT/CPT of the text into the weights |
| JSON that becomes stable once you rephrase the prompt | Prompting + validation / constrained decoding | Jumping straight to RL |
| Format/tool arguments still unstable, with clean pairs | SFT (full or LoRA) | Trying your luck with a bigger prompt |
| Several answers are all correct, but you want safer/shorter/a preferred style | Preference data + DPO or RLHF | Tuning PPO without preference pairs |
| Reasoning problems that can be graded automatically (math, etc.) | Verifiable rewards + GRPO/PPO-style methods | Forcing the same reward onto unverifiable writing-quality tasks |
| The fp16 base model does not fit on the training GPU | QLoRA or a smaller model | Pretending to do full 7B fine-tuning on a CPU |

## Principle A: SFT and Prompt Masking (Concepts + a Small Calculation)

Under teacher forcing, the logits at position \(t\) predict token \(t+1\). The classic implementation aligns `logits[..., :-1]` with `labels[..., 1:]`. The labels of prompt tokens are set to `-100` (the ignore index of PyTorch's `CrossEntropyLoss`), so the average loss reflects only "how good the answer is".

**Worked example (consistent with `examples/worked_examples.py`)**: 30 prompt positions with NLL = 0.1 and 2 answer positions with NLL = 2. Averaged over everything: \((30\times0.1+2\times2)/32=0.21875\); counting only the answer: the average is 2. The former looks "better" but is in fact dominated by the easy-to-copy prompt.

LoRA forward pass: \(W'=W+\frac{\alpha}{r}BA\) (PEFT's default scaling; with `use_rslora` it is \(\alpha/\sqrt{r}\)). Parameter count: \(r(d_{\mathrm{in}}+d_{\mathrm{out}})\). Example: \(d_{\mathrm{in}}=1024,d_{\mathrm{out}}=2048,r=16\) → LoRA 49152, full 2097152.

## Principle B: The RLHF Pipeline (Concepts)

Typical stages (the description follows InstructGPT; refer to the paper for details):

1. **SFT**: tune the model so that it "answers according to instructions".
2. **Reward model (RM)**: fit preferences on answer pairs \((y_w,y_l)\) for the same prompt (commonly Bradley–Terry).
3. **RL optimization of the policy**: maximize expected reward, with a KL penalty to avoid drifting far from the reference policy (often the SFT model).

PPO serves here as one algorithm for updating the policy stably: it scales updates by the advantage function and clips the probability ratio to keep any single update from being too large. When the value network (critic) is the same size as the policy, **memory use and engineering complexity** rise significantly; this is what GRPO aims to alleviate.

## Principle C: GRPO (Concepts)

For the same question \(q\), sample a group of outputs \(\{o_i\}_{i=1}^{G}\) and standardize the rewards \(r_i\) within the group:

$$
A_i=\frac{r_i-\mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\})}
$$

Then update the policy with a PPO-like importance-sampling ratio and clipping, usually with a KL term toward the reference policy. Key point: **the baseline comes from relative quality within the group, not from a separately trained value model**. It suits tasks where the reward can be computed automatically; when rewards are noisy or not comparable, within-group standardization also amplifies the noise.

**Toy numbers**: rewards \([1,0,1,0]\) → mean 0.5, population standard deviation 0.5 → advantages \([1,-1,1,-1]\) (see `grpo_advantages` in the script).

## Principle D: DPO (Concepts)

DPO's key observation: under the KL-constrained RL objective, there is a closed-form relationship between the optimal policy and the reward, so the preference likelihood can be written as a loss **involving only the policy \(\pi_\theta\) and the reference \(\pi_{\mathrm{ref}}\)** (in binary cross-entropy form). Intuition: raise the log-probability of the "winning answer" relative to the reference, lower that of the "losing answer", and use \(\beta\) to control how far it may deviate from the reference.

A margin demonstration for teaching (**not** a full batch loss implementation): if the winner's log-ratio gain relative to the reference exceeds the loser's by \(\Delta\), then \(\beta\Delta\) can be viewed as a demonstration quantity for the implicit reward margin. `dpo_margin_demo` in the script only pins down the arithmetic and does not claim to reproduce the paper's experiments.

**DPO vs. PPO-RLHF (engineering view)**:

| | DPO | PPO-RLHF |
|---|---|---|
| Separate RM needed? | Implicit in the policy | Usually yes |
| Heavy online sampling during training? | Typically offline preference pairs | Needs sampling and an RL loop |
| Implementation complexity | Relatively low | High |
| Preference data | Pairs required (or constructible pairs) | Also needs preferences to train the RM |
| Verifiable automatic rewards | Usable, but not the only choice | Often combined with an RM or rule-based rewards |

There is no silver bullet: with poor data quality, both will learn "the annotators' biases" or "reward loopholes".

## Case Study (Fictional)

After SFT, a customer-service model has a stable format but sometimes over-promises refunds. The team did not reach for PPO; instead: a rule engine intercepts refund verbs + they collected preference pairs of "appropriate refusal vs. reckless promise" for DPO, and they watch the "violating-promise rate" on the evaluation set. A math problem-solving assistant, on the other hand, uses the correctness of a parseable final answer as the reward and pilots GRPO-style group sampling; the copy-editing task does not use the same automatic reward.

## Exercises

1. Explain: why does "SFT the employee handbook into the model" solve neither the monthly updates nor per-department ACLs?
2. With \(d=4096,r=8\), what is the LoRA parameter count for one \(d\times d\) matrix? If each of 32 layers has 4 such projections, what is the total number of trainable parameters?
3. When would you prefer DPO over PPO? When the reverse?
4. What happens to GRPO's advantage standardization when "all samples in the group have the same reward"? How do you guard against it in engineering?

### Reference Answers

1. Monthly updates require updating the corpus version; ACLs are a retrieval/authorization problem, and weights cannot switch the visible sections according to the user's identity.
2. \(8\times(4096+4096)=65536\); \(32\times4\times65536=8388608\).
3. You already have offline paired preferences and want lower engineering complexity → DPO; you need online exploration, the reward is a complex environment signal, or you already have mature RL infrastructure → PPO-style methods. Large-scale sampling that can be graded automatically → consider GRPO.
4. The standard deviation is 0, so the advantage is undefined or all 0. Skip that group, add ε, or require that rewards within the group be distinguishable.

## Summary

Fine-tuning addresses "stable behavior", and alignment addresses "preferences and constraints"; neither replaces external knowledge and programmatic gates. RLHF / PPO / GRPO / DPO differ in where the reward comes from, whether there is an explicit RM, and whether sampling is online; get the data and evaluation first, then pick the acronym.

## Key Terms

SFT, LoRA, QLoRA, RLHF, reward model, PPO, GRPO, DPO, KL constraint, prompt masking

## Further Reading (Original Papers and Documentation)

- Ouyang et al., InstructGPT — https://arxiv.org/abs/2203.02155
- Schulman et al., PPO — https://arxiv.org/abs/1707.06347
- Rafailov et al., DPO — https://arxiv.org/abs/2305.18290
- Shao et al., DeepSeekMath / GRPO — https://arxiv.org/abs/2402.03300
- Hu et al., LoRA — https://arxiv.org/abs/2106.09685
- Dettmers et al., QLoRA — https://arxiv.org/abs/2305.14314
- PEFT LoRA — https://huggingface.co/docs/peft/en/package_reference/lora
- TRL GRPO Trainer — https://huggingface.co/docs/trl/grpo_trainer

---

[🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md)
