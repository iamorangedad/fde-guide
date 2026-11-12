# 3.3 Pretraining and Domain Adaptation (Concepts First)

> 🌐 [中文](03-pretraining-and-adaptation.md)

> [🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md) · [Previous: 3.2](02-applications-rag-agent-eval.en.md) · [Next: 3.4](04-finetuning-and-alignment.en.md)

## Prerequisites

[3.1](01-business-and-model-choice.en.md), [3.2](02-applications-rag-agent-eval.en.md). This section **first explains the business meaning and the intuition behind the objective functions**; the mathematics of neural networks and self-attention is in [3.5](05-neural-nets-and-transformers.en.md), and the engineering implementation in [3.6](06-pytorch-engineering.en.md)–[3.7](07-training-and-inference-systems.en.md).

## Learning Objectives

- Distinguish the "distribution" that pretraining, continued pretraining (CPT), instruction tuning, and alignment each optimize
- Judge whether a customer corpus suits RAG, CPT, or should not go into training at all
- Write down the conceptual form of the causal language modeling loss, and know where the implementation details are covered

## Definitions

**Pretraining**: learning general next-token prediction (or another self-supervised objective) on large-scale unlabeled (or weakly labeled) text to obtain a base model. Large-scale pretraining as publicly reported is platform-level engineering; FDE engagements very rarely start from scratch.

**Continued pretraining / domain-adaptive pretraining (CPT)**: continuing the same kind of self-supervised objective on domain corpora on top of a base model, so that the token distribution moves closer to the domain (terminology, writing style, code dialects). It does **not** directly optimize "answering according to user instructions".

**Instruction tuning / supervised fine-tuning (SFT)**: using "instruction → desired answer" pairs to optimize the conditional distribution \(p(\text{answer}\mid\text{instruction})\). See [3.4](04-finetuning-and-alignment.en.md).

**Alignment**: after SFT, using preference or reward signals to further shape style, safety, and reasoning behavior (RLHF, DPO, GRPO, etc.); also covered in 3.4.

Rule of thumb: **pretraining learns "the statistics of language and the world"; SFT learns "how to do things in the required format"; alignment learns "which of several acceptable answers to prefer".** Knowledge whose facts change should still be kept external first (RAG).

## Principles: Causal Language Modeling (Concepts)

A decoder model writes the joint probability of a sequence \(x_1,\ldots,x_T\) as:

$$
p_\theta(x)=\prod_{t=1}^{T} p_\theta(x_t\mid x_{<t})
$$

(In implementations the product often starts at \(t=2\), with no prediction for the first token; for the exact conventions on shifting and `ignore_index`, see [3.4](04-finetuning-and-alignment.en.md) and [3.6](06-pytorch-engineering.en.md).)

Training minimizes the average negative log-likelihood (NLL):

$$
\mathcal{L}_{\mathrm{CLM}}(\theta)=-\mathbb{E}_{x\sim\mathcal{D}}\sum_{t}\log p_\theta(x_t\mid x_{<t})
$$

**CPT and SFT can share this form of loss**; the difference lies in the data and the masking:

| | Data | Usual masking |
|---|---|---|
| Pretraining / CPT | Continuous documents, code, cleaned web pages | Almost all tokens contribute to the loss |
| SFT | Conversation or instruction templates | The prompt is often masked; the loss is computed only on the answer |

If you use policy text for CPT, the model becomes more likely to "continue text that reads like policy", but it is **not guaranteed** to refuse correctly according to an employee's role or to cite section numbers; the latter requires RAG + evaluation, and SFT on the format if necessary.

### When to Consider CPT (Business Checklist)

Evaluate carefully only when the answers are close to "yes" on all of these at once:

1. There is a large amount of domain text, clearly different from the general web distribution (internal DSLs, regulatory drafting conventions, dense proprietary abbreviations);
2. RAG is already done well, yet the model still systematically "fails to understand" domain phrasing (not a failure to retrieve);
3. There are sustainable processes for data cleaning, decontamination, licensing, and deduplication;
4. There is a reproducible domain evaluation (not just watching the training loss);
5. The compute and person-hour budget allows it, and you accept the cost of redoing CPT whenever the base model is upgraded.

**Common cases where you should not do CPT**: manuals updated monthly, strong ACLs, answers that must be traceable with a click: these are RAG; only the "tone/fields" are unstable: that is SFT; you only need "more polite / more refusals": that is alignment or rules.

### Data Quality (Conceptual Level)

Pretraining corpus engineering usually includes: deduplication, language identification, quality filtering, stripping personal and confidential information, and license compliance. Work such as DeepSeekMath emphasizes that **a large part of domain capability comes from the data selection for continued pretraining**, not just from the final RL formula (see Further Reading). This draft does not repeat unverified benchmark scores; if you cite numbers from a paper, open the original and check its setup.

## Worked Example (Reproducible, No Model Needed)

With a uniformly random vocabulary of size \(V\), the expected NLL at each position is about \(\log V\). If \(V=50257\) (the GPT-2 vocabulary size, a common value on public model cards), then \(\log 50257\approx 10.825\) nats. A training loss that drops to "about this number" only shows "slightly better than guessing"; it does **not** mean the task has been learned. Script: `clm_uniform_nll` in `examples/worked_examples.py`.

## Case Study (Fictional)

A chip company wants the model to "understand" its internal netlist text. A general model splits signal names into English fragments. The team first built a dedicated tokenizer and a syntax-highlighting parser feeding into RAG, and the failure rate dropped; only when that was still not enough did they run small-scale CPT on anonymized netlists, evaluated by "syntactic validity rate + human spot checks". They did **not** CPT the process parameter tables into the model, because the parameters change weekly; parameters go through a database tool.

## Exercises

1. In your own words, distinguish: CPT loss going down, SFT loss going down, and online adoption rate going up. What does each one prove, and what can it not prove?
2. A customer has 20,000 pages of PDF policies + a price list that changes daily. Which go into the index, and which should never be the staple diet of CPT? Why?
3. Why is "we'll pretrain on all of the company's emails" dangerous in terms of both compliance and results? List at least three points.

### Reference Answers

1. CPT loss ↓: domain continuations look more like the training corpus, which does not prove that Q&A is correct; SFT loss ↓: the model fits the labeled answers more closely and may overfit the wording of the reference answers; adoption rate ↑: closer to business value, but error types still need to be broken down.
2. The bulk of the policies and the price list go into an index with permission and freshness metadata; the price list in particular needs a tool/DB. If CPT is done at all, only consider long-term stable conventions/terminology text, strictly anonymized.
3. Privacy and labor compliance; noise and personal attacks in emails get learned into the distribution; freshness and permissions cannot be put into weights; hard to evaluate and roll back.

## Summary

Pretraining and CPT change the "language distribution", not "auditable enterprise facts". Conceptually, lock down the objective and data boundaries first; read on for the formula implementations and the compute accounting.

## Key Terms

Pretraining, continued pretraining, causal language modeling, negative log-likelihood, distribution shift, data licensing

## Further Reading

- Radford et al., GPT-2 report (the causal LM pretraining paradigm) — https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- DeepSeekMath (including the CPT + subsequent RL narrative) — https://arxiv.org/abs/2402.03300
- Next section on alignment and fine-tuning: [3.4](04-finetuning-and-alignment.en.md); mathematical foundations: [3.5](05-neural-nets-and-transformers.en.md)

---

[🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md)
