# 3.1 Business Requirements, Constraints, and Model Choice

> 🌐 [中文](01-business-and-model-choice.md)

> [🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md) · [Next: 3.2](02-applications-rag-agent-eval.en.md)

## Prerequisites

None. It is a good idea to read the whole chapter's table of contents first, to see the order "business → learning methods → model principles → frameworks → hardware", and then decide how deep you need to go.

## Learning Objectives

- Break a customer's spoken "we want an AI" into a verifiable task, data boundaries, and the cost of failure
- Make a first round of choices using four kinds of constraints: latency, cost, privacy, and controllability
- Give a defensible default order among "prompting / RAG / fine-tuning / alignment / in-house pretraining", instead of defaulting to training

## Definitions

**A language-model project in the context of a Forward Deployed Engineer (FDE)** is usually not about training a base model from scratch, but about wiring model capabilities into existing systems in the customer's environment: identity and permissions, knowledge sources, tools, auditing, and evaluation. The success criterion is almost always a business metric (ticket resolution rate, accuracy of locating contract clauses, rate of human edits), not a public leaderboard score.

**Model choice** means choosing, under given constraints: which vendor's API to call, which open-weight model to deploy, whether a private deployment is needed, and whether the weights need to change. Model choice happens before anyone writes a training script.

**Cost of failure** means the business consequence of one wrong answer. A badly worded customer-service reply may be acceptable; a wrong medical dosage, payment instruction, or permission decision is usually not. The latter must be guarded by programmatic gates and human confirmation, not by hoping that "the model learns to be careful".

## Principles: Write the Constraints First, Then the Architecture

Write the requirements down as four tables; if any item is blank, do not move on to "let's fine-tune":

| Dimension | Questions to answer | Typical form of the answer |
|---|---|---|
| Task | What is the input, which fields must the output contain, what counts as success | "Given the user's ticket and the knowledge base, return actionable steps + citations" |
| Data | Does the knowledge change, who may see it, may it leave the country | "Policies updated monthly; per-department ACLs; must not leave the intranet" |
| System | Existing identity provider, ticketing system, audit requirements | "OIDC + ServiceNow; end-to-end audit trail" |
| Operations | p95 latency, concurrency, budget, availability | "First token < 2s; peak 20 QPS; monthly budget cap" |

### The Default Method Ladder (Business View)

From cheap and reversible to expensive and hard to roll back:

1. **Prompts and validators**: format, enumerated fields, JSON schema, refusal policy. Failures can be fixed immediately by changing the prompt.
2. **RAG / tool calling**: the facts live in external systems; permissions are enforced at the retrieval or API layer. Details in [3.2](02-applications-rag-agent-eval.en.md).
3. **Supervised fine-tuning (SFT)**: the behavior (format, tone, shape of tool arguments) is still unstable under prompting, and there are clean input-output pairs. Concepts in [3.4](04-finetuning-and-alignment.en.md).
4. **Preference alignment (DPO / RLHF, etc.)**: among several "equally correct" answers, you need to prefer a certain style or safety policy, and there is paired or grouped preference data.
5. **Continued pretraining / in-house pretraining**: the corpus distribution is very different from that of general base models, and there is large-scale domain text and compute. This is the rarest case in FDE projects; see [3.3](03-pretraining-and-adaptation.en.md).

**Writing policies that change into the weights** is usually the wrong default: when the policy is revised next week, what you need is an index and permission update, not another expensive training run. Separating "memorizing facts" from "learning behavior" is a principle that recurs throughout this chapter.

### APIs, Open Weights, and Private Deployment

| Option | Good for | Main cost |
|---|---|---|
| Hosted API | Fast validation, moderate traffic, data that may be processed under contract | Vendor dependency, egress latency, compliance review |
| Self-hosted open weights | Data must not leave the domain, need to change the serving setup or adapters | You own operations, GPU capacity, and the security baseline |
| Fine-tuned adapters + shared base | Multi-tenant/multi-task, uniform base upgrades | Merged vs. unmerged deployment setups must be designed carefully |
| Pretraining from scratch | Almost only platform-level organizations | Data engineering and clusters, beyond the scope of most FDE engagements |

When choosing, ask three "can we" questions: **Can we give users the original source citations to audit? Can we filter knowledge by identity? Can we refuse high-risk actions outside the model?** Only after all three can be answered in the negative should you talk about bigger models or more training.

## Worked Examples and Cases (Fictional)

**Scenario A (composite)**: a manufacturing company wants "Q&A over equipment manuals". The PDF manuals are updated monthly; line workers and engineers see different sections; answers must include section numbers.

- Correct default: parsing + hybrid retrieval filtered by role metadata + generation with citations; evaluation checks whether "the citation supports the answer".
- Wrong default: SFT on the whole manual to "memorize it". As soon as the manual changes, the weights are out of date, and there is no way to isolate content by role.

**Scenario B (composite)**: a customer-service system must reliably output "ticket-field JSON"; with prompting plus validation, 15% of outputs still have missing fields (this rate is an assumed input for the exercise, not an industry statistic).

- Options: a small amount of "conversation → valid JSON" SFT or strongly constrained decoding; factual fields still come from CRM tool calls, so the model does not make up order numbers.

**Capacity intuition (closed-form, assumed inputs)**: if the SLA requires a p95 end-to-end latency of 3s, of which retrieval takes 200ms and business systems 300ms, the budget for model generation is about 2.5s. If the average output is 400 tokens and decoding runs at about 40 tokens/s (assumed), decoding one request takes about 10s, which does not fit. The conclusion is not "switch to a bigger model" but to cut the output, use streaming interaction, or switch to a faster small model / inference stack. For details on the KV cache and batch scheduling, see [3.7](07-training-and-inference-systems.en.md).

## Exercises

1. The customer says "we want ChatGPT, but the data must not leave the company". Write at least four questions you must follow up with, and explain how each one changes the model choice.
2. For each of the following requirements, mark the first choice among "prompting / RAG / SFT / alignment / pretraining" (several are allowed but must be ranked): (a) answers must cite the currently valid safety procedures; (b) refuse to give medical dosage advice; (c) always close with the company's honorific template; (d) understand rare symbols in scanned internal circuit diagrams (assume general models have almost never seen them).
3. Why is "start with the biggest model, then optimize" often a bad strategy in FDE delivery?

### Reference Answers

1. Examples: data classification and cross-border rules; identity provider and ACLs; whether it must run offline; human-in-the-loop scenarios; latency and concurrency; the current form of the knowledge base; audit fields. A change in any answer can move you from an API to self-hosting, or from "trainable" to "retrieval only".
2. (a) RAG (facts change + citations); (b) system policy / classifier / refusal rules, with preference-data alignment if necessary, but hard refusals belong in the program layer; (c) prompting or a small amount of SFT; (d) may need continued pretraining on domain data or a dedicated visual document pipeline; first confirm whether the problem really is "unseen symbols" or poor parsing quality.
3. A bigger model amplifies unit cost and latency and masks defects in retrieval/permissions/evaluation; after launch it is hard to attribute problems. First use an observable pipeline to prove that the bottleneck is model capability, then scale up the model.

## Summary

The first chapter for an FDE is not CUDA; it is writing the business down as constraints and measurable tasks. The method ladder climbs up from prompting and RAG; training is a prescription for "stable behavior" or "distribution shift", not a status symbol.

## Key Terms

Cost of failure, method ladder, private deployment, adapter, SLA, ACL, reversibility

## Further Reading

- Anthropic, *Building effective agents* — https://www.anthropic.com/engineering/building-effective-agents (workflows before complex agents)
- Hamel Husain on evals — https://hamel.dev/blog/posts/evals/
- Later in this chapter: application systems [3.2](02-applications-rag-agent-eval.en.md); learning methods [3.3](03-pretraining-and-adaptation.en.md), [3.4](04-finetuning-and-alignment.en.md)

---

[🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md)
