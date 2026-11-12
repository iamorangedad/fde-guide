# 3.2 Application Systems: RAG, Agents, and Evaluation

> 🌐 [中文](02-applications-rag-agent-eval.md)

> [🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md) · [Previous: 3.1](01-business-and-model-choice.en.md) · [Next: 3.3](03-pretraining-and-adaptation.en.md)

## Prerequisites

The constraint tables and method ladder from [3.1](01-business-and-model-choice.en.md).

## Learning Objectives

- Draw a RAG pipeline with permission filtering, and tell "retrieval failures" apart from "generation failures"
- Make a defensible choice between an agent and a fixed workflow, and set cost and risk caps for tool calls
- Build layered evaluation: unit assertions, human sampling, and, where necessary, calibrated model judges

## Definitions

**Context window**: the maximum number of tokens a single request can send to the model. It limits "how many retrieved chunks fit in at once"; it does not remove the need for an external knowledge base.

**RAG (Retrieval-Augmented Generation)**: first retrieve relevant chunks from an external corpus, then generate the answer conditioned on them, to improve factuality and traceability.

**Embedding**: mapping text into a vector space so that semantically similar texts are closer together; often combined with keyword retrieval (such as BM25) into **hybrid retrieval**.

**Tool calling**: the model outputs a structured call intent, and the runtime executes it and returns the result; the model itself does not hold direct write access to the database.

**Agent**: the model dynamically decides the next step and tool; **workflow**: the steps are fixed in advance by code. In engineering, prefer the simplest approach that still gets the task done.

**Evaluation (eval)**: a repeatable measurement of quality defined for the task, used to decide whether to change the prompt, the retrieval, or the training.

## Principles: The RAG Pipeline and Diagnosis

```text
Documents → parse/clean → chunk → embed → index
User question → (query rewriting) → retrieve top-k → (rerank) → assemble context → generate + cite
                              ↑
                     Metadata/ACL filtering must happen here
```

| Decision | Options | Trade-offs |
|---|---|---|
| Chunking | Fixed length / by heading and paragraph | Structural chunking often works better for clauses and manuals |
| Retrieval | Vector / lexical / hybrid | IDs, model numbers, and proper nouns depend more on lexical signals |
| Reranking | Cross-encoders, etc. | Precision ↑, latency ↑ |
| Permissions | Filter before retrieval vs. check after generation | **Must filter at retrieval time**; filtering after generation cannot prevent the context from leaking to the model |

**Order of diagnosis**: first look at failing samples by hand. Is the gold passage simply not in the top-k (a recall problem), or is the gold passage there but the answer made up (a generation/prompt/faithfulness problem)? The fixes for these two kinds of problems are completely different; lumping them together only leads to "switch to a bigger model".

### Agents and Workflows

Common patterns: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer. Four hard constraints for enterprise deployment:

1. Strict tool schemas, validated again on the server side;
2. Caps on the maximum number of steps and on cost;
3. Writing to databases, making payments, sending things externally, and the like require human confirmation or an independent policy engine;
4. End-to-end traces (prompt version, retrieval hits, tool inputs and outputs).

Structured extraction example (for teaching; depends on your own runtime environment):

```python
from pydantic import BaseModel, ValidationError

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str

def extract(text, llm, max_tries=3):
    for _ in range(max_tries):
        raw = llm(f"Extract JSON from the invoice with fields vendor,total,currency:\n{text}")
        try:
            return Invoice.model_validate_json(raw)
        except ValidationError as e:
            text = text + f"\nPrevious output was invalid: {e}"
    raise ValueError("Extraction failed; hand off to a human")
```

### Three Layers of Evaluation

1. **Unit-level assertions**: JSON parses, required fields are present, forbidden phrases are absent, cited ids exist; all decided by code.
2. **Model judges (LLM-as-judge)**: must be aligned with human sampling and recalibrated regularly; they cannot be treated as unbiased ground truth.
3. **Human and online**: adoption rate, edit rate, task completion time, share escalated to humans.

Common decomposed metrics for RAG: retrieval Recall@k, answer faithfulness (is it supported by the citations), and answer relevance. First do **error analysis** on dozens to a hundred or more real traces, then build sets targeting the high-frequency buckets, rather than first buying an "automated evaluation platform".

## Productionization Concerns

- **Latency and cost**: cache repeated queries, route simple intents to small models, stream output to improve perceived speed.
- **Security**: prompt injection (including indirect injection in retrieved chunks and tool results), privilege escalation, output filtering.
- **Versioning**: prompts, models, indexes, and chunking rules must all be traceable; rollbacks must roll back the whole chain together.

## Boundaries with Later Sections

The tools of this section are retrieval, permissions, citations, and evaluation. When the policy text gets updated or must be isolated by identity, stay with RAG. When what prompting cannot stabilize is the **format or the shape of tool arguments**, and clean paired data already exists, move on to [3.4](04-finetuning-and-alignment.en.md). For pretraining-style "pouring in corpora", see [3.3](03-pretraining-and-adaptation.en.md). For the KV cache and batch scheduling, see [3.7](07-training-and-inference-systems.en.md). CUDA is in [3.8](08-cuda-and-gpu-kernels.en.md), and it is not the next step for every project.

## Case Study (Fictional)

Legal contract Q&A is inaccurate. Error analysis: pure vector retrieval often mismatches clause numbers such as "Clause 7.2"; multiple versions of a contract are not filtered by version metadata. After switching to hybrid retrieval + version filtering, the citation hit rate rises (the exact percentage must be re-measured on your own set; none is made up here). **Lesson**: look at the data first, then change the architecture.

## Exercises

1. Design RAG for an internal knowledge base, explain how permissions run through retrieval, and explain why "checking for privilege violations after generation" is not enough.
2. Collect (or synthesize) 30 Q&A pairs and build an error classification table: retrieval / chunking / permissions / prompt / hallucination / tools.
3. Invoice field extraction vs. open-ended competitor research: which one uses a workflow, and which one is the case where you would even consider an agent? Why?

### Reference Answers

1. The index or retrieval query must carry the doc ids / department tags the user can access; filtering happens before taking the top-k. Checking after generation cannot stop sensitive chunks from entering the context window.
2. The classification table should map directly onto next week's engineering tasks (change chunking, add BM25, fix ACLs, ...).
3. The steps for invoice fields are predictable → workflow + validation; open-ended research has many branching paths → only then consider a constrained agent, still with step limits and a tool allowlist.

## Summary

An LLM application system = data + retrieval + model + tools + evaluation. Evaluation is the steering wheel; permissions are a retrieval problem, not a prompt problem.

## Key Terms

RAG, hybrid retrieval, reranking, tool calling, agent/workflow, LLM-as-judge, prompt injection, faithfulness

## Further Reading

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- Hamel Husain on evals — https://hamel.dev/blog/posts/evals/
- DeepLearning.AI short courses — https://www.deeplearning.ai/short-courses/

---

[🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md)
