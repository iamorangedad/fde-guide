# Chapter 5 Portfolio Projects

> 🌐 [中文](README.md)

> [🏠 Back to home](../README.en.md)

Every project should simulate a "customer delivery": write a scope statement, deliver code, write a runbook, record a 5-minute demo video, and write a retrospective.

| # | Project | Skills | Duration |
|---|---|---|---|
| P1 | Multi-source data integration and dashboard | Integration, SQL, modeling | 2 weeks |
| P2 | Enterprise knowledge base RAG (with permissions) | RAG, evaluation, security | 3 weeks |
| P3 | Document extraction workflow | LLM structured output, human review | 2 weeks |
| P4 | Tool-calling agent | Agents, tracing, guardrails | 2 weeks |
| P5 | Private deployment package | Docker, IaC, observability | 1–2 weeks |

## P1 Multi-Source Data Integration and Dashboard
- **Goal**: integrate 2–3 sources from public datasets (such as city open data) and build a unified model and dashboard
- **Spec**: incremental pulls, idempotent upserts, data quality tests, freshness monitoring
- **Milestones**: W1 raw layer + staging; W2 marts + dashboard + alerts
- **Acceptance**: re-runs create no duplicates; all quality tests pass; the README lets someone else get it running within 30 minutes
- **Extensions**: CDC, SCD2

## P2 Enterprise Knowledge Base RAG
- **Goal**: build a Q&A system over a set of public documents, simulating permissions for three roles
- **Spec**: hybrid retrieval, reranking, cited sources, permission filtering at retrieval time, an evaluation set of ≥50 items
- **Acceptance**: report Recall@5 and faithfulness; 0 leaks in privilege-violation tests; an error analysis document
- **Extensions**: query rewriting, multi-turn conversation

## P3 Document Extraction Workflow
- **Goal**: extract structured fields from PDFs such as invoices/résumés
- **Spec**: Pydantic validation, low-confidence items routed to a human review interface, field-level accuracy evaluation
- **Acceptance**: a labeled set and an accuracy report; failures are traceable
- **Extensions**: active learning, collecting human corrections

## P4 Tool-Calling Agent
- **Goal**: an operations assistant that can query databases, generate reports, and create tickets (simulated)
- **Spec**: tool argument validation, a maximum number of steps, human confirmation for write operations, end-to-end tracing
- **Acceptance**: 20 scenario test cases; prompt-injection test cases
- **Extensions**: multi-agent orchestration

## P5 Private Deployment Package
- **Goal**: package P2 so that it can be deployed in one step into a "customer environment"
- **Spec**: Docker Compose or Helm, Terraform, health checks, logs/metrics, a runbook, offline installation instructions
- **Acceptance**: deployment completed on a clean virtual machine within 1 hour by following the documentation

## P6 (Elective) Capacity and Release Plan for a Policy Assistant
- **Corresponds to**: [Section 3.7](../02-ai-llm/07-training-and-inference-systems.en.md). Roles that do not involve model training can skip P6
- **Goal**: for a fictional internal Q&A requirement, write the scope, resource budget, baseline, and an experiment with only one independent variable
- **Spec**: knowledge goes through RAG; if full fine-tuning is ruled out, give the byte ledger from Section 3.7; latency gates use the TTFT/p95 definitions; items not run on a GPU go on an unmeasured list
- **Acceptance**: every number can be traced back to a function in [`worked_examples.py`](../02-ai-llm/examples/worked_examples.py) or to your own measurement; no unmeasured speedups

## Scoring Rubric (1–5 each)
Clarity of problem definition / engineering quality / rigor of evaluation / documentation and deliverability / demo presentation

---

[🏠 Back to home](../README.en.md)
