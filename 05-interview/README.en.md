# Chapter 6 Interview Preparation

> 🌐 [中文](README.md)

> [🏠 Back to home](../README.en.md)

## Learning Objectives
Understand the common process, and master problem-decomposition interviews, coding, project deep dives, and behavioral interviews.

## 6.1 Common Stages (Details Vary by Company)
Recruiter call → coding (practical, often involving data processing/APIs) → **problem decomposition / case** (open-ended business problem) → project deep dive → behavioral / customer scenarios → team matching. According to public reports, Palantir's interviews include a decomposition round.

## 6.2 Problem Decomposition Interviews
Framework: **clarify the goal → identify users and metrics → break down sub-problems → data and systems → MVP → risks → iteration**.
Sample question: "A hospital wants to reduce emergency-room waiting times. What would you do?"
Reference points: clarify the definition of "waiting" (arrival → triage → seen by a doctor); data (registration system, electronic medical record timestamps); find the bottleneck (triage staffing, beds); MVP (real-time queue dashboard + arrival volume forecasting); risks (data privacy, changes to clinical workflows); metrics (median/P90 waiting time).
**Scoring (1–5)**: structure, quality of clarifying questions, data awareness, practicality, communication.

## 6.3 Coding
Focus: string/JSON processing, hash tables, sorting, intervals, simple graphs; plus practical problems such as "read paginated API data and aggregate it". When practicing, talk through your reasoning as you write, and write tests.

## 6.4 Behavioral and Customer Scenarios
Use STAR (Situation-Task-Action-Result). Prepare 6 stories: handling ambiguous requirements, customer conflict, a production incident, persuading others, failure and retrospective, delivering under pressure.
Mock scenario: "The customer's data pipeline went down an hour before the demo." Scoring: stop the bleeding and communicate first, explain the impact transparently, have a fallback plan, run a retrospective afterwards.

## 6.5 System Questions Specialized Roles May Ask
Prepare these only when the job description mentions training, CUDA, or inference serving. What you should be able to explain clearly are the trade-offs: RAG or LoRA, why Adam's state is often still 16 bytes per parameter, the definitions of TTFT and p95, and changing only one variable per experiment. Do not memorize speedups that this tutorial has not measured. The material is in [Sections 3.3–3.9 of Chapter 3](../02-ai-llm/README.en.md).

## 6.6 Two-Week Sprint Checklist
- [ ] 10 decomposition questions, 30 minutes each, recorded and reviewed
- [ ] 20 practical coding problems
- [ ] A 5-minute walkthrough + deep-dive Q&A prepared for each portfolio project
- [ ] 6 STAR stories
- [ ] 2 full mock interviews

---

[🏠 Back to home](../README.en.md)
