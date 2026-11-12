# Chapter 3 Language Model Applications and Systems

> 🌐 [中文](README.md)

> This chapter is ordered from shallow to deep, "business → principles → frameworks and hardware": first model choice, pretraining, fine-tuning, and reinforcement learning, which are the most relevant to the business; then neural networks and the Transformer; and finally PyTorch, system performance, CUDA, and the storage stack.

> [🏠 Back to home](../README.en.md)

## Reading Order (Shallow → Deep)

| Section | File | What it covers | Prerequisites |
|---|---|---|---|
| 3.1 | [Business Requirements, Constraints, and Model Choice](01-business-and-model-choice.en.md) | Problem definition, SLAs, compliance, API/open-source/self-hosted, when not to train | None |
| 3.2 | [Application Systems: RAG, Agents, and Evaluation](02-applications-rag-agent-eval.en.md) | Retrieval, workflows, evaluation, productionization | 3.1 |
| 3.3 | [Pretraining and Domain Adaptation (Concepts)](03-pretraining-and-adaptation.en.md) | Pretraining, continued pretraining, data and objectives; technical details forward-reference 3.5–3.7 | 3.1–3.2 |
| 3.4 | [Fine-Tuning and Alignment: SFT, RLHF, PPO, GRPO, DPO](04-finetuning-and-alignment.en.md) | Choosing a method, intuition for the losses, preference learning; detailed formulas forward-reference 3.5 | 3.3 |
| 3.5 | [Neural Networks and the Transformer](05-neural-nets-and-transformers.en.md) | Backpropagation, attention, positional encoding, decoder architecture | Basics of linear algebra and probability |
| 3.6 | [PyTorch Engineering Practice](06-pytorch-engineering.en.md) | **The framework layer** (not hardware): tensors, autograd, modules, training loops | 3.5 |
| 3.7 | [Training and Inference System Performance](07-training-and-inference-systems.en.md) | Memory accounting, mixed precision, KV cache, batch scheduling, capacity | 3.4–3.6 |
| 3.8 | [GPUs and CUDA (Advanced Elective)](08-cuda-and-gpu-kernels.en.md) | SIMT, coalesced memory access, reductions, tiled matrix multiplication, roofline | 3.6–3.7; **not GPU-verified in this environment** |
| 3.9 | [Linux I/O and the Storage Stack (System-Layer Supplement)](09-linux-io-and-storage-stack.en.md) | Page cache, fsync, Direct I/O, block layer, dm, overlayfs, diagnostic tools | 3.6–3.7 |

## How to Read the Numbers and Evidence

- **Identities and closed-form calculations** (parameter counts, byte counts, shifted losses) give exact results and state their assumptions.
- Peak compute, bandwidth, and latency in **hypothetical scenarios** are exercise inputs, not the spec sheet of any particular GPU, and not unmeasured speedups.
- **No unmeasured speedup factors are recorded.** The CUDA examples in this draft were **not compiled and not run on a GPU**.
- All customer cases are **fictional composite scenarios** used to illustrate decisions, not real customer data.
- The main citations point to original papers and official documentation; this chapter is the only version of Chapter 3 in the repository.

## Worked Examples You Can Run Locally

No GPU and no model downloads required:

```bash
cd /workspace/fde-chapter3-draft
python3 examples/worked_examples.py
python3 examples/test_worked_examples.py
```

---

[🏠 Back to home](../README.en.md)
