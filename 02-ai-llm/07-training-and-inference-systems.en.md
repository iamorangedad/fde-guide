# 3.7 Training and Inference System Performance

> 🌐 [中文](07-training-and-inference-systems.md)

> [🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md) · [Previous: 3.6](06-pytorch-engineering.en.md) · [Next: 3.8](08-cuda-and-gpu-kernels.en.md)

## Prerequisites

[3.4](04-finetuning-and-alignment.en.md)–[3.6](06-pytorch-engineering.en.md). The hardware treatment of arithmetic intensity and the roofline is in [3.8](08-cuda-and-gpu-kernels.en.md); this section first gives a **system-level ledger** for model choice and troubleshooting.

## Learning Objectives

- Account separately for: Adam's static state, frozen base weights, activations, and the KV cache
- Distinguish prefill from decode, and assemble end-to-end latency from TTFT and TPOT (kept separate from throughput)
- Reason one variable at a time among "bigger batch / checkpointing / quantization / a different parallelism", without making up speedups

## Definitions

**Static training state**: parameters, gradients, optimizer moments, and so on that stay resident across steps.  
**Activations**: intermediate tensors kept from the forward pass for the backward pass; they vary with batch size and sequence length.  
**Prefill**: processes the prompt tokens in parallel, writes the initial KV, and produces the first output token.  
**Decode**: generates step by step; reads the existing KV and appends to it.  
**TTFT / TPOT**: time to first token / time per subsequent output token. Convention:

$$
T_{\mathrm{e2e}}=\mathrm{TTFT}+(N-1)\cdot\mathrm{TPOT}\quad(N\ge 1)
$$

The reciprocal of single-request latency is not system throughput; throughput is the total number of output tokens across all requests per unit of wall-clock time.

## Principles: The \(16P\) Ledger of Training Memory

For \(P\) **trainable** parameters, the common static total for full-precision Adam is \(16P\) bytes: parameters 4 + gradients 4 + first moment 4 + second moment 4. Mixed-precision Adam (the static ledger commonly used in the ZeRO paper) often **still totals about \(16P\)**: \(2P\) each for low-precision parameters/gradients, plus \(4P\) for the fp32 master weights and \(4P\) each for the fp32 \(m,v\). "Turning on fp16" does not automatically shrink the optimizer state to \(2P\).

**Worked example (assume \(P=7\times10^9\))**: \(16P=1.12\times10^{11}\) bytes ≈ 104.31 GiB. This is multiplication, not a profiler. LoRA replaces \(P\) in the formula with \(P_{\mathrm{train}}\); the bytes of the frozen base must be **listed separately** and added.

A teaching estimate of activations (implementation-dependent; the assumptions must be stated):

$$
L\cdot c\cdot B\cdot S\cdot H\cdot b + \mathbf{1}_{\mathrm{materialize}}\cdot L\cdot B\cdot n_{\mathrm{head}}\cdot S^{2}\cdot b
$$

When a FlashAttention-style implementation does not materialize the \(S\times S\) scores, the second term should be treated as 0; writing the second term while also claiming FlashAttention is double counting. Gradient checkpointing trades recomputation for activation memory; it saves peak memory, not FLOPs.

Other training knobs (concepts): gradient accumulation (logical batch), packing variable-length sequences, DDP/FSDP/ZeRO sharding. For distributed details, rely on the official documentation; this draft makes no claims about any cluster's MFU.

## Principles: KV and Intensity on the Inference Side

Each layer stores K and V for every token; the incremental bytes are:

$$
L\cdot 2\cdot n_{\mathrm{kv}}\cdot d_{\mathrm{head}}\cdot b
$$

then multiplied by \(B\cdot S\). GQA makes \(n_{\mathrm{kv}}<n_{\mathrm{head}}\); what it saves is KV, not weights.

**Worked example (assumptions)**: \(L=32,n_{\mathrm{kv}}=8,d_{\mathrm{head}}=128,b=2,B=1,S=4096\)  
\(131072\) bytes per token; \(512\,\mathrm{MiB}\) for the full length. With \(n_{\mathrm{kv}}=32\) (standard MHA), about \(2\,\mathrm{GiB}\). Script: `kv_cache_bytes`.

A decode step is often close to "matrix × vector": the bytes of weights read are about \(d_{\mathrm{out}}d_{\mathrm{in}}b\), the FLOPs about \(2d_{\mathrm{out}}d_{\mathrm{in}}\), and the arithmetic intensity about \(2/b\) FLOP/byte (about 1 for fp16). Whether it is bandwidth-bound requires plugging **your** peak compute \(\pi\) and bandwidth \(\beta\) into the roofline ([3.8](08-cuda-and-gpu-kernels.en.md)); do not apply unmeasured multipliers.

Latency composition example: TTFT = 200 ms, TPOT = 20 ms, \(N=50\) → \(200+49\times20=1180\) ms. Mistakenly using \(\mathrm{TTFT}+N\cdot\mathrm{TPOT}\) counts one extra TPOT; the SLO must use the same formula as the script.

## Quantization and Speculative Decoding (Conceptual Boundaries)

Weight quantization reduces weight bandwidth and footprint, and may affect quality; **the storage ledger ≠ the task metric**. Speculative decoding has a small model propose and a large model verify; the gain depends on the acceptance rate and must be measured on your traffic. bitsandbytes / vLLM were not run in this draft's environment.

## Case Study (Fictional)

A team complains that "it's still slow even after switching to A100s". Broken down: queuing accounts for half of the TTFT; decode runs with batch = 1; KV is reserved for the maximum of 32k, so concurrency cannot reach the design value. The fixes were limiting the context, paged KV, and continuous batch scheduling; they did **not** start by writing a CUDA kernel. Kernel optimization is reserved for after confirming the bottleneck is inside an operator (3.8).

## Exercises

1. With \(P_{\mathrm{train}}=10^6\), how many GiB of static bytes does full-precision Adam take?
2. Under the same KV assumptions, what are the total KV bytes for \(B=4,S=1024\)? How does it compare with \(B=1,S=4096\)?
3. Why does "mixed precision halves training memory" usually not hold under the \(16P\) static ledger?

### Reference Answers

1. \(1.6\times10^7\) bytes ≈ 0.0149 GiB.  
2. \(131072\times1024\times4=536870912\), the same as \(B=1,S=4096\) (same \(B\cdot S\)), but scheduling and paging behavior differ.  
3. The static total often stays around \(16P\); mixed precision mainly changes the width of activations and some of the bandwidth, as well as numeric-range handling (fp16 often needs loss scaling; bf16 usually does not).

## Summary

First state clearly which bytes and which time definition you mean, then talk about optimization techniques. For training, look at \(16P\) + activations; for inference, look at prefill/decode and KV; when an SLO is written as p95, do not substitute the average.

## Key Terms

Adam static state, activation checkpointing, prefill, decode, TTFT, TPOT, GQA, KV cache, continuous batching

## Further Reading

- ZeRO — https://arxiv.org/abs/1910.02054
- Mixed precision training — https://arxiv.org/abs/1710.03740
- GQA — https://arxiv.org/abs/2305.13245
- FlashAttention — https://arxiv.org/abs/2205.14135
- Attention is All You Need — https://arxiv.org/abs/1706.03762

---

[🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md)
