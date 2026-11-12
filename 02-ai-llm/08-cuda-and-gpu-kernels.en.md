# 3.8 GPUs and CUDA (Advanced Elective)

> 🌐 [中文](08-cuda-and-gpu-kernels.md)

> [🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md) · [Previous: 3.7](07-training-and-inference-systems.en.md)

## Prerequisites

[3.6](06-pytorch-engineering.en.md)–[3.7](07-training-and-inference-systems.en.md). Go deeper only when the profiler shows time spent in fusible operators/custom kernels and the business case justifies the investment. **Most FDE engagements stop at 3.7.**

## Verification Status (Required Reading)

The environment in which this draft was written and packaged:

- **No** usable `nvcc` was detected / the CUDA examples were not compiled;
- **No** run verification was done on a Jetson or any other GPU;
- The formulas and Python reference calculations below can be re-checked on a CPU; any millisecond-level speedup must **not** be copied from this text as a measured result.

This chapter contains no compilable `.cu` examples; when you need to verify on a GPU, write your own and record the experimental results.

## Learning Objectives

- Use SIMT and warps to explain "why branch divergence hurts performance"
- Distinguish coalesced memory access from strided access
- Explain the role of shared memory in reductions and tiled matrix multiplication
- Use the roofline to judge whether a kernel is compute-bound or bandwidth-bound (plugging in assumed peaks, not passing them off as spec-sheet values)

## Definitions

**SIMT**: single instruction, multiple threads; threads in the same warp (usually 32 threads on NVIDIA GPUs) share one instruction stream.  
**Coalesced access**: the global memory accesses of one warp fall into as few sub-transactions as possible.  
**Shared memory**: software-managed scratchpad memory on an SM, used for reuse through tiling.  
**Occupancy**: the ratio of active warps to capacity; high occupancy does not automatically mean high throughput.  
**Roofline**: draws the performance upper bound as \(\min(\pi, \beta\cdot I)\), where \(I\) is the arithmetic intensity (FLOP/byte), \(\pi\) the peak compute, and \(\beta\) the peak bandwidth.

## Key Principles

### Execution

Threads form blocks, and blocks are mapped onto SMs. Divergent branches make a warp execute the different paths serially. For teaching purposes, first ensure correctness and bounds (`if (i < n)`), then use a profiler to see whether divergence really is the bottleneck.

### Memory Access

If a row-major matrix is read column-wise with large strides, it easily turns into many transactions. Transposing, changing the layout, or staging in shared memory are the classic techniques. **The number of byte transactions** is closer to the truth than "it feels parallel".

### Reductions

A tree reduction adds level by level in shared memory; pay attention to the levels at which `__syncthreads` and intra-warp shuffle primitives apply. Correctness: for non-associative floating-point operations, the order of a parallel reduction changes the numerical result; tests should use tolerances and fix the algorithm.

### Tiled Matrix Multiplication

Loading \(C=AB\) tile by tile into shared memory increases data reuse and therefore arithmetic intensity. A complete kernel also involves bank conflicts, double buffering, and so on; without GPU verification, you are only required to derive "why the intensity goes up".

### Roofline Worked Example (Hypothetical Device)

Assume a peak of \(\pi=10\times10^{12}\) FLOP/s and \(\beta=200\times10^{9}\) byte/s; then the ridge-point intensity is \(I^*=\pi/\beta=50\) FLOP/byte.  
A rough estimate for an fp32 square matrix multiplication with \(n=1024\): \(2n^3\) FLOPs; counting only one pass over memory for each of A, B, and C, \(3n^2\cdot4\) bytes, gives \(I\approx 2n^3/(12n^2)=n/6\approx170.7>50\), which is compute-bound under **this set of assumptions**. The intensity of a decode GEMV is about \(2/b\) (see 3.7), often far below the ridge point and thus bandwidth-bound. With real values of \(\pi,\beta\) and real memory access counts, the conclusion may change.



## Further Principles: Orders of Magnitude for Reductions and Tiling

Suppose a block has \(N=256\) threads doing a sum reduction. Naively "atomically adding into global memory" serializes badly. A shared-memory tree reduction takes about \(\log_2 N\) levels of synchronization; within a warp, shuffles can collapse the final 32 lanes one more level. Floating point is not associative: parallel reduction and serial accumulation have different ulp errors, so unit tests should use a more reasonable tolerance than "exact bit equality" and fix the shape of the reduction tree.

Tiled matrix multiplication: with tile side length \(T_B\), each output tile reuses a row strip of \(A\) and a column strip of \(B\). Roughly, arithmetic intensity rises with \(T_B\) (until limited by shared memory capacity). This explains "why tile"; it is not the ms figure from some `nsight` run.

## Further Principles: A One-Sentence Test for Coalesced Access

If the 32 threads of a warp access an aligned contiguous 128-byte region, the hardware is more likely to coalesce them into a few transactions; if the addresses are strided by a large `stride*sizeof(float)`, the number of transactions rises and effective bandwidth falls. When troubleshooting, print/derive the address pattern first, then change the kernel.

## Relationship to PyTorch

`torch.addmm` / cuBLAS / FlashAttention are already highly optimized implementations. In everyday FDE work, prioritize: choosing the right precision and libraries, reducing unnecessary synchronization and D2H copies, and sensible batching. Hand-written CUDA is the last mile, not the starting point.

## Exercises

1. Explain: when half of the threads in a warp take the `if` and half take the `else`, why can the time approach "the sum of the two paths"?
2. Under the assumed \(\pi,\beta\), what limits a kernel with arithmetic intensity 1, and one with 100?
3. Why can you not infer "a 2× speedup" from "raising occupancy from 50% to 100%"?

### Reference Answers

1. Under SIMT, the two paths are issued serially, with the other threads masked off.  
2. \(I=1<50\) is bandwidth-bound; \(I=100>50\) is compute-bound (only for this assumed roofline).  
3. The kernel may already be bandwidth-bound, or latency may be hidden by other factors; occupancy is only one indicator of parallel slack.

## Summary

CUDA explains how instructions and bytes move on the device; the roofline decides whether optimization should target computation or memory access. This draft stops at formulas and assumption-based calculations that can be re-checked; measurement on real hardware is left to an environment with a GPU, to be written into your own experiment log.

## Key Terms

SIMT, warp, coalesced access, shared memory, reduction, tile, roofline, arithmetic intensity

## Further Reading

- CUDA C++ Programming Guide (official NVIDIA) — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Volkov et al.'s classic discussion of occupancy and performance (search for "Better Performance at Lower Occupancy")

---

[🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md)

Next: [3.9 Linux I/O and the Storage Stack](09-linux-io-and-storage-stack.en.md)
