# 3.6 PyTorch Engineering Practice (The Framework Layer, Not Hardware)

> 🌐 [中文](06-pytorch-engineering.md)

> [🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md) · [Previous: 3.5](05-neural-nets-and-transformers.en.md) · [Next: 3.7](07-training-and-inference-systems.en.md)

## Prerequisites

[3.5](05-neural-nets-and-transformers.en.md). This section treats the model as a **PyTorch program**: tensors, `nn.Module`, autograd, and the device abstraction. CUDA kernels and SIMT are left for [3.8](08-cuda-and-gpu-kernels.en.md). **PyTorch is a framework; a GPU is a device.** The vast majority of teaching loops can be completed on a CPU.

## Learning Objectives

- Distinguish the boundaries between `Tensor` data, `nn.Parameter`, and `torch.no_grad`
- Write a minimal training step by hand: forward → loss → `backward` → `optimizer.step`
- Explain how `CrossEntropyLoss(ignore_index=-100)` lines up with label shifting in causal LMs

## Definitions

**Tensor**: a multidimensional array with a `dtype` and a `device` (`cpu` / `cuda:0`, etc.).  
**autograd**: tracks differentiable operations; `.backward()` accumulates `.grad` on leaf `Parameter`s.  
**`nn.Module`**: a container of parameters and submodules; `forward` defines the computation.  
**Optimizer**: updates parameters based on `.grad` (SGD, AdamW, etc.).  
**`ignore_index`**: a target class id ignored by the loss; used when masking prompts in SFT.

## Principles: Minimal Runnable Logic (CPU)

```python
import torch
import torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab=50, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.out = nn.Linear(d, vocab)

    def forward(self, x):
        # x: (B, T)
        return self.out(self.emb(x))  # (B, T, vocab)

model = TinyLM()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 50, (2, 8))
logits = model(x)
# Classic shift: position t predicts t+1
loss = nn.functional.cross_entropy(
    logits[:, :-1, :].reshape(-1, 50),
    x[:, 1:].reshape(-1),
    ignore_index=-100,  # nothing is masked in this example; in SFT, set prompt labels to -100
)
opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()
```

Key points:

1. **Label shifting** is consistent with [3.4](04-finetuning-and-alignment.en.md); `labels[0]` supervises no logit.  
2. `zero_grad`: gradients accumulate; forgetting to clear them mixes in steps from outside the current micro-batches (unless you are deliberately doing gradient accumulation).  
3. `model.train()` / `model.eval()`: Dropout and BN behave differently; comparing logits before and after merging LoRA must be done in `eval()`.  
4. **Device**: `tensor.to(device)` and `model.to(device)` must agree; the framework lets the same code switch between CPU/GPU, which does not automatically mean "kernel optimization has been done".

### Common Pitfalls (FDE View)

| Symptom | Cause |
|---|---|
| Very low loss but gibberish generations | The training template does not match `apply_chat_template` / the inference prefix |
| GPU memory grows mysteriously | References to the computation graph were kept while debugging, or activations were kept in a Python list |
| `loss` is NaN | Learning rate too high, overflow; first verify the masks on a small fp32 model on the CPU |
| "0 trainable parameters" | Everything was frozen, or LoRA was not attached to the target module names |



## Principles: Where Mixed Precision Sits in the Framework

PyTorch AMP (`torch.autocast` + `GradScaler` when needed) decides **which operators** run in low precision. That is a separate account from "whether the optimizer state still keeps fp32 master weights"; see the \(16P\) in [3.7](07-training-and-inference-systems.en.md). bf16 has an exponent range close to fp32, and the documentation usually says loss scaling can be skipped; fp16 more often needs it. FDE troubleshooting: first confirm in fp32 that the loss and masks are correct, then turn on AMP, so you do not mistake a logic bug for a "precision problem".

## Principles: DataLoader and Reproducibility

`DataLoader`'s `num_workers`, cross-process RNG, and distributed samplers all affect "whether the same seed is reproducible". Deliverables should record: framework versions, seeds, whether deterministic algorithms are used, and a hash of the chat template. Reproducibility is not academic fastidiousness; it is the prerequisite for comparing "changing the prompt vs. changing the weights" at a customer site.

## Boundaries with PEFT / Transformers

Production SFT commonly uses Hugging Face `transformers` + `peft` + `trl`. For teaching purposes you should still be able to answer: which tensor the loss is computed on, which ids are ignored, and whether LoRA's `target_modules` strings match the layer names (GPT-2's `c_attn` is a fused QKV, so you cannot count its parameters again as four independent `d×d` matrices; see the worked example in [3.4](04-finetuning-and-alignment.en.md)).

## Exercises

1. Write down the shapes: if `logits` is `(B,T,V)`, what are the two-dimensional input shapes for CE after shifting?
2. Why does "getting TinyLM to run on a CPU" not prove that "the 7B QLoRA script is correct"? List at least two gaps.
3. What happens if `torch.no_grad()` wraps the entire training loop?

### Reference Answers

1. `(B*(T-1), V)` against `(B*(T-1),)`.  
2. Quantization kernels, device memory fragmentation, distributed training, the real tokenizer and template, gradient checkpointing, and so on are all uncovered.  
3. Parameters receive no gradients (or `.grad` remains None), and the optimizer makes no update in any meaningful direction.

## Summary

PyTorch provides differentiable tensor programs and a device abstraction; the correctness of SFT/alignment is first a matter of **tensor-level conventions** (shifting, masks, train/eval), and only then of clusters and CUDA.

## Key Terms

Parameter, autograd, Module, AdamW, ignore_index, device, gradient accumulation

## Further Reading

- PyTorch `CrossEntropyLoss` — https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Chat templating — https://huggingface.co/docs/transformers/en/chat_templating
- Next section on system performance: [3.7](07-training-and-inference-systems.en.md)

---

[🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md)
