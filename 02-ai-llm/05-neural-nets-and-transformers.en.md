# 3.5 Neural Networks and the Transformer

> 🌐 [中文](05-neural-nets-and-transformers.md)

> [🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md) · [Previous: 3.4](04-finetuning-and-alignment.en.md) · [Next: 3.6](06-pytorch-engineering.en.md)

## Prerequisites

Linear algebra (matrix multiplication, transpose), elementary probability (expectation, conditional probability), and the chain rule from calculus. You should already have built intuition for the losses in [3.3](03-pretraining-and-adaptation.en.md)–[3.4](04-finetuning-and-alignment.en.md).

## Learning Objectives

- Write down the shape conventions in the forward and backward passes of a one-layer MLP
- Understand attention as a "weighted average" and write out scaled dot-product attention
- Explain how a decoder-only Transformer supports causal language modeling

## Definitions

**Neural network**: a function \(f_\theta\) composed of differentiable modules (affine transforms, nonlinearities, normalization, attention, etc.), updated by backpropagating the loss \(\mathcal{L}\) with respect to \(\theta\).

**Backpropagation**: using the chain rule to pass \(\partial\mathcal{L}/\partial\theta\) recursively from the output back to each parameter; the autograd of modern frameworks does this automatically, but people still need to understand the shapes and the memory.

**Self-attention**: each position in a sequence takes a content-dependent weighted average of the value vectors of the other positions.

**Transformer**: a stacked architecture built mainly from multi-head attention and feed-forward networks (Vaswani et al., 2017). Most contemporary chat LLMs use the **decoder-only** variant: attention carries a causal mask and can only see past tokens.

## Principles: The MLP and Gradient Shapes

One layer: \(h=\sigma(Wx+b)\), where \(W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\) and \(x\in\mathbb{R}^{d_{\mathrm{in}}}\). If a scalar loss follows, \(\partial\mathcal{L}/\partial W\) has the same shape as \(W\). With a batch, \(X\in\mathbb{R}^{B\times d_{\mathrm{in}}}\); pay attention to whether the framework multiplies on the left or the right and whether it transposes. This is the `nn.Linear` convention question in [3.6](06-pytorch-engineering.en.md).

**Exercise-level conclusion**: the parameter count is determined by the shapes; activation memory is determined by \(B\times\text{seq}\times d\) and by whether intermediate results are stored for the backward pass. Training memory accounting is in [3.7](07-training-and-inference-systems.en.md).

## Principles: Scaled Dot-Product Attention

For queries \(Q\), keys \(K\), and values \(V\) (rows correspond to sequence positions):

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The \(\sqrt{d_k}\) scaling mitigates softmax saturation caused by dot-product magnitudes growing with the dimension (the argument of the original paper). The **causal mask** sets the logits of future positions to \(-\infty\), so that training is consistent with autoregressive generation.

**Multi-head**: split \(d_{\mathrm{model}}\) into \(h\) heads, compute attention for each, concatenate them back, and mix them with an output projection. The intuition is that different heads can attend to different types of relationships; this is an interpretive claim, and in practice you still rely on evaluation.

**Complexity intuition**: in naive attention, \(QK^\top\) is on the order of \(O(T^2 d)\) in the sequence length \(T\); long contexts are expensive, which is one of the fundamental reasons the inference side needs engineering such as the KV cache and paged attention ([3.7](07-training-and-inference-systems.en.md)).

### Positional Information

Attention by itself is approximately permutation-symmetric, so it needs positional encoding: absolute position embeddings, relative position biases, RoPE, and so on. For details, rely on the technical report of the model you use; what an FDE must care about is that **switching positional-encoding schemes usually cannot be done by just swapping the weight file name**: training and inference must agree.

### The Decoder Block (Conceptual Stack)

A common stack (the exact Pre-LN / Post-LN arrangement varies by model):

1. Causal multi-head self-attention  
2. Residual + normalization  
3. Feed-forward (often a widening MLP, or a gated variant)  
4. Residual + normalization  

On top of the stack, an `lm_head` maps the hidden states to vocabulary logits. SFT/RL changes these weights (or the LoRA updates); RAG does not change them, only the input tokens.

## Worked Example

Single head, \(T=2\), \(d_k=2\), and let

$$
Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\;
K=\begin{bmatrix}1&0\\0&1\end{bmatrix},\;
V=\begin{bmatrix}1&2\\3&4\end{bmatrix}
$$

Then \(QK^\top/\sqrt{2}=\frac{1}{\sqrt{2}}I\). After softmax, the first row is about \([0.668,0.332]\) (see `toy_attention` in the script for the numbers); multiplying by \(V\) gives the first row of the output. This calculation only pins down the formula; it is not any production kernel.



## Principles: Why Residuals and Normalization Matter

In deep stacks, residuals provide an "approximately identity" path that mitigates vanishing gradients and degradation. LayerNorm (or RMSNorm) stabilizes the scale of activations. Pre-LN vs. Post-LN affects training stability and final quality; **changing where normalization sits is changing the architecture**, and you cannot just copy the weights. For an FDE: when exporting to ONNX/TensorRT or quantizing, the way normalization is fused often affects the numerics, so you need to run regressions on a held-out set rather than only checking that "it runs".

## Principles: How the KV Cache Relates to the Attention Formula

At inference time, \(K,V\) for past positions have already been computed, so a decoding step only needs to compute \(q_t\) for the new token and its scores against the historical \(K\). This does not change the attention formula; it only changes the **computation schedule and memory layout**. So the KV byte formula in [3.7](07-training-and-inference-systems.en.md) multiplies directly with this section's \(d_{\mathrm{head}}\) and number of layers: what is missing is engineering, not another set of mathematics.

## Connecting to 3.3–3.4

- The NLL of pretraining / SFT is computed on the softmax of the `lm_head`.  
- LoRA is usually attached to \(W_Q,W_K,W_V,W_O\) and the linear layers of the MLP.  
- Reward models often add a separate scalar head, or score using the hidden state of the last token; implementations vary, so read the documentation of the TRL/deep-learning framework you use.

## Exercises

1. Why must causal LM training use a causal mask? If training had no mask but inference did, what conceptual error would occur?
2. When the sequence grows from \(1\mathrm{k}\) to \(4\mathrm{k}\), roughly how does the compute of naive attention change? (Consider only the \(T^2\) term.)
3. Can "replacing one Transformer layer with a larger MLP" replace the content-based addressing of attention? Explain using "citations within a variable-length context".

### Reference Answers

1. Training must match the information set available during autoregressive generation; otherwise the model learns to "peek at the future", and the conditional distribution at inference does not match.
2. \(T\to4T\) gives \(T^2\to16T^2\), about 16 times (ignoring the \(d\) constant and implementation fusion).
3. A fixed MLP's mixing weights over positions do not vary with content; attention can choose "which part of the policy to look at" based on the query's content. Citation tasks depend more on the latter (RAG is still needed to supply the source text).

## Summary

A Transformer decoder = a stack of causal attention + feed-forward layers; the language modeling loss is computed on the vocabulary softmax. Only by understanding the shapes and masks can you understand where fine-tuning attaches and why long contexts are expensive.

## Key Terms

Backpropagation, self-attention, multi-head, causal mask, positional encoding, residual, Pre-LN, logits

## Further Reading

- Vaswani et al., *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- Original blog posts/course materials can serve as intuitive animations, but the paper's formulas are authoritative
- The next section implements these modules in a framework: [3.6](06-pytorch-engineering.en.md)

---

[🏠 Back to home](../README.en.md) · [Back to chapter](README.en.md)
