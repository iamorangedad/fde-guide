# 3.5 神经网络与 Transformer

> [🏠 返回主页](../../README.md) · [返回本章入口](README.md) · [上一节 3.4](04-finetuning-and-alignment.md) · [下一节 3.6](06-pytorch-engineering.md)

## 先修

线性代数（矩阵乘、转置）、初等概率（期望、条件概率）、微积分中的链式法则。[3.3](03-pretraining-and-adaptation.md)–[3.4](04-finetuning-and-alignment.md) 中的损失应已建立直觉。

## 学习目标

- 写出一层 MLP 的前向与反向传播中的形状约定
- 从「加权平均」理解注意力，并写出 scaled dot-product attention
- 说明 Decoder-only Transformer 如何支撑因果语言建模

## 定义

**神经网络**：由可微模块（仿射变换、非线性、归一化、注意力等）组成的函数 \(f_\theta\)，通过损失 \(\mathcal{L}\) 对 \(\theta\) 反传更新。

**反向传播**：用链式法则把 \(\partial\mathcal{L}/\partial\theta\) 从输出端 recursively 传到各参数；现代框架的 autograd 自动完成，但形状与内存仍需人理解。

**自注意力（self-attention）**：序列中每个位置用其它位置的值向量做内容依赖的加权平均。

**Transformer**：以多头注意力与前馈网络为主体的堆叠结构（Vaswani et al., 2017）。当代多数聊天 LLM 采用**仅解码器**变体：注意力带因果掩码，只能看见过去 token。

## 原理：MLP 与梯度形状

一层：\(h=\sigma(Wx+b)\)，其中 \(W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\)，\(x\in\mathbb{R}^{d_{\mathrm{in}}}\)。若后接标量损失，则 \(\partial\mathcal{L}/\partial W\) 与 \(W\) 同形。批量时 \(X\in\mathbb{R}^{B\times d_{\mathrm{in}}}\)，注意框架是「左乘还是右乘」、是否转置——这是 [3.6](06-pytorch-engineering.md) 里 `nn.Linear` 的约定问题。

**练习级结论**：参数量由形状决定；激活内存由 \(B\times\text{seq}\times d\) 与是否存中间结果给反传决定。训练内存账见 [3.7](07-training-and-inference-systems.md)。

## 原理：Scaled Dot-Product Attention

对查询 \(Q\)、键 \(K\)、值 \(V\)（行对应序列位置）：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

\(\sqrt{d_k}\) 缩放减轻点积幅度随维度增大导致的 softmax 饱和（原论文论点）。**因果掩码**把未来位置的 logits 设为 \(-\infty\)，使自回归生成与训练一致。

**多头**：把 \(d_{\mathrm{model}}\) 拆成 \(h\) 个头，各算注意力再拼回，用输出投影混合。直觉是不同头可关注不同关系类型——这是解释性说法，落地仍靠评测。

**复杂度直觉**：朴素注意力中 \(QK^\top\) 对序列长度 \(T\) 是 \(O(T^2 d)\) 量级；长上下文贵，这是推理侧需要 KV cache、分页注意力等工程的根本原因之一（[3.7](07-training-and-inference-systems.md)）。

### 位置信息

注意力本身对置换近似对称，需要位置编码：绝对位置嵌入、相对位置偏置、RoPE 等。细节以你所用模型的技术报告为准；FDE 要关心的是：**换位置编码方案通常不能只换权重文件名**，训练与推理必须一致。

### Decoder block（概念栈）

常见堆叠（具体 Pre-LN / Post-LN 因模型而异）：

1. 因果多头自注意力  
2. 残差 + 归一化  
3. 前馈（常为扩维 MLP，或门控变体）  
4. 残差 + 归一化  

栈顶接 `lm_head` 把隐状态映到词表 logits。SFT/RL 改的是这些权重（或 LoRA 增量）；RAG 不改它们，只改输入 token。

## 演算

单头、\(T=2\)、\(d_k=2\)，令

$$
Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\;
K=\begin{bmatrix}1&0\\0&1\end{bmatrix},\;
V=\begin{bmatrix}1&2\\3&4\end{bmatrix}
$$

则 \(QK^\top/\sqrt{2}=\frac{1}{\sqrt{2}}I\)。第一行 softmax 后约 \([0.668,0.332]\)（数值见脚本 `toy_attention`），再乘 \(V` 得到第一行输出。此演算只锁定公式，不是某生产内核。



## 原理：残差与归一化为何重要

深度堆叠中，残差提供一条「近似恒等」通路，缓解梯度消失/劣化。LayerNorm（或 RMSNorm）稳定激活尺度。Pre-LN 与 Post-LN 影响训练稳定性与最终质量，**换归一化位置等于换架构**，不能只拷贝权重。对 FDE：导出 ONNX/TensorRT 或做量化时，归一化的融合方式常影响数值，需要在保留集上回归，而不是只看「能跑」。

## 原理：KV cache 与注意力公式的联系

推理时，过去位置的 \(K,V\) 已算过，解码步只需对新 token 算 \(q_t\) 以及对历史 \(K\) 的打分。这并不改变 Attention 公式，只改变**计算日程与内存布局**。因此 [3.7](07-training-and-inference-systems.md) 的 KV 字节公式与本节的 \(d_{\mathrm{head}}\)、层数直接相乘——缺的是工程，不是另一套数学。

## 与 3.3–3.4 的衔接

- 预训练 / SFT 的 NLL：在 `lm_head` 的 softmax 上算。  
- LoRA：常挂在 \(W_Q,W_K,W_V,W_O\) 与 MLP 的线性层上。  
- 奖励模型：常另接标量头，或用最后一个 token 隐状态打分——实现各异，读你用的 TRL/深度框架文档。

## 练习

1. 为何因果 LM 训练要用因果掩码？若训练无掩码、推理有掩码，会发生什么概念性错误？
2. 序列从 \(1\mathrm{k}\) 增到 \(4\mathrm{k}\)，朴素注意力计算量大约如何变？（只谈 \(T^2\) 项）
3. 「把一层 Transformer 换成更大的 MLP」能否替代注意力的内容寻址？用「可变长上下文中的引用」说明。

### 参考解答

1. 训练必须与自回归生成的信息集一致；否则模型学会「偷看未来」，推理时条件分布不匹配。
2. \(T\to4T\) 则 \(T^2\to16T^2\)，约 16 倍（忽略 \(d\) 常数与实现融合）。
3. 固定 MLP 对位置的混合权重不随内容变化；注意力可按 query 内容选择「看哪段制度」。引用题更依赖后者（仍需 RAG 提供原文）。

## 小结

Transformer 解码器 = 因果注意力 + 前馈的堆叠；语言建模损失在词表 softmax 上。读懂形状与掩码，才能读懂微调挂在哪里、长上下文贵在哪里。

## 关键术语

反向传播、自注意力、多头、因果掩码、位置编码、残差、Pre-LN、logits

## 延伸阅读

- Vaswani et al., *Attention Is All You Need* — https://arxiv.org/abs/1706.03762
- 原博客/课程资源可作直观动画，但以论文公式为准
- 下一节用框架实现这些模块：[3.6](06-pytorch-engineering.md)

---

[🏠 返回主页](../../README.md) · [返回本章入口](README.md)
