# 3.7 训练与推理系统性能

> 🌐 [English](07-training-and-inference-systems.en.md)

> [🏠 返回主页](../README.md) · [返回本章入口](README.md) · [上一节 3.6](06-pytorch-engineering.md) · [下一节 3.8](08-cuda-and-gpu-kernels.md)

## 先修

[3.4](04-finetuning-and-alignment.md)–[3.6](06-pytorch-engineering.md)。算术强度与屋顶线的硬件展开见 [3.8](08-cuda-and-gpu-kernels.md)；本节先给出**系统级账本**，供选型与排障。

## 学习目标

- 分开计算：Adam 静态状态、冻结基座权重、激活、KV cache
- 区分预填与解码，并用 TTFT、TPOT 组装端到端延迟（与吞吐分开）
- 在「加大 batch / 检查点 / 量化 / 换并行」之间做单变量推理，不编造加速比

## 定义

**静态训练状态**：跨 step 常驻的参数、梯度、优化器矩等。  
**激活**：前向为反传保留的中间张量，随 batch 与序列变。  
**预填（prefill）**：并行处理提示 token，写初始 KV，产首个输出 token。  
**解码（decode）**：逐步生成；读已有 KV 并追加。  
**TTFT / TPOT**：到首 token 时间 / 每后续输出 token 时间。约定：

$$
T_{\mathrm{e2e}}=\mathrm{TTFT}+(N-1)\cdot\mathrm{TPOT}\quad(N\ge 1)
$$

单请求延迟的倒数不是系统吞吐；吞吐是墙钟时间内所有请求的输出 token 总数。

## 原理：训练内存的 \(16P\) 账

对 \(P\) 个**可训练**参数，全精度 Adam 常见静态合计为 \(16P\) 字节：参数 4 + 梯度 4 + 一阶矩 4 + 二阶矩 4。混合精度 Adam（ZeRO 论文常用那张静态账）往往**合计仍约 \(16P\)**：低精度参数/梯度各 \(2P\)，外加 fp32 主权重 \(4P\) 与 fp32 的 \(m,v\) 各 \(4P\)。「打开 fp16」并不把优化器状态自动变成 \(2P\)。

**演算（假设 \(P=7\times10^9\)）**：\(16P=1.12\times10^{11}\) 字节 ≈ 104.31 GiB。这是乘法，不是 profiler。LoRA 把公式里的 \(P\) 换成 \(P_{\mathrm{train}}\)；冻结基座的字节必须**另列**相加。

激活教学估计（实现相关，必须声明假设）：

$$
L\cdot c\cdot B\cdot S\cdot H\cdot b + \mathbf{1}_{\mathrm{materialize}}\cdot L\cdot B\cdot n_{\mathrm{head}}\cdot S^{2}\cdot b
$$

FlashAttention 类实现不物化 \(S\times S\) 分数时，第二项应视为 0；写了第二项又声称 FlashAttention，是重复计数。梯度检查点用重算换激活内存，省的是峰值而非 FLOP。

其它训练旋钮（概念）：梯度累积（逻辑 batch）、打包变长序列、DDP/FSDP/ZeRO 分片。分布式细节以官方文档为准；本稿不声称某集群的 MFU。

## 原理：推理侧 KV 与强度

每层每个 token 存 K、V；增量字节：

$$
L\cdot 2\cdot n_{\mathrm{kv}}\cdot d_{\mathrm{head}}\cdot b
$$

再乘 \(B\cdot S\)。GQA 使 \(n_{\mathrm{kv}}<n_{\mathrm{head}}\)，省的是 KV 不是权重。

**演算（假设）**：\(L=32,n_{\mathrm{kv}}=8,d_{\mathrm{head}}=128,b=2,B=1,S=4096\)  
每 token \(131072\) 字节；全长 \(512\,\mathrm{MiB}\)。若 \(n_{\mathrm{kv}}=32\)（标准 MHA），约 \(2\,\mathrm{GiB}\)。脚本：`kv_cache_bytes`。

解码步常接近「矩阵×向量」：权重读取字节约 \(d_{\mathrm{out}}d_{\mathrm{in}}b\)，FLOP 约 \(2d_{\mathrm{out}}d_{\mathrm{in}}\)，算术强度约 \(2/b\) FLOP/byte（fp16 时约 1）。是否带宽顶，要代入**你的**峰值算力 \(\pi\) 与带宽 \(\beta\) 做屋顶线（[3.8](08-cuda-and-gpu-kernels.md)），不可套用未测量倍数。

延迟合成例：TTFT=200 ms，TPOT=20 ms，\(N=50\) → \(200+49\times20=1180\) ms。若误用 \(\mathrm{TTFT}+N\cdot\mathrm{TPOT}\) 会多算一个 TPOT——SLO 必须与脚本同式。

## 量化与投机解码（概念边界）

权重量化减少权重带宽与占用，可能影响质量；**存储账 ≠ 任务指标**。投机解码用小模型提案、大模型验证，收益依赖接受率，需在你的流量上测。本稿环境未运行 bitsandbytes / vLLM。

## 案例（虚构）

团队抱怨「换 A100 还是慢」。分解后：TTFT 里排队占一半；解码 batch=1；KV 按最大 32k 预留导致并发达不到设计值。措施是限上下文、分页 KV、连续批调度——**没有**先写 CUDA kernel。内核优化留给确认已是算子内瓶颈之后（3.8）。

## 练习

1. \(P_{\mathrm{train}}=10^6\)，全精度 Adam 静态字节是多少 GiB？
2. 同上 KV 假设，\(B=4,S=1024\) 的总 KV 字节？与 \(B=1,S=4096\) 比如何？
3. 为什么「混合精度把训练内存降到一半」在 \(16P\) 静态账里通常不成立？

### 参考解答

1. \(1.6\times10^7\) 字节 ≈ 0.0149 GiB。  
2. \(131072\times1024\times4=536870912\)，与 \(B=1,S=4096\) 相同（\(B\cdot S\) 相同），但调度与分页行为不同。  
3. 静态合计常仍约 \(16P\)；混合精度主要改变激活宽度与部分带宽，以及数值范围处理（fp16 常需 loss scaling；bf16 通常不需要）。

## 小结

先写清哪一笔字节、哪一个时间定义，再谈优化手段。训练看 \(16P\)+激活；推理看预填/解码与 KV；SLO 写 p95 时不要用平均值代替。

## 关键术语

Adam 静态状态、激活检查点、预填、解码、TTFT、TPOT、GQA、KV cache、连续批

## 延伸阅读

- ZeRO — https://arxiv.org/abs/1910.02054
- Mixed precision training — https://arxiv.org/abs/1710.03740
- GQA — https://arxiv.org/abs/2305.13245
- FlashAttention — https://arxiv.org/abs/2205.14135
- Attention is All You Need — https://arxiv.org/abs/1706.03762

---

[🏠 返回主页](../README.md) · [返回本章入口](README.md)
