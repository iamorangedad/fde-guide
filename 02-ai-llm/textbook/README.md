# 第 3 章 语言模型应用与系统（由浅入深新版）

> **新版教材**:本目录按「业务 → 原理 → 框架与硬件」由浅入深重排第 3 章,与上级目录中按专项分开的 3.1–3.6 并存。旧版内容(含 CPU 测试与示例)仍保留在上级目录,未删除。

> [🏠 返回主页](../../README.md) · [旧版第 3 章入口](../README.md)

## 阅读顺序（浅 → 深）

| 节 | 文件 | 在讲什么 | 先修 |
|---|---|---|---|
| 3.1 | [业务需求、约束与模型选型](01-business-and-model-choice.md) | 问题定义、SLA、合规、API/开源/私有化、何时不要训练 | 无 |
| 3.2 | [应用系统：RAG、Agent 与评测](02-applications-rag-agent-eval.md) | 检索、工作流、评测、生产化 | 3.1 |
| 3.3 | [预训练与领域适应（概念）](03-pretraining-and-adaptation.md) | 预训练、继续预训练、数据与目标；技术细节前向引用 3.5–3.7 | 3.1–3.2 |
| 3.4 | [微调与对齐：SFT、RLHF、PPO、GRPO、DPO](04-finetuning-and-alignment.md) | 方法选型、损失直觉、偏好学习；深入公式前向引用 3.5 | 3.3 |
| 3.5 | [神经网络与 Transformer](05-neural-nets-and-transformers.md) | 反向传播、注意力、位置编码、解码器结构 | 线性代数与概率基础 |
| 3.6 | [PyTorch 工程实践](06-pytorch-engineering.md) | **框架层**（不是硬件）：张量、autograd、模块、训练循环 | 3.5 |
| 3.7 | [训练与推理系统性能](07-training-and-inference-systems.md) | 内存账、混合精度、KV cache、批调度、容量 | 3.4–3.6 |
| 3.8 | [GPU 与 CUDA（选修进阶）](08-cuda-and-gpu-kernels.md) | SIMT、合并访存、归约、分块矩阵乘、屋顶线 | 3.6–3.7；**本环境未做 GPU 验证** |
| 3.9 | [Linux I/O 与存储栈（系统层补充）](09-linux-io-and-storage-stack.md) | 页缓存、fsync、Direct I/O、块层、dm、overlayfs、诊断工具 | 3.6–3.7 |

## 数字与证据怎么读

- **恒等式与闭式演算**（参数量、字节数、移位损失）给出确切结果，并写明假设。
- **假设情景**里的峰值算力、带宽、时延是练习输入，不是某张显卡铭牌，也不是未测量的加速比。
- **不记载未测量的加速倍数**。CUDA 示例在本稿中**未编译、未上 GPU 运行**。
- 客户案例均为**虚构合成场景**，用于说明决策，不是真实客户数据。
- 主要引用指向原始论文与官方文档；仓库旧版材料仅作深度参考，本稿编号与仓库不同步。

## 本机可跑的演算

无 GPU、不下载模型：

```bash
cd /workspace/fde-chapter3-draft
python3 examples/worked_examples.py
python3 examples/test_worked_examples.py
```

## 与仓库的关系

| 项 | 说明 |
|---|---|
| 源仓库 | https://github.com/iamorangedad/fde-guide |
| 仓库现状 | PR #3 已合并：`02-ai-llm` 下 3.1–3.6（应用、CUDA、微调、训练、推理、流程） |
| 本稿 | 重新分层；补预训练概念、对齐（RLHF/PPO/GRPO/DPO）、神经网络与 PyTorch 专章；CUDA 放到最后 |
| 上传 | **未执行**任何 git / PR / push；由父代理交付文件后由你决定是否合入 |

## 压缩包

同级目录可生成：`/workspace/fde-chapter3-draft.zip`（生成步骤见交付说明）。

---

[🏠 本章入口](README.md)

[🏠 返回主页](../../README.md)
