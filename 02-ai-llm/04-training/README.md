# 3.4 训练优化

> [🏠 返回主页](../../README.md) · [返回第 3 章](../README.md) · [上一节 3.3](../03-finetuning/README.md) · [下一节 3.5](../05-inference/README.md)

本节是专项。它把"显存不够"拆成可以核对的字节,并把"再开一个开关"换成一次只改一个因素的实验。本节没有在多 GPU 上运行 DDP 或 FSDP。那些小节讲的是通信和内存的取舍,以及文档里的正确性约束,不是一份已经测过的加速表。

## 先修

第 3.3 节的可训练参数量。知道反向传播会保存前向激活。不要求先写过分布式程序。

## 学习目标

- 分开计算权重、梯度、优化器状态和激活
- 解释混合精度为什么常常仍是每参数约 16 字节的优化器账
- 在 token 数不同的 micro-batch 上写出正确的梯度加权
- 比较填充和打包,并写出打包后的因果掩码
- 说明 DDP、FSDP 和 ZeRO 各自分片了什么

## 地图

1. [3.4.1 内存账与混合精度](01-memory-precision.md)
2. [3.4.2 累积、检查点、打包与数据加载](02-batch-checkpoint-packing.md)
3. [3.4.3 数据并行、分片与剖析](03-distributed-profiling.md)

## 代码

| 文件 | 作用 |
|---|---|
| [py/memory_accounting.py](py/memory_accounting.py) | Adam 与激活的字节账 |
| [py/grad_accum.py](py/grad_accum.py) | token 加权和普通平均 |
| [py/packing.py](py/packing.py) | 填充浪费、装箱、块状因果掩码 |
| [py/experiment_log.py](py/experiment_log.py) | 种子、配置、数据哈希、库版本 |

```bash
python3 02-ai-llm/04-training/tests/test_training_math.py
```

## 小结

优化从一张字节清单开始。混合精度、累积、检查点和分片解决的是清单上的不同行。墙上时间以剖析为准,而且每次只改一行。

## 关键术语

Adam 状态、主权重、激活检查点、梯度累积、打包、DDP、FSDP、ZeRO

## 延伸阅读

- ZeRO — https://arxiv.org/abs/1910.02054
- 混合精度训练 — https://arxiv.org/abs/1710.03740
- PyTorch AMP — https://docs.pytorch.org/docs/stable/amp.html

---

[🏠 返回主页](../../README.md)
