# 第 9 章 模型微调

> [🏠 返回主页](../../README.md) · [返回专项](../README.md)

本章是专项。它不把微调写成所有 FDE 项目的默认步骤。会变更、必须引用原文的知识,优先用第 3 章的 RAG。微调用来改变回答的条件分布:格式、语气、工具调用的形状,或提示词稳定不下来的行为。

## 先修

- 第 3 章的提示、RAG 和评测
- 矩阵乘法;不需要先读完第 8 章,但第 10 章的内存账会用到本章的参数量
- 交叉熵的定义: \(-\log p\)

## 学习目标

- 判断一个需求该停留在提示、RAG、全量 SFT、LoRA 还是 QLoRA
- 写出因果语言模型的损失、标签移位和提示屏蔽
- 计算 LoRA 的参数量、缩放和合并后的矩阵
- 准备不会泄漏的划分,并用固定的聊天模板训练和推理
- 说明灾难性遗忘要靠哪两类评测才能被发现

## 地图

1. [何时微调,以及损失怎么定义](01-when-and-loss.md)
2. [LoRA 与 QLoRA](02-lora-qlora.md)
3. [数据、训练、合并与部署](03-data-train-deploy.md)

## 代码

| 文件 | 作用 |
|---|---|
| [examples/lora_math.py](examples/lora_math.py) | 参数量、缩放、\(W+\mathrm{scaling}\,BA\) |
| [examples/loss_mask.py](examples/loss_mask.py) | 移位后的屏蔽交叉熵 |
| [examples/chat_template.py](examples/chat_template.py) | 唯一的教学模板和字符分词器 |
| [examples/data_quality.py](examples/data_quality.py) | 划分、精确泄漏、字二元组 Jaccard |
| [examples/quant_toy.py](examples/quant_toy.py) | 均匀整数 4 bit 的误差上界和字节账,不是 NF4 |
| [examples/synthetic_sop.py](examples/synthetic_sop.py) | 虚构的澄海仓储制度,无真实客户数据 |
| [examples/train_lora.py](examples/train_lora.py) | 随机初始化微型 GPT-2 上的 PEFT 训练 |
| [examples/requirements-cpu.txt](examples/requirements-cpu.txt) | 已跑通的 CPU 版本 |

```bash
python3 07-specialization/09-finetuning/tests/test_finetune_cpu.py
```

安装 CPU 依赖后:

```bash
python3 07-specialization/09-finetuning/examples/train_lora.py --steps 1 --self-check
```

`--self-check` 核对可训练参数量、冻结权重不被更新,以及 `merge_and_unload()` 前后 logits 一致。脚本打印的损失是这次随机初始化上的教学数值。随机模型的交叉熵会接近 \(\log|\mathrm{词表}|\),那只说明分布还接近均匀,不是质量分数。

## 故意不做的事

- 不下载预训练权重,不调用付费模型 API
- 不把 QLoRA 的 4 bit 内核当作已验证路径。均匀量化玩具只说明步长和字节
- 不报告"微调后准确率提升了多少"。没有保留集上的任务指标,就不能有这种句子

## 小结

先写清目标分布是知识还是行为。行为才进入训练。训练损失只在你没有屏蔽掉的 token 上有意义。LoRA 减少的是可训练参数和优化器状态,不自动保证任务质量,也不自动避免遗忘。

## 关键术语

SFT、因果语言建模、ignore index、LoRA、QLoRA、聊天模板、泄漏、灾难性遗忘

## 延伸阅读

- LoRA 论文 — https://arxiv.org/abs/2106.09685
- QLoRA 论文 — https://arxiv.org/abs/2305.14314
- PEFT LoRA 概念指南 — https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- Transformers 聊天模板 — https://huggingface.co/docs/transformers/en/chat_templating

---

[🏠 返回主页](../../README.md)
