# 3.6 PyTorch 工程实践（框架层，不是硬件）

> [🏠 返回主页](../README.md) · [返回本章入口](README.md) · [上一节 3.5](05-neural-nets-and-transformers.md) · [下一节 3.7](07-training-and-inference-systems.md)

## 先修

[3.5](05-neural-nets-and-transformers.md)。本节把模型当成 **PyTorch 程序**：张量、`nn.Module`、autograd、设备抽象。CUDA 内核与 SIMT 留在 [3.8](08-cuda-and-gpu-kernels.md)。**PyTorch 是框架；GPU 是设备。** 可以在 CPU 上完成绝大多数教学循环。

## 学习目标

- 分清 `Tensor` 数据、`nn.Parameter` 与 `torch.no_grad` 的边界
- 手写最小训练步：前向 → 损失 → `backward` → `optimizer.step`
- 解释 `CrossEntropyLoss(ignore_index=-100)` 与因果 LM 标签移位如何对齐

## 定义

**张量**：多维数组，带 `dtype` 与 `device`（`cpu` / `cuda:0` 等）。  
**autograd**：追踪可微运算，`.backward()` 在叶子 `Parameter` 上累积 `.grad`。  
**`nn.Module`**：参数与子模块的容器；`forward` 定义计算。  
**优化器**：根据 `.grad` 更新参数（SGD、AdamW 等）。  
**`ignore_index`**：损失中忽略的目标类别 id，SFT 屏蔽提示时用。

## 原理：最小可运行逻辑（CPU）

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
# 经典移位：用位置 t 预测 t+1
loss = nn.functional.cross_entropy(
    logits[:, :-1, :].reshape(-1, 50),
    x[:, 1:].reshape(-1),
    ignore_index=-100,  # 本例未屏蔽；SFT 时把提示处 label 设为 -100
)
opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()
```

要点：

1. **标签移位**与 [3.4](04-finetuning-and-alignment.md) 一致；`labels[0]` 不监督任何 logit。  
2. `zero_grad`：梯度是累积的，忘清零会把多个 micro-batch 外的步混在一起（除非你故意梯度累积）。  
3. `model.train()` / `model.eval()`：Dropout、BN 行为不同；对比 LoRA 合并前后 logits 必须在 `eval()`。  
4. **设备**：`tensor.to(device)` 与 `model.to(device)` 要一致；框架允许同一代码在 CPU/GPU 间切换，这不自动等于「已经做了内核优化」。

### 常见坑（FDE 视角）

| 现象 | 原因 |
|---|---|
| 损失很低但生成胡言 | 训练模板与 `apply_chat_template` / 推理前缀不一致 |
| 显存莫名涨 | 调试时保留了计算图引用；或把激活留在 Python 列表 |
| `loss` 是 NaN | lr 过大、溢出；先在 fp32 CPU 小模型验证掩码 |
| 「可训练参数 0」 | 冻住了全部、或 LoRA 没挂上目标模块名 |



## 原理：混合精度在框架中的位置

PyTorch AMP（`torch.autocast` + 必要时 `GradScaler`）决定**哪些算子**走低精度。这与「优化器状态是否仍以 fp32 保存主权重」是两件账，见 [3.7](07-training-and-inference-systems.md) 的 \(16P\)。bf16 指数范围与 fp32 相近，文档通常说明可不做 loss scaling；fp16 更常需要。FDE 排障：先在 fp32 确认损失与掩码正确，再开 AMP，避免把逻辑 bug 当成「精度问题」。

## 原理：DataLoader 与可复现

`DataLoader` 的 `num_workers`、跨进程 RNG、与分布式 sampler 都会影响「同 seed 是否可复现」。交付里应记录：框架版本、种子、是否 deterministic 算法、以及聊天模板哈希。可复现不是学术洁癖，而是客户现场对比「改提示 vs 改权重」的前提。

## 与 PEFT / Transformers 的边界

生产 SFT 常用 Hugging Face `transformers` + `peft` + `trl`。教学上仍应能回答：损失在哪个张量上、哪些 id 被 ignore、LoRA 的 `target_modules` 字符串是否匹配层名（GPT-2 的 `c_attn` 是融合 QKV，不能按四个独立 `d×d` 再数一遍参数——见 [3.4](04-finetuning-and-alignment.md) 演算）。

## 练习

1. 写出形状：`logits` 为 `(B,T,V)`，移位后 CE 的输入二维形状是什么？
2. 为何说「在 CPU 上跑通 TinyLM」不能证明「7B QLoRA 脚本正确」？列出至少两个缺口。
3. `torch.no_grad()` 包住整段训练循环会怎样？

### 参考解答

1. `(B*(T-1), V)` 对 `(B*(T-1),)`。  
2. 量化内核、设备内存碎片、分布式、真实分词器与模板、梯度检查点等均未覆盖。  
3. 参数收不到梯度（或 `.grad` 仍为 None），优化器一步不更新有效方向。

## 小结

PyTorch 提供可微张量程序与设备抽象；SFT/对齐的正确性首先是**张量级约定**（移位、mask、train/eval），然后才是集群与 CUDA。

## 关键术语

Parameter、autograd、Module、AdamW、ignore_index、device、梯度累积

## 延伸阅读

- PyTorch `CrossEntropyLoss` — https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Chat templating — https://huggingface.co/docs/transformers/en/chat_templating
- 下一节系统性能：[3.7](07-training-and-inference-systems.md)

---

[🏠 返回主页](../README.md) · [返回本章入口](README.md)
