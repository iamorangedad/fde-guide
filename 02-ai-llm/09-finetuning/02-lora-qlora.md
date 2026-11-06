# 9.2 LoRA 与 QLoRA

> [🏠 返回主页](../../README.md) · [返回本章](README.md)

## 先修

矩阵形状、[9.1](01-when-and-loss.md) 的 SFT 目标。不需要 GPU。

## 学习目标

- 写出 LoRA 前向、缩放和参数量
- 区分"可训练参数变少"和"基座激活内存变少"
- 说明 QLoRA 的 4 bit 基座、适配器精度和合并之间的三笔不同的账

## 定义

对冻结的 \(W\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\),LoRA 学习 \(A\in\mathbb{R}^{r\times d_{\mathrm{in}}}\) 和 \(B\in\mathbb{R}^{d_{\mathrm{out}}\times r}\):

$$
W' = W + \mathrm{scaling}\cdot BA
$$

Hu 等人的论文把增量写成低秩分解。PEFT 的默认缩放是

$$
\mathrm{scaling} = \frac{\alpha}{r}
$$

`use_rslora=True` 时改为 \(\alpha/\sqrt{r}\),对应 Kalajdzievski 的 rank-stabilized LoRA。PEFT 默认把 \(A\) 做 Kaiming 均匀初始化、把 \(B\) 置 0,因此第一步前向的增量是 0,模型一开始等于基座。这个性质可以用来做回归测试:适配器打开但 \(B=0\) 时,logits 应与基座一致。

参数量是 \(r(d_{\mathrm{in}}+d_{\mathrm{out}})\),不含冻结的 \(W\),也不含没有参数的 dropout。偏差只有在 `bias` 不是 `"none"` 时才额外训练。

## 原理:参数量演算

设隐藏维 \(d=4096\),层数 \(L=32\),秩 \(r=8\),\(\alpha=16\)。这是**假设结构**,用来练公式,不是某个发布模型的参数表。

一个 \(d\times d\) 投影:

$$
r(d+d) = 8\times 8192 = 65536
$$

注意力的 q、k、v、o 四块、全部层:

$$
32\times 4\times 65536 = 8388608
$$

这四块若全量训练,参数量是

$$
32\times 4\times 4096^2 = 2147483648
$$

可训练比例是 \(8388608/2147483648 = 0.00390625\),也就是 0.390625%。比例小是因为分母是全量矩阵,不是因为优化器或激活免费。

若还有三块 MLP,中间维 \(d_{\mathrm{ff}}=11008\)(仍然是假设):

$$
r(d+d_{\mathrm{ff}}) = 8\times 15104 = 120832
$$

三块一层是 \(362496\),32 层是 \(11599872\)。q/k/v/o 与这三块 MLP 合计 \(8388608+11599872=19988480\) 个可训练参数。全量 MLP 的参数量是 \(32\times 3\times 4096\times 11008\),比 LoRA 大两个数量级。把这些数写成"节省了 99% 内存"是错的:节省的是**这些矩阵的梯度和优化器状态**。冻结基座的权重还在,前向激活还在。第 10 章把两笔账分开。

GPT-2 的 `c_attn` 是融合的 QKV,输出维是 \(3d\) 而不是 \(d\)。教学脚本只适配 `c_attn` 时,一层的参数量是

$$
r(d+3d) = 4rd
$$

`n_layer=2`,`r=4`,`n_embd=64` 时总数是 \(2\times 4\times 4\times 64 = 2048\)。`train_lora.py` 会在运行时检查 PEFT 报出的可训练参数等于 2048。融合投影不能按四块独立的 \(d\times d\) 再数一遍。

缩放的数值例子:\(\alpha=16,r=8\) 时默认缩放是 2;`use_rslora` 且 \(r=16,\alpha=16\) 时缩放是 \(16/4=4\)。合并就是把 \(\mathrm{scaling}\cdot BA\) 加进 \(W\)。一个 \(2\times 2\) 的手算由 `apply_lora` 的测试锁定:

- \(W=\begin{bmatrix}1&2\\3&4\end{bmatrix}\),\(A=I_2\),\(B=\mathrm{diag}(0.5,0.25)\),缩放 2
- \(BA=\mathrm{diag}(0.5,0.25)\),乘 2 后加到 \(W\) 得 \(\begin{bmatrix}2&2\\3&4.5\end{bmatrix}\)

## QLoRA 在改哪一笔账

Dettmers 等人的 QLoRA 把冻结基座存成 NormalFloat4:按正态分位数取码字,而不是均匀整数网格,并可以对量化常数再做一次量化。适配器仍用更高精度训练。基座在矩阵乘之前被反量化,因此计算量不按 4 bit 的算术峰值去想,存储按打包后的字节去想。

本仓库的 [quant_toy.py](examples/quant_toy.py) **不是** NF4。它是每组一个最小值和步长的均匀整数网格,用来验证两件事:

- 落在网格上的值可以精确还原;一般值的绝对误差不超过半个步长
- 打包字节数是 \(\lceil NP/8\rceil\) 再加上每组的元数据

元数据方案必须写明。教学函数里的 `"fp16_scale_and_zero"` 是每组 4 字节(一个 fp16 步长和一个 fp16 零点)。\(P=8\)、4 bit、组大小 64 时,数据 4 字节、元数据 4 字节、fp16 对照 16 字节。换成 \(P=7\times 10^9\)、组大小 64、同样的元数据假设:

- 打包权重 \(0.5P = 3.5\times 10^9\) 字节
- 组数 \(P/64\),元数据 \(4\times P/64 = P/16 = 4.375\times 10^8\) 字节
- 合计 \(3.9375\times 10^9\) 字节

这是存储恒等式。它不预测困惑度,也不包含反量化内核的临时缓冲。均匀网格的误差上界不能拿去代替 NF4 的任务评测。PEFT 还提供 LoftQ 初始化,用低秩项去补偿量化误差;那是另一种初始化,不是本脚本的默认 Kaiming。

合并也要分开说:

- 未合并:基座可以继续以低比特驻留,多个任务共享基座,每个请求多做两次小矩阵乘。PEFT 文档指出未合并时可能有额外延迟,合并后可以去掉这份额外计算。
- 合并进 \(W\):得到一份任务专用的全精度(或随后再量化的)权重。多租户切换适配器就不再是换一个小文件。若基座原来是 4 bit,合并通常要先还原到更高精度,再决定是否重新量化。重新量化是一次新的近似,不是恒等变换。

## 实例

```bash
python3 -c "
import sys
sys.path.insert(0,'02-ai-llm/09-finetuning/examples')
from lora_math import attention_lora_count, trainable_fraction
t = attention_lora_count(32, 4096, 8)
print(t, trainable_fraction(t, 32*4*4096*4096))
"
```

QLoRA 的 bitsandbytes 路径依赖特定 GPU 和 CUDA,编写环境没有运行它。不要把玩具量化的最大误差写成"QLoRA 的精度损失"。

## 排障

| 现象 | 原因 |
|---|---|
| 可训练参数比公式大一个 `d` | 把融合 QKV 按三块独立矩阵数了,或把 `lm_head` 也标成了可训练 |
| 训练开始损失就和基座不同 | \(B\) 没有置 0,或缩放在初始化时被写进了权重又乘了一次 |
| 合并后服务结果变了 | dropout 没关,或合并的是训练模式的随机分支;先在 `eval()` 下比较 logits |
| 4 bit 模型合并后又变回十几个 GiB | 合并物化了高精度权重。这是预期的账,不是文件损坏 |
| 量化后任务指标下降,存储账却是对的 | 存储账本来就不包含离群通道和任务损失。要在保留集上比较 |

## 练习

1. \(d_{\mathrm{in}}=1024\),\(d_{\mathrm{out}}=2048\),\(r=16\)。LoRA 参数量是多少?全量参数量是多少?
2. 为什么只把秩从 8 加到 64,不能推出保留集分数会上升?
3. 客户有一张 24 GiB 的卡,想全量微调一个假设的 \(7\times 10^9\) 参数模型。仅凭第 10 章"混合精度 Adam 约 16 字节/参数"这一条静态账,能不能放得下?LoRA 改变的是哪一项?

### 参考解答

1. LoRA:\(16\times(1024+2048)=49152\)。全量:\(1024\times 2048=2097152\)。比例约 2.34%,仍然只覆盖这一层矩阵的优化器状态。
2. 秩决定增量所在的子空间维数。任务若只需要很小的秩,加大秩只增加可训练参数和过拟合机会。秩不够时加大才可能有帮助。两种情况都要靠保留集,而不是靠参数公式。
3. \(16\times 7\times 10^9 = 1.12\times 10^{11}\) 字节,约 104.31 GiB,一张 24 GiB 的卡放不下静态状态,而且这还没算激活。LoRA 把优化器状态缩到可训练参数上;冻结基座的权重和激活仍要单列。放不放得下要看基座精度、激活和批次,不能只看 104 GiB 这个数被"省掉了"。

## 小结

LoRA 是 \(W+\mathrm{scaling}\,BA\),参数量 \(r(d_{\mathrm{in}}+d_{\mathrm{out}})\)。QLoRA 改变的是冻结基座的存储和反量化,不把均匀 4 bit 的误差上界变成任务指标。合并是部署形态的选择:省一次额外矩阵乘,还是保留可切换的小适配器。

## 关键术语

低秩、缩放、rank-stabilized LoRA、NormalFloat4、反量化、合并、LoftQ

## 延伸阅读

- LoRA — https://arxiv.org/abs/2106.09685
- QLoRA — https://arxiv.org/abs/2305.14314
- Rank-stabilized LoRA — https://arxiv.org/abs/2312.03732
- PEFT `LoraConfig` — https://huggingface.co/docs/peft/en/package_reference/lora
- PEFT 概念指南(合并与初始化) — https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- bitsandbytes 4 bit 集成说明 — https://huggingface.co/blog/4bit-transformers-bitsandbytes

---

[🏠 返回主页](../../README.md)
