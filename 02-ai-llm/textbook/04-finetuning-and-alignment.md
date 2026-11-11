# 3.4 微调与对齐：SFT、RLHF、PPO、GRPO、DPO

> [🏠 返回主页](../../README.md) · [返回本章入口](README.md) · [上一节 3.3](03-pretraining-and-adaptation.md) · [下一节 3.5](05-neural-nets-and-transformers.md)

## 先修

[3.1](01-business-and-model-choice.md)–[3.3](03-pretraining-and-adaptation.md)。本节以**概念与选型**为主；反向传播与注意力在 [3.5](05-neural-nets-and-transformers.md)，LoRA 参数量与内存账在后文与 [3.7](07-training-and-inference-systems.md) 展开。仓库旧版「何时微调 / LoRA」材料的深度被吸收进本节与演算脚本，但章节编号以本稿为准。

## 学习目标

- 用「知识是否外置、行为是否稳定、是否有偏好数据」选择 SFT / 偏好优化 / 不上训练
- 说明 RLHF（奖励模型 + RL）、PPO、GRPO、DPO 在流水线上差在哪一步
- 完成提示屏蔽与 LoRA 参数量的闭式练习；不把论文榜分数抄成自己的结果

## 定义与选型表

**提示**：不改权重。  
**RAG**：不改权重，外挂知识。  
**SFT**：用配对数据最大化 \(\log p_\theta(y\mid x)\)（常屏蔽 \(x\) 的损失）。  
**全量微调**：更新全部（或近乎全部）权重。  
**LoRA**：冻结 \(W\)，只训低秩增量 \(BA\)（Hu et al., 2021）。  
**QLoRA**：冻结基座以低比特存储，适配器高精度训练（Dettmers et al., 2023）。  
**RLHF**：先（或同时）有偏好数据 → 学奖励模型 → 用 RL 优化策略并通常加 KL 约束（InstructGPT 叙事：Ouyang et al., 2022）。  
**PPO**：一种带裁剪的策略梯度算法，常作 RLHF 中的 RL 步骤（Schulman et al., 2017）。  
**GRPO**：DeepSeekMath 提出的 PPO 变体，用同题多样本的组内奖励标准化估计优势，省去价值网络（Shao et al., 2024）。  
**DPO**：把「奖励模型 + RL」目标改写成对策略的分类损失，直接在偏好对上优化（Rafailov et al., 2023）。

| 你真正要的 | 先考虑 | 不要先考虑 |
|---|---|---|
| 会更新的制度 + 引用 | RAG + ACL | 把正文 SFT/CPT 进权重 |
| 换说法就能稳定 JSON | 提示 + 校验 / 约束解码 | 一上来 RL |
| 格式/工具参数仍不稳，有干净配对 | SFT（全量或 LoRA） | 用更大提示碰运气 |
| 多种都对，但要更安全/更简短/更偏好某风格 | 偏好数据 + DPO 或 RLHF | 无偏好对却调 PPO |
| 可自动判分的推理题（数学等） | 可验证奖励 + GRPO/PPO 类方法 | 把不可验证文采题硬套同一奖励 |
| fp16 基座放不进训练卡 | QLoRA 或更小模型 | 在 CPU 上假装在做 7B 全量 |

## 原理 A：SFT 与提示屏蔽（概念 + 小演算）

教师强制下，位置 \(t\) 的 logits 预测 token \(t+1\)。经典实现取 `logits[..., :-1]` 对齐 `labels[..., 1:]`。提示 token 的 label 设为 `-100`（PyTorch `CrossEntropyLoss` 的 ignore index），则平均损失只反映「回答好不好」。

**演算（与 `examples/worked_examples.py` 一致）**：30 个提示位置 NLL=0.1，2 个回答位置 NLL=2。全部平均：\((30\times0.1+2\times2)/32=0.21875\)；只计回答：平均为 2。前者看起来「更好」，其实被易复制的提示主导。

LoRA 前向：\(W'=W+\frac{\alpha}{r}BA\)（PEFT 默认缩放；`use_rslora` 时为 \(\alpha/\sqrt{r}\)）。参数量 \(r(d_{\mathrm{in}}+d_{\mathrm{out}})\)。例：\(d_{\mathrm{in}}=1024,d_{\mathrm{out}}=2048,r=16\) → LoRA 49152，全量 2097152。

## 原理 B：RLHF 流水线（概念）

典型阶段（叙述与 InstructGPT 一致，细节以论文为准）：

1. **SFT**：把模型调到「会按指令答」。
2. **奖励模型（RM）**：在同一提示下的回答对 \((y_w,y_l)\) 上拟合偏好（常用 Bradley–Terry）。
3. **RL 优化策略**：最大化期望奖励，同时用 KL 惩罚避免远离参考策略（常取 SFT 模型）。

PPO 在此作为稳定更新策略的一种算法：用优势函数缩放更新，并对概率比做裁剪，减轻一次更新过大。价值网络（critic）与策略同量级时，**显存与工程复杂度**显著上升——这是 GRPO 想缓解的点。

## 原理 C：GRPO（概念）

对同一问题 \(q\) 采样一组输出 \(\{o_i\}_{i=1}^{G}\)，用奖励 \(r_i\) 做组内标准化：

$$
A_i=\frac{r_i-\mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\})}
$$

再以类似 PPO 的重要性采样比率与裁剪更新策略，并常加对参考策略的 KL 项。要点：**基线来自组内相对好坏，而不是单独训练的 value model**。适合奖励可自动计算的任务；奖励噪声大或不可比时，组内标准化也会放大噪声。

**玩具数值**：奖励 \([1,0,1,0]\) → 均值 0.5，总体标准差 0.5 → 优势 \([1,-1,1,-1]\)（见脚本 `grpo_advantages`）。

## 原理 D：DPO（概念）

DPO 的关键观察：在 KL 约束的 RL 目标下，最优策略与奖励之间存在闭式关系，从而可把偏好似然写成**只含策略 \(\pi_\theta\) 与参考 \(\pi_{\mathrm{ref}}\)** 的损失（二元交叉熵形式）。直觉：提高「胜出回答」相对参考的对数概率，降低「落败回答」，并用 \(\beta\) 控制偏离参考的幅度。

教学用边界演示（**不是**完整 batch 损失实现）：若胜出相对参考的 log 比增益比落败多 \(\Delta\)，则 \(\beta\Delta\) 可视为隐式奖励间隔的演示量。脚本中 `dpo_margin_demo` 仅锁定算术，不声称复现论文实验。

**DPO vs PPO-RLHF（工程视角）**：

| | DPO | PPO-RLHF |
|---|---|---|
| 要不要单独 RM | 隐式在策略里 | 通常要 |
| 训练期是否大量在线采样 | 典型离线偏好对 | 需要采样与 RL 循环 |
| 实现复杂度 | 相对低 | 高 |
| 偏好数据 | 必需成对（或可构造对） | 亦需偏好以训 RM |
| 可验证自动奖励 | 可用，但不是唯一选择 | 常与 RM 或规则奖励结合 |

没有银弹：数据质量差时，两种都会学到「标注员的偏见」或「奖励漏洞」。

## 案例（虚构）

客服 SFT 后格式稳定，但有时过度承诺退款。团队没有上 PPO，而是：规则引擎拦截退款动词 + 收集「得体拒答 vs 乱承诺」偏好对做 DPO，并在评测集上盯「违规承诺率」。数学解题助手则用可解析的最终答案正确性当奖励，试点 GRPO 式组采样；文案润色任务不用同一套自动奖励。

## 练习

1. 解释：为何「把员工手册 SFT 进模型」既解决不了月更，也解决不了按部门 ACL？
2. \(d=4096,r=8\)，一个 \(d\times d\) 矩阵的 LoRA 参数量？若 32 层各 4 个这样的投影，可训练参数总量？
3. 何时你会优先 DPO而不是 PPO？何时相反？
4. GRPO 的优势标准化在「组内所有样本奖励相同」时会发生什么？工程上怎么防？

### 参考解答

1. 月更需要更新语料版本；ACL 是检索/授权问题，权重无法按用户身份切换可见章节。
2. \(8\times(4096+4096)=65536\)；\(32\times4\times65536=8388608\)。
3. 已有离线成对偏好、想降低工程复杂度 → DPO；需要在线探索、奖励是复杂环境信号、或已有成熟 RL 基础设施 → PPO 类。可自动判分的大规模采样 → 可考虑 GRPO。
4. 标准差为 0，优势未定义或全 0。应跳过该组、加 ε、或要求组内奖励有区分度。

## 小结

微调解决「稳定行为」，对齐解决「偏好与约束」，二者都替代不了外置知识与程序门禁。RLHF / PPO / GRPO / DPO 差在奖励从哪来、是否显式 RM、是否在线采样；先有数据与评测，再选字母缩写。

## 关键术语

SFT、LoRA、QLoRA、RLHF、奖励模型、PPO、GRPO、DPO、KL 约束、提示屏蔽

## 延伸阅读（原始文献与文档）

- Ouyang et al., InstructGPT — https://arxiv.org/abs/2203.02155
- Schulman et al., PPO — https://arxiv.org/abs/1707.06347
- Rafailov et al., DPO — https://arxiv.org/abs/2305.18290
- Shao et al., DeepSeekMath / GRPO — https://arxiv.org/abs/2402.03300
- Hu et al., LoRA — https://arxiv.org/abs/2106.09685
- Dettmers et al., QLoRA — https://arxiv.org/abs/2305.14314
- PEFT LoRA — https://huggingface.co/docs/peft/en/package_reference/lora
- TRL GRPO Trainer — https://huggingface.co/docs/trl/grpo_trainer

---

[🏠 返回主页](../../README.md) · [返回本章入口](README.md)
