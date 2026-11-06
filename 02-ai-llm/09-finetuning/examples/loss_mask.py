"""因果语言模型的标签移位与屏蔽。

PyTorch CrossEntropyLoss 默认 ignore_index=-100。Transformers 的因果 LM
在内部把 logits[..., :-1] 与 labels[..., 1:] 对齐,因此 labels[t] 是由
前缀 0..t-1 预测的 token。labels[0] 会被这次移位丢掉。

监督微调若只想学习回答,就把 system/user 和角色标记设为 -100,只保留
assistant 内容(以及停止符)。填充位置同样是 -100,并且 attention_mask 为 0。
继续预训练整篇文档时则不应做提示屏蔽。这两种目标不要混在同一次损失里却不说明。
"""

from __future__ import annotations

import math

IGNORE_INDEX = -100


def causal_labels(input_ids: list[int], prompt_len: int, pad_id: int | None = None) -> list[int]:
    if prompt_len < 0 or prompt_len > len(input_ids):
        raise ValueError("prompt_len 超出序列")
    labels = []
    for i, token in enumerate(input_ids):
        if i < prompt_len or (pad_id is not None and token == pad_id):
            labels.append(IGNORE_INDEX)
        else:
            labels.append(token)
    return labels


def shift_logits_and_labels(
    logits: list[list[float]], labels: list[int]
) -> tuple[list[list[float]], list[int]]:
    """与 Transformers 因果 LM 相同的一步移位。logits[t] 预测 labels[t+1]。"""
    if len(logits) != len(labels):
        raise ValueError("logits 与 labels 长度必须相同")
    if len(labels) < 2:
        return [], []
    return logits[:-1], labels[1:]


def token_nll(logit_row: list[float], target: int) -> float:
    if not 0 <= target < len(logit_row):
        raise ValueError("target 超出词表")
    peak = max(logit_row)
    log_z = peak + math.log(sum(math.exp(x - peak) for x in logit_row))
    return -(logit_row[target] - log_z)


def masked_mean_nll(logits: list[list[float]], labels: list[int]) -> float:
    shifted_logits, shifted_labels = shift_logits_and_labels(logits, labels)
    total = 0.0
    count = 0
    for row, target in zip(shifted_logits, shifted_labels):
        if target == IGNORE_INDEX:
            continue
        total += token_nll(row, target)
        count += 1
    if count == 0:
        raise ValueError("没有参与损失的 token")
    return total / count


def supervised_token_count(labels: list[int]) -> int:
    """移位后真正进入损失的位置数。"""
    if len(labels) < 2:
        return 0
    return sum(1 for target in labels[1:] if target != IGNORE_INDEX)
