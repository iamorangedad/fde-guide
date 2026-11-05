"""LoRA 的形状、参数量和缩放。公式与 PEFT 文档一致,不依赖 GPU。

对冻结矩阵 W (out_features, in_features),LoRA 学习
A 形状 (r, in_features)、B 形状 (out_features, r):

    W' = W + scaling * (B @ A)

默认 scaling = lora_alpha / r。use_rslora 时 scaling = lora_alpha / sqrt(r)
(见 PEFT LoraConfig 文档与 Kalajdzievski 的 rank-stabilized LoRA)。
B 在常见初始化里是 0,所以训练开始时适配器输出为 0。

参数量只计 A 和 B: r * (in_features + out_features)。偏差、dropout 和
被冻结的 W 都不在这个计数里。
"""

from __future__ import annotations

import math


def lora_parameter_count(in_features: int, out_features: int, rank: int) -> int:
    if in_features <= 0 or out_features <= 0 or rank <= 0:
        raise ValueError("in_features、out_features、rank 必须为正")
    return rank * (in_features + out_features)


def lora_scaling(alpha: float, rank: int, use_rslora: bool = False) -> float:
    if rank <= 0:
        raise ValueError("rank 必须为正")
    if use_rslora:
        return alpha / math.sqrt(rank)
    return alpha / rank


def full_parameter_count(in_features: int, out_features: int) -> int:
    if in_features <= 0 or out_features <= 0:
        raise ValueError("维度必须为正")
    return in_features * out_features


def trainable_fraction(trainable: int, reference: int) -> float:
    if reference <= 0:
        raise ValueError("reference 必须为正")
    return trainable / reference


def apply_lora(
    weight: list[list[float]],
    a_matrix: list[list[float]],
    b_matrix: list[list[float]],
    scaling: float,
) -> list[list[float]]:
    """weight 是 out x in,a 是 r x in,b 是 out x r。返回 W + scaling * B @ A。"""
    out_features = len(weight)
    in_features = len(weight[0]) if out_features else 0
    rank = len(a_matrix)
    if rank == 0 or len(b_matrix) != out_features or len(b_matrix[0]) != rank:
        raise ValueError("B 的形状应为 (out, r)")
    if any(len(row) != in_features for row in a_matrix):
        raise ValueError("A 的形状应为 (r, in)")
    delta = [[0.0 for _ in range(in_features)] for _ in range(out_features)]
    for o in range(out_features):
        for i in range(in_features):
            acc = 0.0
            for r in range(rank):
                acc += b_matrix[o][r] * a_matrix[r][i]
            delta[o][i] = scaling * acc
    return [
        [weight[o][i] + delta[o][i] for i in range(in_features)] for o in range(out_features)
    ]


def attention_lora_count(
    layers: int, hidden: int, rank: int, projections: int = 4
) -> int:
    """每个投影都是 hidden x hidden 时的总 LoRA 参数量。projections=4 表示 q,k,v,o。"""
    per = lora_parameter_count(hidden, hidden, rank)
    return layers * projections * per


def mlp_lora_count(layers: int, hidden: int, intermediate: int, rank: int, matrices: int = 3) -> int:
    """每个矩阵是 hidden x intermediate 或 intermediate x hidden,参数量相同。"""
    per = lora_parameter_count(hidden, intermediate, rank)
    return layers * matrices * per
