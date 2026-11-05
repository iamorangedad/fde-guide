"""梯度累积应当对应哪一种总体损失。

若每个 micro-batch 的损失是该批未屏蔽 token 上的平均,则这些平均的
算术平均等于全局 token 平均,当且仅当每个 micro-batch 的 token 数相同。
token 数不同时,全局梯度是

    g = sum_i (n_i / N) g_i

其中 g_i 是第 i 个 micro-batch 的平均梯度,n_i 是其 token 数,N = sum n_i。
DDP 默认再对 rank 做一次梯度平均,因此不同 rank 的 token 数也不相同时,
只靠框架默认值不一定得到全局 token 平均。以数值对照实验确认。
"""

from __future__ import annotations


def weighted_average(micro_grads: list[list[float]], token_counts: list[int]) -> list[float]:
    if not micro_grads or len(micro_grads) != len(token_counts):
        raise ValueError("梯度列表与 token 数必须非空且等长")
    total = sum(token_counts)
    if total <= 0 or any(n < 0 for n in token_counts):
        raise ValueError("token 数必须非负且总和为正")
    width = len(micro_grads[0])
    acc = [0.0] * width
    for grad, count in zip(micro_grads, token_counts):
        if len(grad) != width:
            raise ValueError("梯度维度不一致")
        for i, value in enumerate(grad):
            acc[i] += (count / total) * value
    return acc


def unweighted_average(micro_grads: list[list[float]]) -> list[float]:
    if not micro_grads:
        raise ValueError("梯度列表不能为空")
    width = len(micro_grads[0])
    acc = [0.0] * width
    for grad in micro_grads:
        if len(grad) != width:
            raise ValueError("梯度维度不一致")
        for i, value in enumerate(grad):
            acc[i] += value / len(micro_grads)
    return acc
