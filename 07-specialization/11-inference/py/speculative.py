"""投机解码的期望 token 数。这不是墙钟加速比。

Leviathan、Kalman、Matias 在接受概率 α 对每个草稿 token 独立且恒定、
草稿长度为 γ 的设定下,给出目标模型每次前向期望产生的 token 数:

    (1 - α^(γ+1)) / (1 - α)

α = 1 时极限是 γ+1;α = 0 时是 1(目标模型自己产生的那个 token)。

若再假设每个草稿 token 花费 t_draft、一次目标模型验证花费 t_target,
可以得到一个完全由这些输入决定的模型化速率。t_draft 与 t_target
必须由调用方给定。本模块不提供、也不隐含任何实测加速比。
"""

from __future__ import annotations


def expected_tokens(alpha: float, gamma: int) -> float:
    if gamma < 0:
        raise ValueError("gamma 不能为负")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha 必须在 [0, 1]")
    if alpha == 1:
        return float(gamma + 1)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def modeled_speedup(alpha: float, gamma: int, t_draft: float, t_target: float) -> float:
    """相对"每次目标前向只产生 1 个 token"的模型化比值。"""
    if t_draft < 0 or t_target <= 0:
        raise ValueError("t_draft 不能为负,t_target 必须为正")
    step = gamma * t_draft + t_target
    if step <= 0:
        raise ValueError("一步时间必须为正")
    speculative_rate = expected_tokens(alpha, gamma) / step
    baseline_rate = 1.0 / t_target
    return speculative_rate / baseline_rate
