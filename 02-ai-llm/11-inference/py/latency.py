"""延迟定义。这里的函数不测量 GPU,只把已经给出的 TTFT 和 TPOT 组合成端到端时间。

约定:TTFT 含第一个输出 token 的等待(排队 + 预填)。
之后每个输出 token 再花一个 TPOT。因此输出 N 个 token 的时间为

    TTFT + (N - 1) * TPOT

单请求延迟的倒数不是系统吞吐。吞吐是单位时间里所有请求完成的 token 数。
分位数使用 Hyndman-Fan type 7(与 NumPy 默认 percentile 相同的线性插值)。
"""

from __future__ import annotations

import math


def request_latency(ttft: float, tpot: float, output_tokens: int) -> float:
    if output_tokens < 1:
        raise ValueError("output_tokens 至少为 1")
    if ttft < 0 or tpot < 0:
        raise ValueError("TTFT 和 TPOT 不能为负")
    return ttft + (output_tokens - 1) * tpot


def percentile_type7(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("values 不能为空")
    if not 0 <= percent <= 100:
        raise ValueError("percent 必须在 0 到 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percent / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def output_throughput(total_output_tokens: int, wall_seconds: float) -> float:
    if total_output_tokens < 0 or wall_seconds <= 0:
        raise ValueError("token 数不能为负,墙钟时间必须为正")
    return total_output_tokens / wall_seconds
