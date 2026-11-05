"""占用率的手算估计。

占用率 = 每个 SM 上的活跃 warp 数 / 该 SM 允许的最大 warp 数。
活跃块数受线程槽、warp 槽、寄存器、共享内存和最大驻留块数共同限制。

这是估计,不是 cudaOccupancyMaxActiveBlocksPerMultiprocessor 的替代品。
调用方传入的寄存器数和共享内存字节数应是编译器实际预留的量(可用
--ptxas-options=-v 或占用率 API 查看),不是源码里手写的局部变量个数。
分配粒度随计算能力变化;这里用显式的 granularity 参数,默认 1 表示不额外取整。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def align_up(value: int, granularity: int) -> int:
    if granularity <= 1:
        return value
    if value < 0:
        raise ValueError("value 不能为负")
    return ((value + granularity - 1) // granularity) * granularity


@dataclass(frozen=True)
class DeviceLimits:
    """一次设备查询或文档表格中的上限。不要把某一张 GPU 的数字当成通用常数。"""

    max_threads_per_sm: int
    max_threads_per_block: int
    regs_per_sm: int
    max_regs_per_block: int
    smem_bytes_per_sm: int
    max_smem_bytes_per_block: int
    max_blocks_per_sm: int
    warp_size: int = 32
    reg_granularity: int = 1
    smem_granularity: int = 1


@dataclass(frozen=True)
class OccupancyEstimate:
    active_blocks: int
    warps_per_block: int
    active_warps: int
    max_warps: int
    occupancy: float
    limiter: str


def estimate_occupancy(
    limits: DeviceLimits,
    threads_per_block: int,
    regs_per_thread: int,
    smem_bytes_per_block: int,
) -> OccupancyEstimate:
    if threads_per_block <= 0:
        raise ValueError("threads_per_block 必须为正")
    if limits.warp_size <= 0:
        raise ValueError("warp_size 必须为正")
    if threads_per_block > limits.max_threads_per_block:
        return OccupancyEstimate(0, 0, 0, 0, 0.0, "threads_per_block")

    regs_per_thread_aligned = align_up(regs_per_thread, limits.reg_granularity)
    smem = align_up(smem_bytes_per_block, limits.smem_granularity)
    if regs_per_thread_aligned < 0 or smem < 0:
        raise ValueError("资源用量不能为负")

    warps_per_block = math.ceil(threads_per_block / limits.warp_size)
    max_warps = limits.max_threads_per_sm // limits.warp_size
    candidates: dict[str, int] = {
        "threads": limits.max_threads_per_sm // threads_per_block,
        "warps": max_warps // warps_per_block if warps_per_block else 0,
        "max_blocks": limits.max_blocks_per_sm,
    }
    if regs_per_thread_aligned > 0:
        regs_per_block = regs_per_thread_aligned * threads_per_block
        if regs_per_block > limits.max_regs_per_block or regs_per_block > limits.regs_per_sm:
            return OccupancyEstimate(0, warps_per_block, 0, max_warps, 0.0, "registers")
        candidates["registers"] = limits.regs_per_sm // regs_per_block
    if smem > 0:
        if smem > limits.max_smem_bytes_per_block or smem > limits.smem_bytes_per_sm:
            return OccupancyEstimate(0, warps_per_block, 0, max_warps, 0.0, "shared_memory")
        candidates["shared_memory"] = limits.smem_bytes_per_sm // smem

    active_blocks = min(candidates.values())
    limiter = "+".join(name for name, blocks in candidates.items() if blocks == active_blocks)
    active_warps = active_blocks * warps_per_block
    occupancy = 0.0 if max_warps == 0 else active_warps / max_warps
    return OccupancyEstimate(
        active_blocks=active_blocks,
        warps_per_block=warps_per_block,
        active_warps=active_warps,
        max_warps=max_warps,
        occupancy=occupancy,
        limiter=limiter,
    )
