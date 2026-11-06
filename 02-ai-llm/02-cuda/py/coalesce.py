"""全局内存 32 字节事务与共享内存 bank 的教学模型。

模型对应 CUDA 编程指南对计算能力 6.0 及更新设备的描述:一个 warp 的并发访问
合并为覆盖这些字节所需的 32 字节事务数。这是事务计数,不是实测带宽。
共享内存按连续 32 位字映射到 32 个 bank。读同一地址视为广播;同一 bank 上的
不同地址会串行,冲突度取一个 bank 上不同地址数的最大值。
"""

from __future__ import annotations

from collections import Counter


def count_segments(addresses: list[int], word_bytes: int = 4, segment_bytes: int = 32) -> int:
    """返回覆盖所有 [addr, addr + word_bytes) 所需的对齐段数量。"""
    if word_bytes <= 0 or segment_bytes <= 0:
        raise ValueError("word_bytes 和 segment_bytes 必须为正")
    segments: set[int] = set()
    for addr in addresses:
        if addr < 0:
            raise ValueError("地址不能为负")
        last = addr + word_bytes - 1
        for seg in range(addr // segment_bytes, last // segment_bytes + 1):
            segments.add(seg)
    return len(segments)


def consecutive_float_addresses(base: int, lanes: int = 32, word_bytes: int = 4) -> list[int]:
    return [base + i * word_bytes for i in range(lanes)]


def shared_banks(word_indices: list[int], banks: int = 32) -> list[int]:
    if banks <= 0:
        raise ValueError("banks 必须为正")
    return [idx % banks for idx in word_indices]


def read_conflict_degree(word_indices: list[int], banks: int = 32) -> int:
    """读请求的 bank 冲突度。同一地址只算一次(广播)。"""
    if not word_indices:
        raise ValueError("word_indices 不能为空")
    unique_addresses = set(word_indices)
    counts = Counter(addr % banks for addr in unique_addresses)
    return max(counts.values())


def column_indices(column: int, rows: int, leading_dim: int) -> list[int]:
    """列主向访问:第 column 列、连续 rows 行,行距为 leading_dim。"""
    return [column + r * leading_dim for r in range(rows)]
