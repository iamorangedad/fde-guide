"""填充浪费、首次适应装箱,以及打包序列的块状因果掩码。

打包后位置 i 只能看见同一段且下标不超过 i 的 token。段与段之间必须为假。
位置编码要在段内重新从 0 计;本模块只负责掩码和区间,调用方必须单独重置 position ids。
"""

from __future__ import annotations


def padding_efficiency(lengths: list[int]) -> float:
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("lengths 必须是正整数")
    slots = len(lengths) * max(lengths)
    return sum(lengths) / slots


def padding_waste_fraction(lengths: list[int]) -> float:
    return 1.0 - padding_efficiency(lengths)


def pack_first_fit(sequences: list[list[int]], max_len: int) -> list[dict]:
    """按长度降序做首次适应。返回的段用原序列下标。"""
    if max_len <= 0:
        raise ValueError("max_len 必须为正")
    bins: list[dict] = []
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True)
    for index in order:
        seq = sequences[index]
        if len(seq) > max_len:
            raise ValueError(f"序列 {index} 长于 max_len")
        placed = False
        for bucket in bins:
            if bucket["length"] + len(seq) <= max_len:
                start = bucket["length"]
                bucket["segments"].append(
                    {"index": index, "start": start, "end": start + len(seq)}
                )
                bucket["length"] += len(seq)
                placed = True
                break
        if not placed:
            bins.append(
                {
                    "length": len(seq),
                    "segments": [{"index": index, "start": 0, "end": len(seq)}],
                }
            )
    return bins


def block_causal_mask(length: int, segments: list[tuple[int, int]]) -> list[list[bool]]:
    """mask[i][j] 为真表示位置 i 可以注意位置 j。"""
    if length < 0:
        raise ValueError("length 不能为负")
    mask = [[False for _ in range(length)] for _ in range(length)]
    covered = [False] * length
    for start, end in segments:
        if not 0 <= start < end <= length:
            raise ValueError("段必须落在 [0, length) 且非空")
        for pos in range(start, end):
            if covered[pos]:
                raise ValueError("段重叠")
            covered[pos] = True
        for i in range(start, end):
            for j in range(start, i + 1):
                mask[i][j] = True
    return mask
