"""KV cache 的字节推导。推导结果是容量账,不是吞吐测量。

每一层、每一条序列、每一个 token,要保存 K 和 V。
分组查询注意力(GQA)里 K/V 的头数是 num_kv_heads,可以小于查询头数。
多头注意力是 num_kv_heads == num_heads 的特例;多查询注意力是 num_kv_heads == 1。

    bytes = layers * 2 * batch * num_kv_heads * seq_len * head_dim * bytes_per_elem
    bytes_per_token(对整个 batch) = layers * 2 * batch * num_kv_heads * head_dim * bytes_per_elem

分页 KV 不改变一条序列的逻辑字节数,它改变的是这些字节能否不连续存放、
末块浪费和前缀能否共享。逻辑字节数仍按上式计算;物理浪费另计。
"""

from __future__ import annotations


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    bytes_per_elem: int = 2,
) -> dict[str, int]:
    for name, value in {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "batch": batch,
        "bytes_per_elem": bytes_per_elem,
    }.items():
        if value < 0:
            raise ValueError(f"{name} 不能为负")
    per_token = num_layers * 2 * num_kv_heads * head_dim * bytes_per_elem
    total = per_token * seq_len * batch
    return {
        "bytes_per_token_per_sequence": per_token,
        "total_bytes": total,
        "batch": batch,
        "seq_len": seq_len,
    }


def paged_blocks(seq_len: int, block_size: int) -> int:
    if seq_len < 0 or block_size <= 0:
        raise ValueError("seq_len 不能为负,block_size 必须为正")
    if seq_len == 0:
        return 0
    return (seq_len + block_size - 1) // block_size


def paged_waste_fraction(seq_len: int, block_size: int) -> float:
    """只计最后一块里没用到的 token 槽,不含分配器碎片。"""
    blocks = paged_blocks(seq_len, block_size)
    if blocks == 0:
        return 0.0
    used = seq_len
    slots = blocks * block_size
    return (slots - used) / slots
