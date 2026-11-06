"""均匀整数量化的教学实现,用来说明误差和存储账,不是 NF4,也不是 bitsandbytes。

QLoRA 论文中的 NormalFloat4 按正态分布的分位数取码字,并且还有二次量化。
这里的均匀 4 bit 网格只回答两个可以在 CPU 上验证的问题:
1. 量化再反量化的误差不超过半个量化步长(端点落在网格上时端点误差为 0);
2. 打包后的字节数如何随参数量和分组大小变化。
不能用这个玩具的误差去预测某个真实模型的任务质量。
"""

from __future__ import annotations

import math


def quantize_uniform(values: list[float], bits: int = 4, group_size: int = 8) -> dict:
    if bits <= 0 or bits > 8:
        raise ValueError("教学实现只支持 1 到 8 bit")
    if group_size <= 0:
        raise ValueError("group_size 必须为正")
    levels = (1 << bits) - 1
    codes: list[int] = []
    scales: list[float] = []
    mins: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        lo = min(group)
        hi = max(group)
        scale = 0.0 if hi == lo else (hi - lo) / levels
        mins.append(lo)
        scales.append(scale)
        for value in group:
            if scale == 0:
                codes.append(0)
            else:
                q = int(round((value - lo) / scale))
                codes.append(max(0, min(levels, q)))
    return {"codes": codes, "scales": scales, "mins": mins, "bits": bits, "group_size": group_size, "levels": levels}


def dequantize_uniform(packed: dict) -> list[float]:
    out = []
    group_size = packed["group_size"]
    for index, code in enumerate(packed["codes"]):
        group = index // group_size
        out.append(packed["mins"][group] + code * packed["scales"][group])
    return out


def max_abs_error(original: list[float], packed: dict) -> float:
    restored = dequantize_uniform(packed)
    return max(abs(a - b) for a, b in zip(original, restored))


def theoretical_max_error(values: list[float], bits: int = 4, group_size: int = 8) -> float:
    """均匀量化四舍五入的误差上界是半个步长。常数分组的上界是 0。"""
    levels = (1 << bits) - 1
    bound = 0.0
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        span = max(group) - min(group)
        if span == 0 or levels == 0:
            continue
        bound = max(bound, 0.5 * span / levels)
    return bound


def storage_bytes(num_params: int, bits: int = 4, group_size: int = 64, metadata: str = "fp16_scale_and_zero") -> dict:
    """打包权重字节 + 分组元数据。metadata 方案是教学假设,要在报告里写明。"""
    if num_params < 0 or group_size <= 0 or bits <= 0:
        raise ValueError("参数非法")
    data_bytes = math.ceil(num_params * bits / 8)
    groups = math.ceil(num_params / group_size) if num_params else 0
    if metadata == "fp16_scale_and_zero":
        meta = groups * 4  # 一个 fp16 scale 和一个 fp16 zero/min
    elif metadata == "fp32_scale_and_min":
        meta = groups * 8
    else:
        raise ValueError("未知 metadata 方案")
    fp16 = num_params * 2
    return {
        "data_bytes": data_bytes,
        "metadata_bytes": meta,
        "total_bytes": data_bytes + meta,
        "fp16_bytes": fp16,
        "groups": groups,
    }
