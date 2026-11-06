"""训练状态与激活的字节账。这是恒等式,不是 profiler 读数。

全精度 Adam:
    参数 fp32 + 梯度 fp32 + 一阶矩 fp32 + 二阶矩 fp32 = 16P 字节。

ZeRO 论文所述的混合精度 Adam 静态状态也是 16P 字节,组成不同:
    fp16/bf16 参数 2P + fp16/bf16 梯度 2P + fp32 主权重 4P
    + fp32 动量 4P + fp32 方差 4P。
混合精度的主要内存收益在激活和带宽,而不是把 Adam 状态从 16P 变成更小,
除非再换成 8-bit 优化器或做分片。激活项依赖实现(是否物化注意力分数、
是否做激活检查点),所以单独列出,并要求用设备上的分配峰值核对。
"""

from __future__ import annotations


def adam_fp32_bytes(num_params: int) -> dict[str, int]:
    _check_params(num_params)
    four = 4 * num_params
    return {
        "parameters": four,
        "gradients": four,
        "momentum": four,
        "variance": four,
        "total": 16 * num_params,
    }


def adam_mixed_precision_bytes(num_params: int) -> dict[str, int]:
    """ZeRO 论文中的混合精度静态账,不含激活、碎片和分配器保留。"""
    _check_params(num_params)
    return {
        "low_precision_parameters": 2 * num_params,
        "low_precision_gradients": 2 * num_params,
        "fp32_master_parameters": 4 * num_params,
        "fp32_momentum": 4 * num_params,
        "fp32_variance": 4 * num_params,
        "total": 16 * num_params,
    }


def lora_adam_bytes(trainable_params: int, optimizer: str = "adam_fp32") -> dict[str, int]:
    """只对可训练适配器参数计优化器状态。冻结基座另计推理权重。"""
    if optimizer == "adam_fp32":
        state = adam_fp32_bytes(trainable_params)
    elif optimizer == "adam_mixed":
        state = adam_mixed_precision_bytes(trainable_params)
    else:
        raise ValueError("optimizer 只能是 adam_fp32 或 adam_mixed")
    state["trainable_params"] = trainable_params
    return state


def activation_bytes(
    batch: int,
    seq: int,
    hidden: int,
    layers: int,
    bytes_per_elem: int,
    saved_tensors_per_layer: int,
    materialized_scores: bool = False,
    heads: int | None = None,
) -> dict[str, int]:
    """教学用下界/上界开关。

    saved_tensors_per_layer 是假设每层保留多少份 batch*seq*hidden 张量,
    必须和具体实现一起报告。materialized_scores 为真时加上
    B * heads * S * S * bytes 的注意力分数;FlashAttention 一类实现
    不物化这项,此时应传 False。
    """
    for name, value in {
        "batch": batch,
        "seq": seq,
        "hidden": hidden,
        "layers": layers,
        "bytes_per_elem": bytes_per_elem,
        "saved_tensors_per_layer": saved_tensors_per_layer,
    }.items():
        if value < 0:
            raise ValueError(f"{name} 不能为负")
    hidden_like = layers * saved_tensors_per_layer * batch * seq * hidden * bytes_per_elem
    scores = 0
    if materialized_scores:
        if heads is None or heads < 0:
            raise ValueError("物化注意力分数时必须给出 heads")
        scores = layers * batch * heads * seq * seq * bytes_per_elem
    return {"hidden_like": hidden_like, "scores": scores, "total": hidden_like + scores}


def checkpoint_boundary_bytes(
    batch: int, seq: int, hidden: int, segments: int, bytes_per_elem: int
) -> int:
    """假设每个检查点段只保留边界隐状态,段内前向在反向时重算。

    这是 Chen 等人子线性显存技巧的记账形式,不是某次运行少掉的字节数。
    """
    if min(batch, seq, hidden, segments, bytes_per_elem) < 0:
        raise ValueError("参数不能为负")
    return segments * batch * seq * hidden * bytes_per_elem


def gib(num_bytes: float) -> float:
    return num_bytes / (1024**3)


def _check_params(num_params: int) -> None:
    if num_params < 0:
        raise ValueError("num_params 不能为负")
