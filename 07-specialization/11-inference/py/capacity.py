"""服务容量与费用的输入输出关系。价格和时延都是参数,不是报价或测量。"""

from __future__ import annotations


def weight_bytes(num_params: int, bytes_per_elem: float) -> float:
    if num_params < 0 or bytes_per_elem <= 0:
        raise ValueError("参数非法")
    return num_params * bytes_per_elem


def remaining_bytes(gpu_bytes: int, weight: float, reserved_bytes: float) -> float:
    if gpu_bytes < 0 or weight < 0 or reserved_bytes < 0:
        raise ValueError("字节数不能为负")
    remaining = gpu_bytes - weight - reserved_bytes
    if remaining < 0:
        raise ValueError("权值与预留已经超过 GPU 容量")
    return remaining


def max_cached_tokens(kv_budget_bytes: float, bytes_per_token: float) -> int:
    if kv_budget_bytes < 0 or bytes_per_token <= 0:
        raise ValueError("参数非法")
    return int(kv_budget_bytes // bytes_per_token)


def cost_from_inputs(num_gpus: int, wall_seconds: float, price_per_gpu_hour: float) -> dict[str, float]:
    """price_per_gpu_hour 使用调用方选定的货币单位。不要把示例里的 1 当成云厂商价格。"""
    if num_gpus < 0 or wall_seconds < 0 or price_per_gpu_hour < 0:
        raise ValueError("参数不能为负")
    gpu_hours = num_gpus * wall_seconds / 3600.0
    return {"gpu_hours": gpu_hours, "cost": gpu_hours * price_per_gpu_hour}
