"""屋顶线模型的算术,不是实测性能。

Williams、Waterman、Patterson 的屋顶线把可达到性能写成:

    attainable = min(peak_flops, arithmetic_intensity * peak_bandwidth)

脊点 = peak_flops / peak_bandwidth,单位是 FLOP/byte。
算术强度低于脊点时,即使实现完美,模型也判为带宽顶限制;
高于脊点时,只说明带宽顶不低于算力顶,不证明内核真的打到算力顶。

GEMM 的"强制流量"算术强度假设每个 A、B、C 元素只进出内存一次
(完美复用后的下界流量,因而也是算术强度的上界)。额外的缓存未命中、
写分配和溢出只会降低算术强度。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bound:
    arithmetic_intensity: float
    ridge_flops_per_byte: float
    attainable_flops: float
    limited_by: str


def ridge_point(peak_flops: float, peak_bytes_per_s: float) -> float:
    if peak_flops <= 0 or peak_bytes_per_s <= 0:
        raise ValueError("峰值必须为正")
    return peak_flops / peak_bytes_per_s


def attainable(arithmetic_intensity: float, peak_flops: float, peak_bytes_per_s: float) -> Bound:
    if arithmetic_intensity < 0:
        raise ValueError("算术强度不能为负")
    ridge = ridge_point(peak_flops, peak_bytes_per_s)
    bandwidth_roof = arithmetic_intensity * peak_bytes_per_s
    if arithmetic_intensity < ridge:
        return Bound(arithmetic_intensity, ridge, bandwidth_roof, "bandwidth")
    return Bound(arithmetic_intensity, ridge, peak_flops, "compute")


def gemm_flops(m: int, n: int, k: int) -> int:
    if min(m, n, k) < 0:
        raise ValueError("维度不能为负")
    return 2 * m * n * k


def gemm_compulsory_bytes(m: int, n: int, k: int, bytes_per_elem: float) -> float:
    if bytes_per_elem <= 0:
        raise ValueError("bytes_per_elem 必须为正")
    elements = m * k + k * n + m * n
    return elements * bytes_per_elem


def gemm_compulsory_ai(m: int, n: int, k: int, bytes_per_elem: float) -> float:
    traffic = gemm_compulsory_bytes(m, n, k, bytes_per_elem)
    if traffic == 0:
        raise ValueError("空 GEMM 没有算术强度")
    return gemm_flops(m, n, k) / traffic


def square_gemm_compulsory_ai(n: int, bytes_per_elem: float) -> float:
    return gemm_compulsory_ai(n, n, n, bytes_per_elem)
