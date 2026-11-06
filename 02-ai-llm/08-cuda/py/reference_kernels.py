"""与 examples/*.cu 同一索引规则的 CPU 参考实现。

这些函数用来锁住算法,不用来声称 GPU 速度。浮点归约按并行步模拟:
同一步内先读旧值再写,因为 CUDA 里一个 stride 内的线程并发执行。
"""

from __future__ import annotations


def add_scalar(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("长度必须相同")
    return [x + y for x, y in zip(a, b)]


def add_float4(a: list[float], b: list[float]) -> list[float]:
    if len(a) % 4 != 0:
        raise ValueError("float4 路径要求长度为 4 的倍数")
    return add_scalar(a, b)


def reduce_naive_block(values: list[float]) -> float:
    s = list(values)
    n = len(s)
    stride = 1
    while stride < n:
        updated = list(s)
        for tid in range(n):
            if tid % (2 * stride) == 0 and tid + stride < n:
                updated[tid] = s[tid] + s[tid + stride]
        s = updated
        stride *= 2
    return s[0]


def reduce_sequential_block(values: list[float]) -> float:
    s = list(values)
    n = len(s)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("顺序寻址参考实现要求长度为 2 的幂,与教学内核的块大小一致")
    stride = n // 2
    while stride > 0:
        updated = list(s)
        for tid in range(stride):
            updated[tid] = s[tid] + s[tid + stride]
        s = updated
        stride //= 2
    return s[0]


def _warp_sum(values: list[float]) -> list[float]:
    s = list(values)
    offset = 16
    while offset > 0:
        updated = list(s)
        for lane in range(32):
            if lane + offset < 32:
                updated[lane] = s[lane] + s[lane + offset]
        s = updated
        offset //= 2
    return s


def reduce_warp_block(values: list[float]) -> float:
    """对应 reduce_warp:块大小为 32 的倍数且不超过 1024。"""
    n = len(values)
    if n == 0 or n % 32 != 0 or n > 1024:
        raise ValueError("warp 归约参考实现要求块大小为 32 的倍数且 <= 1024")
    warp_partial = []
    for warp in range(n // 32):
        chunk = values[warp * 32 : (warp + 1) * 32]
        warp_partial.append(_warp_sum(chunk)[0])
    padded = warp_partial + [0.0] * (32 - len(warp_partial))
    return _warp_sum(padded)[0]


def matmul_reference(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m = len(a)
    k = len(a[0]) if m else 0
    if len(b) != k:
        raise ValueError("内维不匹配")
    n = len(b[0]) if k else 0
    c = [[0.0 for _ in range(n)] for _ in range(m)]
    for row in range(m):
        for col in range(n):
            acc = 0.0
            for kk in range(k):
                acc += a[row][kk] * b[kk][col]
            c[row][col] = acc
    return c


def matmul_tiled_sim(
    a: list[list[float]], b: list[list[float]], tile: int = 16
) -> list[list[float]]:
    """按 matmul_tiled 的线程索引模拟,包含越界补零。"""
    if tile <= 0:
        raise ValueError("tile 必须为正")
    m = len(a)
    k = len(a[0]) if m else 0
    if len(b) != k:
        raise ValueError("内维不匹配")
    n = len(b[0]) if k else 0
    c = [[0.0 for _ in range(n)] for _ in range(m)]
    grid_y = (m + tile - 1) // tile
    grid_x = (n + tile - 1) // tile
    tiles = (k + tile - 1) // tile
    for by in range(grid_y):
        for bx in range(grid_x):
            acc = [[0.0 for _ in range(tile)] for _ in range(tile)]
            for t in range(tiles):
                tile_a = [[0.0 for _ in range(tile)] for _ in range(tile)]
                tile_b = [[0.0 for _ in range(tile)] for _ in range(tile)]
                for ty in range(tile):
                    for tx in range(tile):
                        row = by * tile + ty
                        col = bx * tile + tx
                        a_col = t * tile + tx
                        b_row = t * tile + ty
                        if row < m and a_col < k:
                            tile_a[ty][tx] = a[row][a_col]
                        if b_row < k and col < n:
                            tile_b[ty][tx] = b[b_row][col]
                for ty in range(tile):
                    for tx in range(tile):
                        for kk in range(tile):
                            acc[ty][tx] += tile_a[ty][kk] * tile_b[kk][tx]
            for ty in range(tile):
                for tx in range(tile):
                    row = by * tile + ty
                    col = bx * tile + tx
                    if row < m and col < n:
                        c[row][col] = acc[ty][tx]
    return c
