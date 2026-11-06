# 第 8 章 GPU 与 CUDA 编程

> [🏠 返回主页](../../README.md) · [返回本部分](../README.md)

本章是专项。目标是让你能读懂一个内核为什么慢,并写出可以核对正确性的 CUDA,而不是背一组与设备无关的"最佳 block size"。

## 先修

- C 或 C++ 的指针、数组和函数调用
- 矩阵乘法的三重循环
- 第 3 章不是前提;本章不依赖 Transformer

## 学习目标

- 用 grid、block、warp、thread 描述一次启动,并说明 SIMT 与 SIMD 的差别
- 判断一次 warp 访存会生成多少 32 字节事务,以及共享内存是否发生 bank 冲突
- 写出带错误检查、预热和事件计时的向量加法、归约和分块矩阵乘
- 用手算估计占用率,并用屋顶线区分"即使完美实现仍受带宽限制"和"有可能打到算力顶"

## 本章地图

1. [执行层次与 SIMT](01-execution-simt.md)
2. [内存层次与合并访问](02-memory-coalescing.md)
3. [同步、竞争和归约](03-sync-reduction.md)
4. [分块矩阵乘](04-tiled-matmul.md)
5. [占用率与屋顶线](05-occupancy-roofline.md)

## 代码

| 文件 | 作用 |
|---|---|
| [examples/vector_add.cu](examples/vector_add.cu) | 标量与 `float4` 向量加法,主机对照 |
| [examples/reduction.cu](examples/reduction.cu) | 发散归约、顺序寻址、warp 洗牌 |
| [examples/matmul_tiled.cu](examples/matmul_tiled.cu) | 朴素矩阵乘与带填充的分块矩阵乘 |
| [py/coalesce.py](py/coalesce.py) | 32 字节事务和 bank 冲突的 CPU 模型 |
| [py/occupancy.py](py/occupancy.py) | 占用率估计 |
| [py/roofline.py](py/roofline.py) | 屋顶线与 GEMM 强制流量 |
| [py/reference_kernels.py](py/reference_kernels.py) | 与 `.cu` 索引规则一致的参考实现 |

有 CUDA 工具链和 GPU 时:

```bash
cd 02-ai-llm/08-cuda/examples
make
./vector_add
./reduction
./matmul_tiled
```

程序会打印 `cudaGetDeviceProperties` 的上限,并用全 1 或小整数数据做精确比较。打印的毫秒数只描述那一次运行。编写本教程的环境没有 `nvidia-smi` 和 `nvcc`,这三个程序没有编译,也没有在 GPU 上运行。没有 GPU 时不要把"编译失败"理解成算法失败;先跑:

```bash
python3 02-ai-llm/08-cuda/tests/test_cuda_models.py
```

## 硬件相关的限制

线程块大小、每 SM 寄存器数、共享内存、最大驻留块数都写在计算能力表里,并且应在运行时用 `cudaGetDeviceProperties` 再读一次。本章的数字例子都标明了"假设这次查询得到"。把 A100 的上限抄到另一张卡上,占用率估计会错。

## 小结

GPU 用大量简单线程掩盖访存延迟。正确性来自索引和同步;性能来自合并访问、避免多余的全局流量,以及让算术强度落在屋顶线的哪一侧。这些都能量化,不需要先宣称一个加速比。

## 关键术语

SIMT、warp、合并访问、bank 冲突、占用率、屋顶线、算术强度

## 延伸阅读

- CUDA Programming Guide,编程模型 — https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
- 编写 SIMT 内核 — https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- 进阶内核编程 — https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
- 计算能力附录 — https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html

---

[🏠 返回主页](../../README.md)
