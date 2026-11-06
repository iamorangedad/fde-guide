#pragma once
// 教学用 CUDA 错误检查。内核启动是异步的,只检查 cudaMalloc 的返回值不够。

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr)                                                         \
  do {                                                                           \
    cudaError_t _err = (expr);                                                   \
    if (_err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(_err));                                    \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

inline void cuda_check_launch(const char* file, int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA launch error %s:%d: %s\n", file, line,
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK_LAUNCH() cuda_check_launch(__FILE__, __LINE__)

// 在同一条流上计时。调用方必须先做预热并同步,避免把上下文初始化和懒加载算进结果。
struct CudaEventTiming {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CudaEventTiming() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~CudaEventTiming() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void record_start() { CUDA_CHECK(cudaEventRecord(start)); }
  void record_stop() {
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
  }
  float milliseconds() const {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

inline void print_device_limits() {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf(
      "device=%s cc=%d.%d SMs=%d warpSize=%d maxThreadsPerBlock=%d "
      "maxThreadsPerMultiProcessor=%d regsPerMultiprocessor=%d "
      "sharedMemPerBlock=%zu sharedMemPerMultiprocessor=%zu "
      "maxBlocksPerMultiProcessor=%d\n",
      prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.warpSize,
      prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor,
      prop.regsPerMultiprocessor, prop.sharedMemPerBlock,
      prop.sharedMemPerMultiprocessor, prop.maxBlocksPerMultiProcessor);
  std::printf(
      "以上限制来自本次 cudaGetDeviceProperties,随计算能力变化,不能写成所有 GPU 的常数。\n");
}
