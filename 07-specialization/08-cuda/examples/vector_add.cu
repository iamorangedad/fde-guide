// 向量加法:每个输出元素一次加法、两次读和一次写。
// 计时前先 warmup,再用 cudaEvent 记录。正确性见输出中的 correct。
// scalar 与 float4 的算术强度相同。float4 减少的是指令数,不改变屋顶线上界。
// float4 要求 n 是 4 的倍数,且 cudaMalloc 返回的指针对齐到 16 字节。

#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void add_scalar(const float* a, const float* b, float* c, int n) {
  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

__global__ void add_float4(const float4* a, const float4* b, float4* c, int n4) {
  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 xa = a[i];
    float4 xb = b[i];
    float4 y;
    y.x = xa.x + xb.x;
    y.y = xa.y + xb.y;
    y.z = xa.z + xb.z;
    y.w = xa.w + xb.w;
    c[i] = y;
  }
}

template <typename Launch>
float time_ms(Launch&& launch, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CudaEventTiming timing;
  timing.record_start();
  for (int i = 0; i < iters; ++i) launch();
  timing.record_stop();
  return timing.milliseconds() / static_cast<float>(iters);
}

int main(int argc, char** argv) {
  int n = 1 << 20;
  if (argc > 1) n = std::atoi(argv[1]);
  if (n <= 0 || (n % 4) != 0) {
    std::cerr << "n 必须是正的 4 的倍数\n";
    return 2;
  }

  CUDA_CHECK(cudaFree(0));
  print_device_limits();

  std::vector<float> ha(n), hb(n), hc(n), href(n);
  for (int i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i % 17) * 0.25f;
    hb[i] = static_cast<float>((i * 3) % 13) * 0.5f;
    href[i] = ha[i] + hb[i];
  }

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = std::min((n + threads - 1) / threads, 2048);
  auto launch_scalar = [&]() {
    add_scalar<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK_LAUNCH();
  };
  launch_scalar();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (hc[i] != href[i]) {
      std::cerr << "scalar mismatch at " << i << "\n";
      return 1;
    }
  }

  const int n4 = n / 4;
  const int blocks4 = std::min((n4 + threads - 1) / threads, 2048);
  auto launch_vec = [&]() {
    add_float4<<<blocks4, threads>>>(reinterpret_cast<const float4*>(da),
                                     reinterpret_cast<const float4*>(db),
                                     reinterpret_cast<float4*>(dc), n4);
    CUDA_CHECK_LAUNCH();
  };
  launch_vec();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    if (hc[i] != href[i]) {
      std::cerr << "float4 mismatch at " << i << "\n";
      return 1;
    }
  }

  const float scalar_ms = time_ms(launch_scalar, /*warmup=*/5, /*iters=*/20);
  const float vector_ms = time_ms(launch_vec, 5, 20);
  std::cout << "correct=1 scalar_ms=" << scalar_ms << " float4_ms=" << vector_ms
            << "\n";
  std::cout << "计时只描述本次进程和当前设备,不是可引用的加速比。\n";

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  return 0;
}
