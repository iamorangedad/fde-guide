// 块内归约的三种写法,结果应一致,差别在发散、同步和洗牌。
// 计时前先 warmup,再用 cudaEvent 记录。正确性见输出中的 correct。
// 多块结果用同一内核反复归约到标量。全 1 输入的和在 float32 精确范围内可精确比较。

#include "common.cuh"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

__global__ void reduce_naive(const float* in, float* out, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  s[tid] = (gid < n) ? in[gid] : 0.f;
  __syncthreads();
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0 && tid + stride < blockDim.x) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

__global__ void reduce_sequential(const float* in, float* out, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  s[tid] = (gid < n) ? in[gid] : 0.f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

__device__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__global__ void reduce_warp(const float* in, float* out, int n) {
  __shared__ float warp_partial[32];
  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + tid;
  float v = (gid < n) ? in[gid] : 0.f;
  v = warp_sum(v);
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  if (lane == 0) warp_partial[warp_id] = v;
  __syncthreads();
  const int nwarps = blockDim.x >> 5;
  float w = (tid < nwarps) ? warp_partial[tid] : 0.f;
  if (warp_id == 0) w = warp_sum(w);
  if (tid == 0) out[blockIdx.x] = w;
}

enum class Kind { Naive, Sequential, Warp };

float reduce_all(const float* d_input, int n, Kind kind) {
  const int threads = 256;
  float* d_cur = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cur, static_cast<size_t>(n) * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_cur, d_input, static_cast<size_t>(n) * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  int cur_n = n;
  while (cur_n > 1) {
    const int blocks = (cur_n + threads - 1) / threads;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(blocks) * sizeof(float)));
    if (kind == Kind::Naive) {
      reduce_naive<<<blocks, threads, threads * sizeof(float)>>>(d_cur, d_out, cur_n);
    } else if (kind == Kind::Sequential) {
      reduce_sequential<<<blocks, threads, threads * sizeof(float)>>>(d_cur, d_out,
                                                                      cur_n);
    } else {
      reduce_warp<<<blocks, threads>>>(d_cur, d_out, cur_n);
    }
    CUDA_CHECK_LAUNCH();
    CUDA_CHECK(cudaFree(d_cur));
    d_cur = d_out;
    cur_n = blocks;
  }
  float host = 0.f;
  CUDA_CHECK(cudaMemcpy(&host, d_cur, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_cur));
  return host;
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

int main() {
  CUDA_CHECK(cudaFree(0));
  print_device_limits();

  const int n = 256 * 100;
  std::vector<float> h(n, 1.f);
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));

  const float expected = static_cast<float>(n);
  const float naive = reduce_all(d, n, Kind::Naive);
  const float sequential = reduce_all(d, n, Kind::Sequential);
  const float warp = reduce_all(d, n, Kind::Warp);
  if (naive != expected || sequential != expected || warp != expected) {
    std::cerr << "mismatch naive=" << naive << " sequential=" << sequential
              << " warp=" << warp << " expected=" << expected << "\n";
    return 1;
  }

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  float* partial = nullptr;
  CUDA_CHECK(cudaMalloc(&partial, static_cast<size_t>(blocks) * sizeof(float)));
  auto launch_naive = [&]() {
    reduce_naive<<<blocks, threads, threads * sizeof(float)>>>(d, partial, n);
    CUDA_CHECK_LAUNCH();
  };
  auto launch_warp = [&]() {
    reduce_warp<<<blocks, threads>>>(d, partial, n);
    CUDA_CHECK_LAUNCH();
  };
  const float naive_ms = time_ms(launch_naive, 5, 20);
  const float warp_ms = time_ms(launch_warp, 5, 20);
  std::cout << "correct=1 n=" << n << " naive_partial_ms=" << naive_ms
            << " warp_partial_ms=" << warp_ms << "\n";
  std::cout << "计时只覆盖单次部分归约内核,且只属于本次设备,不是加速比结论。\n";
  CUDA_CHECK(cudaFree(partial));
  CUDA_CHECK(cudaFree(d));
  return 0;
}
