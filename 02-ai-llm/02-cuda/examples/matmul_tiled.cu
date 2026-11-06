// C = A @ B,行主序。朴素核每个输出直接扫 K;分块核把 A、B 的 tile 放进共享内存。
// 计时前先 warmup,再用 cudaEvent 记录。正确性见输出中的 correct。
// 共享内存列填充为 TILE+1,避免按列读取时 32 个 bank 冲突。见第 8 章。

#include "common.cuh"

#include <cmath>
#include <iostream>
#include <vector>

constexpr int TILE = 16;

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N,
                             int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N,
                             int K) {
  __shared__ float As[TILE][TILE + 1];
  __shared__ float Bs[TILE][TILE + 1];
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.f;
  const int tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < tiles; ++t) {
    const int a_col = t * TILE + threadIdx.x;
    const int b_row = t * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
    Bs[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

void check_close(const std::vector<float>& got, const std::vector<float>& ref,
                 const char* name) {
  for (size_t i = 0; i < ref.size(); ++i) {
    if (got[i] != ref[i]) {
      std::cerr << name << " mismatch at " << i << " got=" << got[i]
                << " ref=" << ref[i] << "\n";
      std::exit(1);
    }
  }
}

void run_case(int M, int N, int K, bool time_it) {
  std::vector<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N),
      C(static_cast<size_t>(M) * N), ref(static_cast<size_t>(M) * N);
  for (int r = 0; r < M; ++r) {
    for (int k = 0; k < K; ++k) A[r * K + k] = static_cast<float>((r + k) % 5);
  }
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < N; ++c) B[k * N + c] = static_cast<float>((k + 2 * c) % 4);
  }
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[r * K + k] * B[k * N + c];
      ref[r * N + c] = acc;
    }
  }

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, C.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
  check_close(C, ref, "naive");

  matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(float), cudaMemcpyDeviceToHost));
  check_close(C, ref, "tiled");

  if (time_it) {
    for (int i = 0; i < 3; ++i) {
      matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
      matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CudaEventTiming timing;
    timing.record_start();
    for (int i = 0; i < 10; ++i) matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    timing.record_stop();
    CUDA_CHECK_LAUNCH();
    const float naive_ms = timing.milliseconds() / 10.f;
    timing.record_start();
    for (int i = 0; i < 10; ++i) matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    timing.record_stop();
    CUDA_CHECK_LAUNCH();
    const float tiled_ms = timing.milliseconds() / 10.f;
    std::cout << "time_MNK=" << M << "x" << N << "x" << K << " naive_ms=" << naive_ms
              << " tiled_ms=" << tiled_ms
              << " (本次设备计时,不是加速比结论)\n";
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  std::cout << "correct=1 M=" << M << " N=" << N << " K=" << K << "\n";
}

int main() {
  CUDA_CHECK(cudaFree(0));
  print_device_limits();
  run_case(18, 20, 19, false);
  run_case(64, 64, 64, true);
  return 0;
}
