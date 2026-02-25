#include "utils.cuh"

#include <cuda_runtime.h>

// Row-major naive GEMM:
// A: [M, K], B: [K, N], C: [M, N]
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x; // j
  int row = blockIdx.y * blockDim.y + threadIdx.y; // i

  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
  }
}

// 提供给 main 调用的 launcher
void launch_gemm_naive(const float* dA, const float* dB, float* dC,
                       int M, int N, int K,
                       cudaStream_t stream = nullptr) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

  gemm_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}