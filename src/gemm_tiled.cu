#include "utils.cuh"

#include <cuda_runtime.h>

// 简单 16x16 tiled GEMM（FP32）
// A: [M, K], B: [K, N], C: [M, N]，row-major
constexpr int TILE = 16;

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;

  // 沿 K 维分块
  int num_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < num_tiles; ++t) {
    int a_col = t * TILE + tx; // A[row, a_col]
    int b_row = t * TILE + ty; // B[b_row, col]

    // A tile load
    if (row < M && a_col < K) {
      As[ty][tx] = A[row * K + a_col];
    } else {
      As[ty][tx] = 0.0f;
    }

    // B tile load
    if (b_row < K && col < N) {
      Bs[ty][tx] = B[b_row * N + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

void launch_gemm_tiled(const float* dA, const float* dB, float* dC,
                       int M, int N, int K,
                       cudaStream_t stream = nullptr) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  gemm_tiled_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}