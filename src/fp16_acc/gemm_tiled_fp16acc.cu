// src/gemm_tiled_fp16acc.cu
#include "utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// FP16 input, FP32 accumulate, FP16->FP32 convert in registers.
// A: [M,K] half, B: [K,N] half, C: [M,N] float (row-major)
//
// Simple 16x16 tile, each thread computes 1 output element.
// Not using tensor cores, just HFMA in FP32 (conversion + fmaf).

constexpr int TILE = 16;

__global__ void gemm_tiled_fp16acc_kernel(const half* A, const half* B, float* C,
                                         int M, int N, int K) {
  __shared__ half As[TILE][TILE];
  __shared__ half Bs[TILE][TILE];

  int tx = threadIdx.x; // 0..15
  int ty = threadIdx.y; // 0..15

  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;

  int num_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < num_tiles; ++t) {
    int k_base = t * TILE;

    // Load A tile (half)
    int a_col = k_base + tx;
    if (row < M && a_col < K) As[ty][tx] = A[row * K + a_col];
    else As[ty][tx] = __float2half(0.0f);

    // Load B tile (half)
    int b_row = k_base + ty;
    if (b_row < K && col < N) Bs[ty][tx] = B[b_row * N + col];
    else Bs[ty][tx] = __float2half(0.0f);

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      float a = __half2float(As[ty][kk]);
      float b = __half2float(Bs[kk][tx]);
      acc = fmaf(a, b, acc);
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

void launch_gemm_tiled_fp16acc(const half* dA, const half* dB, float* dC,
                               int M, int N, int K, cudaStream_t stream = nullptr) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  gemm_tiled_fp16acc_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}