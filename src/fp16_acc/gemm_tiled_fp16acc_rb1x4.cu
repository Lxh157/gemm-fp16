// src/gemm_tiled_fp16acc_rb1x4.cu
#include "utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// FP16 input + FP32 accumulate
// A: [M, K] half, B: [K, N] half, C: [M, N] float, row-major
//
// Block threads: (16, 16)
// Output tile per block: 16 rows x (16 * 4) cols = 16 x 64
// Each thread computes 1x4 outputs.

constexpr int TILE_M_FP16ACC_RB1X4 = 16;
constexpr int TILE_K_FP16ACC_RB1X4 = 16;
constexpr int RB_N_FP16ACC_RB1X4   = 4;

__global__ void gemm_tiled_fp16acc_rb1x4_kernel(const half* A, const half* B, float* C,
                                                int M, int N, int K) {
  __shared__ half As[TILE_M_FP16ACC_RB1X4][TILE_K_FP16ACC_RB1X4];
  __shared__ half Bs[TILE_K_FP16ACC_RB1X4][TILE_M_FP16ACC_RB1X4 * RB_N_FP16ACC_RB1X4];

  const int tx = threadIdx.x;  // [0, 15]
  const int ty = threadIdx.y;  // [0, 15]

  const int row = blockIdx.y * TILE_M_FP16ACC_RB1X4 + ty;
  const int col_base = blockIdx.x * (TILE_M_FP16ACC_RB1X4 * RB_N_FP16ACC_RB1X4) + tx * RB_N_FP16ACC_RB1X4;

  float acc[RB_N_FP16ACC_RB1X4] = {0.f, 0.f, 0.f, 0.f};

  const int num_k_tiles = (K + TILE_K_FP16ACC_RB1X4 - 1) / TILE_K_FP16ACC_RB1X4;

  for (int t = 0; t < num_k_tiles; ++t) {
    const int k_base = t * TILE_K_FP16ACC_RB1X4;

    // Load A tile: 16x16, each thread loads 1 element
    {
      const int a_col = k_base + tx;
      if (row < M && a_col < K) {
        As[ty][tx] = A[row * K + a_col];
      } else {
        As[ty][tx] = __float2half(0.0f);
      }
    }

    // Load B tile: 16x64, 256 threads, each thread loads 4 elements
    {
      const int tid = ty * blockDim.x + tx;  // 0..255
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int idx = tid + i * (TILE_M_FP16ACC_RB1X4 * TILE_M_FP16ACC_RB1X4); // +256
        const int br = idx / (TILE_M_FP16ACC_RB1X4 * RB_N_FP16ACC_RB1X4);         // /64
        const int bc = idx % (TILE_M_FP16ACC_RB1X4 * RB_N_FP16ACC_RB1X4);         // %64

        const int g_row = k_base + br;
        const int g_col = blockIdx.x * (TILE_M_FP16ACC_RB1X4 * RB_N_FP16ACC_RB1X4) + bc;

        if (g_row < K && g_col < N) {
          Bs[br][bc] = B[g_row * N + g_col];
        } else {
          Bs[br][bc] = __float2half(0.0f);
        }
      }
    }

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < TILE_K_FP16ACC_RB1X4; ++kk) {
      const float a_val = __half2float(As[ty][kk]);
      const int b_col0 = tx * RB_N_FP16ACC_RB1X4;

      #pragma unroll
      for (int j = 0; j < RB_N_FP16ACC_RB1X4; ++j) {
        const float b_val = __half2float(Bs[kk][b_col0 + j]);
        acc[j] = fmaf(a_val, b_val, acc[j]);
      }
    }

    __syncthreads();
  }

  if (row < M) {
    #pragma unroll
    for (int j = 0; j < RB_N_FP16ACC_RB1X4; ++j) {
      const int col = col_base + j;
      if (col < N) {
        C[row * N + col] = acc[j];
      }
    }
  }
}

void launch_gemm_tiled_fp16acc_rb1x4(const half* dA, const half* dB, float* dC,
                                     int M, int N, int K,
                                     cudaStream_t stream = nullptr) {
  dim3 block(TILE_M_FP16ACC_RB1X4, TILE_M_FP16ACC_RB1X4);  // (16,16)
  dim3 grid((N + TILE_M_FP16ACC_RB1X4 * RB_N_FP16ACC_RB1X4 - 1) / (TILE_M_FP16ACC_RB1X4 * RB_N_FP16ACC_RB1X4),
            (M + TILE_M_FP16ACC_RB1X4 - 1) / TILE_M_FP16ACC_RB1X4);

  gemm_tiled_fp16acc_rb1x4_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}