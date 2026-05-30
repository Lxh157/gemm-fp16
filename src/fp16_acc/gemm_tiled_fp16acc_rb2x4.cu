// src/gemm_tiled_fp16acc_rb2x4.cu
#include "utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// FP16 input + FP32 accumulate
// A: [M, K] half, B: [K, N] half, C: [M, N] float, row-major
//
// Block threads: (16, 16)
// Output tile per block: (16 * 2) rows x (16 * 4) cols = 32 x 64
// Each thread computes: 2 x 4 outputs

constexpr int TILE_M_FP16ACC_RB2X4 = 16;
constexpr int TILE_K_FP16ACC_RB2X4 = 16;
constexpr int RB_M_FP16ACC_RB2X4   = 2;
constexpr int RB_N_FP16ACC_RB2X4   = 4;

__global__ void gemm_tiled_fp16acc_rb2x4_kernel(const half* A, const half* B, float* C,
                                                int M, int N, int K) {
  __shared__ half As[TILE_M_FP16ACC_RB2X4 * RB_M_FP16ACC_RB2X4][TILE_K_FP16ACC_RB2X4]; // 32x16
  __shared__ half Bs[TILE_K_FP16ACC_RB2X4][TILE_M_FP16ACC_RB2X4 * RB_N_FP16ACC_RB2X4]; // 16x64

  const int tx = threadIdx.x;  // [0, 15]
  const int ty = threadIdx.y;  // [0, 15]

  const int row_base = blockIdx.y * (TILE_M_FP16ACC_RB2X4 * RB_M_FP16ACC_RB2X4) + ty * RB_M_FP16ACC_RB2X4;
  const int col_base = blockIdx.x * (TILE_M_FP16ACC_RB2X4 * RB_N_FP16ACC_RB2X4) + tx * RB_N_FP16ACC_RB2X4;

  float acc[RB_M_FP16ACC_RB2X4][RB_N_FP16ACC_RB2X4] = {
      {0.f, 0.f, 0.f, 0.f},
      {0.f, 0.f, 0.f, 0.f}
  };

  const int num_k_tiles = (K + TILE_K_FP16ACC_RB2X4 - 1) / TILE_K_FP16ACC_RB2X4;

  for (int t = 0; t < num_k_tiles; ++t) {
    const int k_base = t * TILE_K_FP16ACC_RB2X4;

    // -----------------------------
    // Load A tile: 32x16 = 512 elems
    // 256 threads -> each thread loads 2 elems
    // -----------------------------
    {
      #pragma unroll
      for (int i = 0; i < RB_M_FP16ACC_RB2X4; ++i) {
        const int a_row = row_base + i;
        const int a_col = k_base + tx;
        const int s_row = ty * RB_M_FP16ACC_RB2X4 + i;

        if (a_row < M && a_col < K) {
          As[s_row][tx] = A[a_row * K + a_col];
        } else {
          As[s_row][tx] = __float2half(0.0f);
        }
      }
    }

    // -----------------------------
    // Load B tile: 16x64 = 1024 elems
    // 256 threads -> each thread loads 4 elems
    // -----------------------------
    {
      const int tid = ty * blockDim.x + tx;  // 0..255
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int idx = tid + i * (TILE_M_FP16ACC_RB2X4 * TILE_M_FP16ACC_RB2X4); // +256
        const int br = idx / (TILE_M_FP16ACC_RB2X4 * RB_N_FP16ACC_RB2X4);         // /64 -> [0,15]
        const int bc = idx % (TILE_M_FP16ACC_RB2X4 * RB_N_FP16ACC_RB2X4);         // %64 -> [0,63]

        const int g_row = k_base + br;
        const int g_col = blockIdx.x * (TILE_M_FP16ACC_RB2X4 * RB_N_FP16ACC_RB2X4) + bc;

        if (g_row < K && g_col < N) {
          Bs[br][bc] = B[g_row * N + g_col];
        } else {
          Bs[br][bc] = __float2half(0.0f);
        }
      }
    }

    __syncthreads();

    // -----------------------------
    // Compute: each thread accumulates 2x4 outputs
    // -----------------------------
    #pragma unroll
    for (int kk = 0; kk < TILE_K_FP16ACC_RB2X4; ++kk) {
      float a_frag[RB_M_FP16ACC_RB2X4];
      #pragma unroll
      for (int i = 0; i < RB_M_FP16ACC_RB2X4; ++i) {
        a_frag[i] = __half2float(As[ty * RB_M_FP16ACC_RB2X4 + i][kk]);
      }

      const int b_col0 = tx * RB_N_FP16ACC_RB2X4;
      float b_frag[RB_N_FP16ACC_RB2X4];
      #pragma unroll
      for (int j = 0; j < RB_N_FP16ACC_RB2X4; ++j) {
        b_frag[j] = __half2float(Bs[kk][b_col0 + j]);
      }

      #pragma unroll
      for (int i = 0; i < RB_M_FP16ACC_RB2X4; ++i) {
        #pragma unroll
        for (int j = 0; j < RB_N_FP16ACC_RB2X4; ++j) {
          acc[i][j] = fmaf(a_frag[i], b_frag[j], acc[i][j]);
        }
      }
    }

    __syncthreads();
  }

  // Store
  #pragma unroll
  for (int i = 0; i < RB_M_FP16ACC_RB2X4; ++i) {
    const int row = row_base + i;
    if (row < M) {
      #pragma unroll
      for (int j = 0; j < RB_N_FP16ACC_RB2X4; ++j) {
        const int col = col_base + j;
        if (col < N) {
          C[row * N + col] = acc[i][j];
        }
      }
    }
  }
}

void launch_gemm_tiled_fp16acc_rb2x4(const half* dA, const half* dB, float* dC,
                                     int M, int N, int K,
                                     cudaStream_t stream = nullptr) {
  dim3 block(TILE_M_FP16ACC_RB2X4, TILE_M_FP16ACC_RB2X4);  // (16,16)
  dim3 grid((N + TILE_M_FP16ACC_RB2X4 * RB_N_FP16ACC_RB2X4 - 1) / (TILE_M_FP16ACC_RB2X4 * RB_N_FP16ACC_RB2X4),
            (M + TILE_M_FP16ACC_RB2X4 * RB_M_FP16ACC_RB2X4 - 1) / (TILE_M_FP16ACC_RB2X4 * RB_M_FP16ACC_RB2X4));

  gemm_tiled_fp16acc_rb2x4_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}