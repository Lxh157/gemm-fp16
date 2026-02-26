#include "utils.cuh"

#include <cuda_runtime.h>

// Tiled GEMM + Register Blocking (1x4 per thread)
// A: [M, K], B: [K, N], C: [M, N], row-major
//
// Block threads: (16, 16)
// Output tile per block: 16 rows x (16 * 4) cols = 16 x 64
//
// Each thread computes:
//   C[row, col_base + 0..3]

constexpr int TILE_M = 16;
constexpr int TILE_K = 16;
constexpr int RB_N   = 4;   // coarsening factor: register blocking in N dimension (1x4)

__global__ void gemm_tiled_rb1x4_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
  // Shared tiles:
  // A tile: [16 x 16]
  // B tile: [16 x 64]
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_M * RB_N];

  const int tx = threadIdx.x;  // [0, 15]
  const int ty = threadIdx.y;  // [0, 15]

  const int row = blockIdx.y * TILE_M + ty;
  const int col_base = blockIdx.x * (TILE_M * RB_N) + tx * RB_N;

  float acc[RB_N] = {0.f, 0.f, 0.f, 0.f};

  const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

  for (int t = 0; t < num_k_tiles; ++t) {
    const int k_base = t * TILE_K;

    // -----------------------------
    // Load A tile: 16x16 = 256 elems
    // Each thread loads 1 element
    // As[ty][tx] = A[row, k_base + tx]
    // -----------------------------
    {
      const int a_col = k_base + tx;
      if (row < M && a_col < K) {
        As[ty][tx] = A[row * K + a_col];
      } else {
        As[ty][tx] = 0.0f;
      }
    }

    // -----------------------------
    // Load B tile: 16x64 = 1024 elems
    // 256 threads -> each thread loads 4 elements
    //
    // Flatten thread id tid in [0,255]
    // Each thread loads indices: idx = tid + i*256, i=0..3
    // Map idx -> (br, bc) in [0..15] x [0..63]
    // Bs[br][bc] = B[k_base + br, block_col_base + bc]
    // -----------------------------
    {
      const int tid = ty * blockDim.x + tx;  // 0..255
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int idx = tid + i * (TILE_M * TILE_M);  // +256
        const int br = idx / (TILE_M * RB_N);         // /64 -> [0,15]
        const int bc = idx % (TILE_M * RB_N);         // %64 -> [0,63]

        const int g_row = k_base + br;                         // K dimension row in B
        const int g_col = blockIdx.x * (TILE_M * RB_N) + bc;   // N dimension col in B

        if (g_row < K && g_col < N) {
          Bs[br][bc] = B[g_row * N + g_col];
        } else {
          Bs[br][bc] = 0.0f;
        }
      }
    }

    __syncthreads();

    // -----------------------------
    // Compute: each thread accumulates 4 outputs
    // -----------------------------
    #pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      const float a_val = As[ty][kk];
      const int b_col0 = tx * RB_N;
      #pragma unroll
      for (int j = 0; j < RB_N; ++j) {
        acc[j] += a_val * Bs[kk][b_col0 + j];
      }
    }

    __syncthreads();
  }

  // Store results
  if (row < M) {
    #pragma unroll
    for (int j = 0; j < RB_N; ++j) {
      const int col = col_base + j;
      if (col < N) {
        C[row * N + col] = acc[j];
      }
    }
  }
}

void launch_gemm_tiled_rb1x4(const float* dA, const float* dB, float* dC,
                             int M, int N, int K,
                             cudaStream_t stream = nullptr) {
  dim3 block(TILE_M, TILE_M);  // (16,16)
  dim3 grid((N + TILE_M * RB_N - 1) / (TILE_M * RB_N),
            (M + TILE_M - 1) / TILE_M);

  gemm_tiled_rb1x4_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}