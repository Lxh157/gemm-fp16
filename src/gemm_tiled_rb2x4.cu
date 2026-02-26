#include "utils.cuh"

#include <cuda_runtime.h>

// Tiled GEMM + Register Blocking (2x4 per thread)
// A: [M, K], B: [K, N], C: [M, N], row-major
//
// Block threads: (16, 16)  -> 256 threads
// Output tile per block: (16*2) rows x (16*4) cols = 32 x 64
//
// Each thread computes:
//   C[row0, col_base + 0..3]  and  C[row1, col_base + 0..3]
// where row0 = blockRow + ty, row1 = row0 + 16

constexpr int TILE_M = 16;
constexpr int TILE_K = 16;
constexpr int RB_N   = 4;   // register blocking in N dimension (..x4)
constexpr int RB_M   = 2;   // coarsening in M dimension (2 rows per thread)

__global__ void gemm_tiled_rb2x4_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
  // Shared tiles:
  // A tile: [32 x 16]  (two 16x16 blocks stacked in M)
  // B tile: [16 x 64]
  __shared__ float As[TILE_M * RB_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_M * RB_N];

  const int tx = threadIdx.x;  // [0, 15]
  const int ty = threadIdx.y;  // [0, 15]

  const int blockRow = blockIdx.y * (TILE_M * RB_M);  // step 32
  const int blockCol = blockIdx.x * (TILE_M * RB_N);  // step 64

  const int row0 = blockRow + ty;
  const int row1 = row0 + TILE_M;

  const int col_base = blockCol + tx * RB_N;

  float acc0[RB_N] = {0.f, 0.f, 0.f, 0.f};
  float acc1[RB_N] = {0.f, 0.f, 0.f, 0.f};

  const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

  for (int t = 0; t < num_k_tiles; ++t) {
    const int k_base = t * TILE_K;

    // -----------------------------
    // Load A tile: 32x16 = 512 elems
    // 256 threads -> each thread loads 2 elements
    //
    // We flatten tid in [0,255], then load:
    //   idx0 = tid
    //   idx1 = tid + 256
    // Map idx -> (ar, ac) in [0..31] x [0..15]
    // As[ar][ac] = A[blockRow + ar, k_base + ac]
    // -----------------------------
    {
      const int tid = ty * blockDim.x + tx;  // 0..255

      // first 256 elements
      {
        const int idx = tid;
        const int ar = idx / TILE_K; // /16 -> [0,15]
        const int ac = idx % TILE_K; // %16 -> [0,15]
        const int g_row = blockRow + ar;
        const int g_col = k_base + ac;
        if (g_row < M && g_col < K) As[ar][ac] = A[g_row * K + g_col];
        else                        As[ar][ac] = 0.0f;
      }

      // second 256 elements (rows 16..31)
      {
        const int idx = tid + TILE_M * TILE_M; // +256
        const int ar = idx / TILE_K; // /16 -> [16,31]
        const int ac = idx % TILE_K; // %16 -> [0,15]
        const int g_row = blockRow + ar;
        const int g_col = k_base + ac;
        if (g_row < M && g_col < K) As[ar][ac] = A[g_row * K + g_col];
        else                        As[ar][ac] = 0.0f;
      }
    }

    // -----------------------------
    // Load B tile: 16x64 = 1024 elems
    // 256 threads -> each thread loads 4 elements (same as rb1x4)
    //
    // idx = tid + i*256, i=0..3
    // Map idx -> (br, bc) in [0..15] x [0..63]
    // Bs[br][bc] = B[k_base + br, blockCol + bc]
    // -----------------------------
    {
      const int tid = ty * blockDim.x + tx;  // 0..255
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int idx = tid + i * (TILE_M * TILE_M);  // +256
        const int br = idx / (TILE_M * RB_N);         // /64 -> [0,15]
        const int bc = idx % (TILE_M * RB_N);         // %64 -> [0,63]

        const int g_row = k_base + br;
        const int g_col = blockCol + bc;

        if (g_row < K && g_col < N) Bs[br][bc] = B[g_row * N + g_col];
        else                        Bs[br][bc] = 0.0f;
      }
    }

    __syncthreads();

    // -----------------------------
    // Compute: each thread accumulates 2x4 outputs
    // -----------------------------
    #pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      const float a0 = As[ty][kk];                 // row0 is As[ty][:]
      const float a1 = As[ty + TILE_M][kk];        // row1 is As[ty+16][:]

      const int b_col0 = tx * RB_N;
      #pragma unroll
      for (int j = 0; j < RB_N; ++j) {
        const float b = Bs[kk][b_col0 + j];
        acc0[j] += a0 * b;
        acc1[j] += a1 * b;
      }
    }

    __syncthreads();
  }

  // Store results
  if (row0 < M) {
    #pragma unroll
    for (int j = 0; j < RB_N; ++j) {
      const int col = col_base + j;
      if (col < N) C[row0 * N + col] = acc0[j];
    }
  }
  if (row1 < M) {
    #pragma unroll
    for (int j = 0; j < RB_N; ++j) {
      const int col = col_base + j;
      if (col < N) C[row1 * N + col] = acc1[j];
    }
  }
}

void launch_gemm_tiled_rb2x4(const float* dA, const float* dB, float* dC,
                             int M, int N, int K,
                             cudaStream_t stream = nullptr) {
  dim3 block(TILE_M, TILE_M);  // (16,16)
  dim3 grid((N + TILE_M * RB_N - 1) / (TILE_M * RB_N),
            (M + TILE_M * RB_M - 1) / (TILE_M * RB_M));

  gemm_tiled_rb2x4_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
}