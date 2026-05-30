#include "utils.cuh"
#include <cuda_runtime.h>

// FP32 GEMM: explicit 1D thread-tile version
// A: [M, K], B: [K, N], C: [M, N], row-major
//
// block tile:
//   BM = 16, BK = 16, BN = 64
//
// block threads:
//   (16, 16) => 256 threads
//
// thread tile (1D):
//   each thread computes 1 x TN outputs
//   TN = 4
//
// mapping:
//   thread (ty, tx) owns:
//     C[row, col_base + 0..3]
//   where
//     row      = blockIdx.y * BM + ty
//     col_base = blockIdx.x * BN + tx * TN
//
// first version goal:
//   1. establish explicit thread-tile ownership
//   2. keep the mainloop simple and readable
//   3. make it easy to evolve into thread_tiled_2d later

constexpr int BM = 16;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TN = 4;   // 1D thread tile width in N dimension

__global__ void gemm_thread_tiled_1d_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K) {

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int tx = threadIdx.x;   // [0, 15]
    const int ty = threadIdx.y;   // [0, 15]

    const int row = blockIdx.y * BM + ty;
    const int col_base = blockIdx.x * BN + tx * TN;

    // thread-local accumulators: 1 x TN
    float acc[TN] = {0.f, 0.f, 0.f, 0.f};

    const int num_k_tiles = (K + BK - 1) / BK;  //tileK走的次数

    for (int t = 0; t < num_k_tiles; ++t) {
        const int k_base = t * BK;

        // ------------------------------------------------------------
        // load A tile: [BM, BK] = [16, 16] => 256 elems
        // each thread loads exactly 1 element:
        //   As[ty][tx] = A[row, k_base + tx]
        // ------------------------------------------------------------
        {
            const int a_col = k_base + tx;
            if (row < M && a_col < K) {
                As[ty][tx] = A[row * K + a_col];
            } else {
                As[ty][tx] = 0.0f;
            }
        }

        // ------------------------------------------------------------
        // load B tile: [BK, BN] = [16, 64] => 1024 elems
        // 256 threads => each thread loads 4 elements
        //
        // flatten tid in [0, 255]
        // idx = tid + i * 256, i = 0..3
        // map idx -> (br, bc) in [0..15] x [0..63]
        // Bs[br][bc] = B[k_base + br, block_col_base + bc]
        // ------------------------------------------------------------
        {
            const int tid = ty * blockDim.x + tx;  // 0..255

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = tid + i * (BM * BK);   // +256
                const int br  = idx / BN;              // [0, 15]
                const int bc  = idx % BN;              // [0, 63]

                const int g_row = k_base + br;
                const int g_col = blockIdx.x * BN + bc;

                if (g_row < K && g_col < N) {
                    Bs[br][bc] = B[g_row * N + g_col];
                } else {
                    Bs[br][bc] = 0.0f;
                }
            }
        }

        __syncthreads();

        // ------------------------------------------------------------
        // compute:
        // each thread computes a 1 x TN output strip
        //
        // for each kk:
        //   a_val = As[ty][kk]
        //   b_val[j] = Bs[kk][tx * TN + j]
        //   acc[j] += a_val * b_val[j]
        // ------------------------------------------------------------
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float a_val = As[ty][kk];
            const int b_col0 = tx * TN;

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                acc[j] += a_val * Bs[kk][b_col0 + j];
            }
        }

        __syncthreads();
    }

    // store results
    if (row < M) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = col_base + j;
            if (col < N) {
                C[row * N + col] = acc[j];
            }
        }
    }
}

void launch_gemm_thread_tiled_1d(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr) {

    dim3 block(BM, BM);  // (16, 16)
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    gemm_thread_tiled_1d_kernel<<<grid, block, 0, stream>>>(
        dA, dB, dC, M, N, K
    );
}