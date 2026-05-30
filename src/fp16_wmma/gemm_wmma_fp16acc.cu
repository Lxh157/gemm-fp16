// src/gemm_wmma_fp16acc.cu
#include "utils.cuh"

#include <mma.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace nvcuda;

// WMMA tile
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// block = 2 x 4 warps => 32 x 64 output tile
constexpr int WARPS_PER_BLOCK_M = 2;
constexpr int WARPS_PER_BLOCK_N = 4;
constexpr int WARPS_PER_BLOCK   = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N; // 8
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;                  // 256

constexpr int BLOCK_M = WARPS_PER_BLOCK_M * WMMA_M; // 32
constexpr int BLOCK_N = WARPS_PER_BLOCK_N * WMMA_N; // 64
constexpr int BLOCK_K = WMMA_K;                     // 16

// small skew to reduce shared-memory bank conflicts in WMMA loads
constexpr int SKEW_HALF = 8;
constexpr int SMEM_STRIDE_A = BLOCK_K + SKEW_HALF; // 24
constexpr int SMEM_STRIDE_B = BLOCK_N + SKEW_HALF; // 72

__global__ void gemm_wmma_fp16acc_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    __shared__ half smemA[BLOCK_M * SMEM_STRIDE_A]; // [32][24]
    __shared__ half smemB[BLOCK_K * SMEM_STRIDE_B]; // [16][72]

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;   // 0..7
    const int lane_id = tid % 32;
    (void)lane_id;

    const int warp_m = warp_id / WARPS_PER_BLOCK_N; // 0..1
    const int warp_n = warp_id % WARPS_PER_BLOCK_N; // 0..3

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int c_row = block_row + warp_m * WMMA_M;
    const int c_col = block_col + warp_n * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // stage A tile: [BLOCK_M, BLOCK_K] = [32, 16]
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += THREADS_PER_BLOCK) {
            const int row = idx / BLOCK_K; // 0..31
            const int col = idx % BLOCK_K; // 0..15
            smemA[row * SMEM_STRIDE_A + col] = A[(block_row + row) * K + (k0 + col)];
        }

        // stage B tile: [BLOCK_K, BLOCK_N] = [16, 64]
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            const int row = idx / BLOCK_N; // 0..15
            const int col = idx % BLOCK_N; // 0..63
            smemB[row * SMEM_STRIDE_B + col] = B[(k0 + row) * N + (block_col + col)];
        }

        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        const half* A_smem_ptr = &smemA[(warp_m * WMMA_M) * SMEM_STRIDE_A];
        const half* B_smem_ptr = &smemB[warp_n * WMMA_N];

        wmma::load_matrix_sync(a_frag, A_smem_ptr, SMEM_STRIDE_A);
        wmma::load_matrix_sync(b_frag, B_smem_ptr, SMEM_STRIDE_B);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    float* C_ptr = C + c_row * N + c_col;
    wmma::store_matrix_sync(C_ptr, c_frag, N, wmma::mem_row_major);
}

void launch_gemm_wmma_fp16acc(
    const half* dA, const half* dB, float* dC,
    int M, int N, int K, cudaStream_t stream) {

    if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        std::fprintf(stderr,
            "gemm_wmma_fp16acc: currently requires M,N,K to be multiples of 16. "
            "Got M=%d N=%d K=%d\n",
            M, N, K);
        std::exit(EXIT_FAILURE);
    }

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(
        N / BLOCK_N,
        M / BLOCK_M
    );

    gemm_wmma_fp16acc_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}