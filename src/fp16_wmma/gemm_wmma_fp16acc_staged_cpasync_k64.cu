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
constexpr int BLOCK_K = 4 * WMMA_K; // 64

constexpr int SKEW_HALF = 8;
constexpr int SMEM_STRIDE_A = BLOCK_K + SKEW_HALF; // 72
constexpr int SMEM_STRIDE_B = BLOCK_N + SKEW_HALF; // 72

constexpr int CHUNK_BYTES = 16;
constexpr int CHUNK_HALF  = CHUNK_BYTES / sizeof(half); // 8 half

constexpr int A_ROW_CHUNKS = BLOCK_K / CHUNK_HALF; // 64 / 8 = 8
constexpr int A_NUM_CHUNKS = BLOCK_M * A_ROW_CHUNKS; // 32 * 8 = 256

constexpr int B_ROW_CHUNKS = BLOCK_N / CHUNK_HALF; // 64 / 8 = 8
constexpr int B_NUM_CHUNKS = BLOCK_K * B_ROW_CHUNKS; // 64 * 8 = 512

__device__ __forceinline__ void cp_async_cg_16B(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void load_stage_to_shared_cpasync_k64(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ smemA_stage,
    half* __restrict__ smemB_stage,
    int block_row, int block_col, int k0,
    int K, int N, int tid) {

    // A tile: [32, 64], eight 16B chunks per row
    for (int chunk = tid; chunk < A_NUM_CHUNKS; chunk += THREADS_PER_BLOCK) {
        const int row = chunk / A_ROW_CHUNKS;          // 0..31
        const int chunk_in_row = chunk % A_ROW_CHUNKS; // 0..7
        const int col = chunk_in_row * CHUNK_HALF;     // 0,8,16,...,56

        half* smem_dst = &smemA_stage[row * SMEM_STRIDE_A + col];
        const half* gmem_src = &A[(block_row + row) * K + (k0 + col)];

        cp_async_cg_16B(smem_dst, gmem_src);
    }

    // B tile: [64, 64], eight 16B chunks per row
    for (int chunk = tid; chunk < B_NUM_CHUNKS; chunk += THREADS_PER_BLOCK) {
        const int row = chunk / B_ROW_CHUNKS;          // 0..31
        const int chunk_in_row = chunk % B_ROW_CHUNKS; // 0..7
        const int col = chunk_in_row * CHUNK_HALF;     // 0,8,16,...,56

        half* smem_dst = &smemB_stage[row * SMEM_STRIDE_B + col];
        const half* gmem_src = &B[(k0 + row) * N + (block_col + col)];

        cp_async_cg_16B(smem_dst, gmem_src);
    }

    cp_async_commit();
}

__global__ void gemm_wmma_fp16acc_staged_cpasync_k64_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    __shared__ half smemA[2][BLOCK_M * SMEM_STRIDE_A];
    __shared__ half smemB[2][BLOCK_K * SMEM_STRIDE_B];

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

    int read_buf = 0;

    // preload stage 0
    load_stage_to_shared_cpasync_k64(
        A, B,
        smemA[read_buf], smemB[read_buf],
        block_row, block_col, 0,
        K, N, tid
    );
    cp_async_wait_all();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        const int next_k0   = k0 + BLOCK_K;
        const int write_buf = read_buf ^ 1;

        if (next_k0 < K) {
            load_stage_to_shared_cpasync_k64(
                A, B,
                smemA[write_buf], smemB[write_buf],
                block_row, block_col, next_k0,
                K, N, tid
            );
        }

        // kk = 0, 16, 32, 48
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
        
            const half* A_smem_ptr = &smemA[read_buf][(warp_m * WMMA_M) * SMEM_STRIDE_A + kk];
            const half* B_smem_ptr = &smemB[read_buf][kk * SMEM_STRIDE_B + warp_n * WMMA_N];
        
            wmma::load_matrix_sync(a_frag, A_smem_ptr, SMEM_STRIDE_A);
            wmma::load_matrix_sync(b_frag, B_smem_ptr, SMEM_STRIDE_B);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        if (next_k0 < K) {
            cp_async_wait_all();
        }
        __syncthreads();

        read_buf ^= 1;
    }

    float* C_ptr = C + c_row * N + c_col;
    wmma::store_matrix_sync(C_ptr, c_frag, N, wmma::mem_row_major);
}

void launch_gemm_wmma_fp16acc_staged_cpasync_k64(
    const half* dA, const half* dB, float* dC,
    int M, int N, int K, cudaStream_t stream) {

    if (M % 16 != 0 || N % 16 != 0 || K % 64 != 0) {
        std::fprintf(stderr,
            "gemm_wmma_fp16acc_staged_cpasync_k64: currently requires M,N to be multiples of 16 and K to be a multiple of 64. "
            "Got M=%d N=%d K=%d\n",
            M, N, K);
        std::exit(EXIT_FAILURE);
    }

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(
        N / BLOCK_N,
        M / BLOCK_M
    );

    gemm_wmma_fp16acc_staged_cpasync_k64_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}