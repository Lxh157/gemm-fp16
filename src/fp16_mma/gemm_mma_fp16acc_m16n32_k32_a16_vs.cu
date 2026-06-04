// K32 staged cp.async variant with vectorized C stores and no final CTA sync.
#include "utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// 4x2 warps/block. Each warp computes one 16x32 tile.
// CTA output tile: 64x64. Shared stage depth: K=32.
// Global-to-shared staging uses double-buffered cp.async.
constexpr int WARPS_PER_BLOCK_M = 4;
constexpr int WARPS_PER_BLOCK_N = 2;
constexpr int WARPS_PER_BLOCK = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

constexpr int WARP_TILE_M = MMA_M;
constexpr int WARP_TILE_N = 4 * MMA_N;                   // 32
constexpr int BLOCK_M = WARPS_PER_BLOCK_M * WARP_TILE_M; // 64
constexpr int BLOCK_N = WARPS_PER_BLOCK_N * WARP_TILE_N; // 64
constexpr int BLOCK_K = 2 * MMA_K;                       // 32

constexpr int SKEW_A_HALF = 16;
constexpr int SKEW_B_HALF = 8;
constexpr int SMEM_STRIDE_A = BLOCK_K + SKEW_A_HALF; // 48
constexpr int SMEM_STRIDE_B = BLOCK_N + SKEW_B_HALF; // 72

constexpr int CHUNK_BYTES = 16;
constexpr int CHUNK_HALF = CHUNK_BYTES / sizeof(half); // 8 half

constexpr int A_ROW_CHUNKS = BLOCK_K / CHUNK_HALF;   // 4
constexpr int A_NUM_CHUNKS = BLOCK_M * A_ROW_CHUNKS; // 256

constexpr int B_ROW_CHUNKS = BLOCK_N / CHUNK_HALF;   // 8
constexpr int B_NUM_CHUNKS = BLOCK_K * B_ROW_CHUNKS; // 256

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 smem_ptr64;              \n"
        "  cvta.to.shared.u64 smem_ptr64, %1; \n"
        "  cvt.u32.u64 %0, smem_ptr64;        \n"
        "}\n"
        : "=r"(addr)
        : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1,
                                                  uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
}

__device__ __forceinline__ void cp_async_cg_16B(void* smem_ptr,
                                                const void* gmem_ptr) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void mma_m16n8k16(uint32_t a0, uint32_t a1,
                                             uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1,
                                             float& c0, float& c1,
                                             float& c2, float& c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void store_c_frag_16x8(float* C, int ldc,
                                                  int lane,
                                                  int tile_row, int tile_col,
                                                  float c0, float c1,
                                                  float c2, float c3,
                                                  int M, int N) {
    const int row_pair = lane / 4;
    const int col_pair = (lane % 4) * 2;

    const int r0 = tile_row + row_pair;
    const int r1 = tile_row + row_pair + 8;
    const int c_base = tile_col + col_pair;

    if (r0 < M && c_base + 1 < N) {
        *reinterpret_cast<float2*>(&C[r0 * ldc + c_base]) = make_float2(c0, c1);
    }
    if (r1 < M && c_base + 1 < N) {
        *reinterpret_cast<float2*>(&C[r1 * ldc + c_base]) = make_float2(c2, c3);
    }
}

__device__ __forceinline__ void mma_16x32_step(
    const half* smemA, const half* smemB,
    int warp_m, int warp_n, int lane, int kk,
    float& c00, float& c01, float& c02, float& c03,
    float& c10, float& c11, float& c12, float& c13,
    float& c20, float& c21, float& c22, float& c23,
    float& c30, float& c31, float& c32, float& c33) {

    uint32_t a0, a1, a2, a3;
    {
        const int ldm_row = warp_m * WARP_TILE_M + (lane % 16);
        const int ldm_col = kk + (lane / 16) * 8;
        uint32_t addr = cvta_to_shared_u32(
            &smemA[ldm_row * SMEM_STRIDE_A + ldm_col]);
        ldmatrix_x4(a0, a1, a2, a3, addr);
    }

    const int b_col_base = warp_n * WARP_TILE_N;
    uint32_t b0, b1;

    uint32_t b_addr0 = cvta_to_shared_u32(
        &smemB[(kk + (lane % 16)) * SMEM_STRIDE_B + b_col_base + 0]);
    ldmatrix_x2_trans(b0, b1, b_addr0);
    mma_m16n8k16(a0, a1, a2, a3, b0, b1, c00, c01, c02, c03);

    uint32_t b_addr1 = cvta_to_shared_u32(
        &smemB[(kk + (lane % 16)) * SMEM_STRIDE_B + b_col_base + 8]);
    ldmatrix_x2_trans(b0, b1, b_addr1);
    mma_m16n8k16(a0, a1, a2, a3, b0, b1, c10, c11, c12, c13);

    uint32_t b_addr2 = cvta_to_shared_u32(
        &smemB[(kk + (lane % 16)) * SMEM_STRIDE_B + b_col_base + 16]);
    ldmatrix_x2_trans(b0, b1, b_addr2);
    mma_m16n8k16(a0, a1, a2, a3, b0, b1, c20, c21, c22, c23);

    uint32_t b_addr3 = cvta_to_shared_u32(
        &smemB[(kk + (lane % 16)) * SMEM_STRIDE_B + b_col_base + 24]);
    ldmatrix_x2_trans(b0, b1, b_addr3);
    mma_m16n8k16(a0, a1, a2, a3, b0, b1, c30, c31, c32, c33);
}

__device__ __forceinline__ void load_stage_to_shared_cpasync(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ smemA_stage,
    half* __restrict__ smemB_stage,
    int block_row, int block_col, int k0,
    int K, int N, int tid) {

    for (int chunk = tid; chunk < A_NUM_CHUNKS; chunk += THREADS_PER_BLOCK) {
        const int row = chunk / A_ROW_CHUNKS;
        const int chunk_in_row = chunk % A_ROW_CHUNKS;
        const int col = chunk_in_row * CHUNK_HALF;

        half* smem_dst = &smemA_stage[row * SMEM_STRIDE_A + col];
        const half* gmem_src = &A[(block_row + row) * K + (k0 + col)];
        cp_async_cg_16B(smem_dst, gmem_src);
    }

    for (int chunk = tid; chunk < B_NUM_CHUNKS; chunk += THREADS_PER_BLOCK) {
        const int row = chunk / B_ROW_CHUNKS;
        const int chunk_in_row = chunk % B_ROW_CHUNKS;
        const int col = chunk_in_row * CHUNK_HALF;

        half* smem_dst = &smemB_stage[row * SMEM_STRIDE_B + col];
        const half* gmem_src = &B[(k0 + row) * N + (block_col + col)];
        cp_async_cg_16B(smem_dst, gmem_src);
    }

    cp_async_commit();
}

__global__ void gemm_mma_fp16acc_m16n32_k32_a16_vs_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {

    __shared__ half smemA[2][BLOCK_M * SMEM_STRIDE_A]; // [2][64][48]
    __shared__ half smemB[2][BLOCK_K * SMEM_STRIDE_B]; // [2][32][72]

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;

    const int warp_m = warp_id / WARPS_PER_BLOCK_N;
    const int warp_n = warp_id % WARPS_PER_BLOCK_N;

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int tile_row = block_row + warp_m * WARP_TILE_M;
    const int tile_col = block_col + warp_n * WARP_TILE_N;

    float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
    float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
    float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
    float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

    int read_buf = 0;

    load_stage_to_shared_cpasync(
        A, B, smemA[read_buf], smemB[read_buf],
        block_row, block_col, 0,
        K, N, tid);
    cp_async_wait_all();
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        const int next_k0 = k0 + BLOCK_K;
        const int write_buf = read_buf ^ 1;

        if (next_k0 < K) {
            load_stage_to_shared_cpasync(
                A, B, smemA[write_buf], smemB[write_buf],
                block_row, block_col, next_k0,
                K, N, tid);
        }

        mma_16x32_step(
            smemA[read_buf], smemB[read_buf], warp_m, warp_n, lane, 0,
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            c20, c21, c22, c23,
            c30, c31, c32, c33);

        mma_16x32_step(
            smemA[read_buf], smemB[read_buf], warp_m, warp_n, lane, MMA_K,
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            c20, c21, c22, c23,
            c30, c31, c32, c33);

        if (next_k0 < K) {
            cp_async_wait_all();
            __syncthreads();
        }
        read_buf ^= 1;
    }

    store_c_frag_16x8(C, N, lane, tile_row, tile_col + 0,
                      c00, c01, c02, c03, M, N);
    store_c_frag_16x8(C, N, lane, tile_row, tile_col + 8,
                      c10, c11, c12, c13, M, N);
    store_c_frag_16x8(C, N, lane, tile_row, tile_col + 16,
                      c20, c21, c22, c23, M, N);
    store_c_frag_16x8(C, N, lane, tile_row, tile_col + 24,
                      c30, c31, c32, c33, M, N);
}

} // namespace

void launch_gemm_mma_fp16acc_m16n32_k32_a16_vs(
    const half* dA, const half* dB, float* dC,
    int M, int N, int K, cudaStream_t stream) {

    if (M % BLOCK_M != 0 || N % BLOCK_N != 0 || K % BLOCK_K != 0) {
        std::fprintf(stderr,
                     "gemm_mma_fp16acc_m16n32_k32_a16_vs: currently requires "
                     "M %% 64 == 0, N %% 64 == 0, and K %% 32 == 0. "
                     "Got M=%d N=%d K=%d\n",
                     M, N, K);
        std::exit(EXIT_FAILURE);
    }

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(N / BLOCK_N, M / BLOCK_M);

    gemm_mma_fp16acc_m16n32_k32_a16_vs_kernel<<<grid, block, 0, stream>>>(
        dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
