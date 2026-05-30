// src/gemm_mma_fp16acc.cu
#include "utils.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARPS_PER_BLOCK_M = 1;
constexpr int WARPS_PER_BLOCK_N = 4;
constexpr int WARPS_PER_BLOCK   = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N; // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;                  // 128

constexpr int BLOCK_M = WARPS_PER_BLOCK_M * MMA_M; // 16
constexpr int BLOCK_N = WARPS_PER_BLOCK_N * MMA_N; // 32
constexpr int BLOCK_K = MMA_K;                     // 16

// 首版先不用复杂 permute；只保留轻微 skew，避免最基础的 shared bank 冲突
constexpr int SKEW_HALF = 8;
constexpr int SMEM_STRIDE_A = BLOCK_K + SKEW_HALF; // 24
constexpr int SMEM_STRIDE_B = BLOCK_N + SKEW_HALF; // 40

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 smem_ptr64;          \n"
        "  cvta.to.shared.u64 smem_ptr64, %1; \n"
        "  cvt.u32.u64 %0, smem_ptr64;    \n"
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
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

// 对 m16n8k16.row.col.f32.f16.f16.f32，首版按常见 4-acc/thread 布局写回。
// lane -> 2 rows x 2 cols
__device__ __forceinline__ void store_c_frag_16x8(float* C, int ldc,
                                                  int lane,
                                                  int tile_row, int tile_col,
                                                  float c0, float c1,
                                                  float c2, float c3,
                                                  int M, int N) {
    const int row_pair = lane / 4;         // 0..7
    const int col_pair = (lane % 4) * 2;   // 0,2,4,6

    const int r0 = tile_row + row_pair;
    const int r1 = tile_row + row_pair + 8;
    const int c_base = tile_col + col_pair;

    if (r0 < M && c_base + 1 < N) {
        C[r0 * ldc + c_base + 0] = c0;
        C[r0 * ldc + c_base + 1] = c1;
    }
    if (r1 < M && c_base + 1 < N) {
        C[r1 * ldc + c_base + 0] = c2;
        C[r1 * ldc + c_base + 1] = c3;
    }
}

__global__ void gemm_mma_fp16acc_kernel(const half* __restrict__ A,
                                        const half* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
    __shared__ half smemA[BLOCK_M * SMEM_STRIDE_A]; // [16][24]
    __shared__ half smemB[BLOCK_K * SMEM_STRIDE_B]; // [16][40]

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;   // 0..3
    const int lane    = tid % 32;   // 0..31

    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    const int warp_m = 0;
    const int warp_n = warp_id;     // 4 warps across N

    const int tile_row = block_row + warp_m * MMA_M;
    const int tile_col = block_col + warp_n * MMA_N;

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // stage A: [16,16]
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += THREADS_PER_BLOCK) {
            const int r = idx / BLOCK_K;
            const int c = idx % BLOCK_K;
            const int gr = block_row + r;
            const int gc = k0 + c;
            half v = __float2half(0.0f);
            if (gr < M && gc < K) {
                v = A[gr * K + gc];
            }
            smemA[r * SMEM_STRIDE_A + c] = v;
        }

        // stage B: [16,32]
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            const int r = idx / BLOCK_N;
            const int c = idx % BLOCK_N;
            const int gr = k0 + r;
            const int gc = block_col + c;
            half v = __float2half(0.0f);
            if (gr < K && gc < N) {
                v = B[gr * N + gc];
            }
            smemB[r * SMEM_STRIDE_B + c] = v;
        }

        __syncthreads();

        // A: 16x16 -> ldmatrix x4
        uint32_t a0, a1, a2, a3;
        {
            // ldmatrix addressing:
            // row = lane % 16, col = (lane / 16) * 8
            const int ldm_row = lane % 16;
            const int ldm_col = (lane / 16) * 8;
            uint32_t addr = cvta_to_shared_u32(
                &smemA[ldm_row * SMEM_STRIDE_A + ldm_col]);
            ldmatrix_x4(a0, a1, a2, a3, addr);
        }

        // B: each warp gets its own [16,8] tile, row-major in smem, use x2.trans
        uint32_t b0, b1;
        {
            const int ldm_row = lane % 16;
            const int warp_tile_col = warp_n * MMA_N; // 0/8/16/24
            uint32_t addr = cvta_to_shared_u32(
                &smemB[ldm_row * SMEM_STRIDE_B + warp_tile_col]);
            ldmatrix_x2_trans(b0, b1, addr);
        }

        mma_m16n8k16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);

        __syncthreads();
    }

    store_c_frag_16x8(C, N, lane, tile_row, tile_col, c0, c1, c2, c3, M, N);
}

} // namespace

void launch_gemm_mma_fp16acc(const half* dA, const half* dB, float* dC,
                             int M, int N, int K, cudaStream_t stream) {
    if (K % MMA_K != 0) {
        std::fprintf(stderr,
                     "gemm_mma_fp16acc: currently requires K %% 16 == 0. "
                     "Got K=%d\n", K);
        std::exit(EXIT_FAILURE);
    }

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    gemm_mma_fp16acc_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}