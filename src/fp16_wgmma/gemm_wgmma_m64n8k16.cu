
// Minimal Hopper WGMMA teaching kernel: one warp-group computes one 64x8xK tile.
#include "utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {

constexpr int WG_THREADS = 128;
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 8;
constexpr int BLOCK_K = 16;
constexpr int A_ELEMS = BLOCK_M * BLOCK_K;
constexpr int B_ELEMS = BLOCK_K * BLOCK_N;

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

__device__ __forceinline__ uint64_t make_wgmma_desc(const void* smem_ptr,
                                                     int leading_bytes,
                                                     int stride_bytes) {
    uint64_t addr = cvta_to_shared_u32(smem_ptr);
    uint64_t desc = 0;
    desc |= ((addr & 0x3FFFFull) >> 4);
    desc |= (uint64_t(leading_bytes) << 16);
    desc |= (uint64_t(stride_bytes) << 32);
    // leading_bytes/stride_bytes are descriptor fields in 16B units.
    // bits 63:62 = 0: no swizzle. Base offset is zero for no-swizzle mode.
    return desc;
}

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n8k16_f32_f16_f16_scale0(
    float& d0, float& d1, float& d2, float& d3,
    uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0, %1, %2, %3}, %4, %5, 0, 1, 1, 0, 1;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ void wgmma_m64n8k16_f32_f16_f16_scale1(
    float& d0, float& d1, float& d2, float& d3,
    uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 1;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ void store_accum_m64n8(float* C, int ldc,
                                                   int block_row, int block_col,
                                                   int tid,
                                                   float d0, float d1,
                                                   float d2, float d3) {
    const int warpgroup_warp = tid / 32;
    const int lane_in_warp = tid & 31;
    const int row0 = block_row + warpgroup_warp * 16 + lane_in_warp / 4;
    const int row1 = row0 + 8;
    const int col = block_col + (lane_in_warp & 3) * 2;

    *reinterpret_cast<float2*>(&C[row0 * ldc + col]) = make_float2(d0, d1);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col]) = make_float2(d2, d3);
}

__global__ void gemm_wgmma_m64n8k16_kernel(const half* __restrict__ A,
                                                  const half* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int M, int N, int K) {
    __shared__ __align__(16) half smem_a[A_ELEMS];
    __shared__ __align__(16) half smem_b[B_ELEMS];

    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        for (int idx = tid; idx < A_ELEMS; idx += WG_THREADS) {
            const int r = idx / BLOCK_K;
            const int k = idx % BLOCK_K;
            // WGMMA consumes A as 8x8 core matrices. For row-major A, the two
            // K-core columns live in separate 512-element regions.
            const int smem_idx = (k / 8) * 512 + r * 8 + (k % 8);
            smem_a[smem_idx] = A[(block_row + r) * K + (k0 + k)];
        }
        for (int idx = tid; idx < B_ELEMS; idx += WG_THREADS) {
            const int k = idx / BLOCK_N;
            const int n = idx % BLOCK_N;
            // B is row-major in this benchmark. Use WGMMA's trans-B flag and
            // keep the 16x8 tile linear in shared memory.
            smem_b[k * BLOCK_N + n] = B[(k0 + k) * N + (block_col + n)];
        }
        __syncthreads();
        fence_proxy_async_shared();

        uint64_t desc_a = make_wgmma_desc(smem_a, 64, 8);
        uint64_t desc_b = make_wgmma_desc(smem_b, 8, 1);
        wgmma_fence();
        if (k0 == 0) {
            wgmma_m64n8k16_f32_f16_f16_scale0(d0, d1, d2, d3, desc_a, desc_b);
        } else {
            wgmma_m64n8k16_f32_f16_f16_scale1(d0, d1, d2, d3, desc_a, desc_b);
        }
        wgmma_commit();
        wgmma_wait0();
        __syncthreads();
    }

    store_accum_m64n8(C, N, block_row, block_col, tid, d0, d1, d2, d3);
}

} // namespace

void launch_gemm_wgmma_m64n8k16(
    const half* dA, const half* dB, float* dC,
    int M, int N, int K, cudaStream_t stream) {
    if (M % BLOCK_M != 0 || N % BLOCK_N != 0 || K % BLOCK_K != 0) {
        std::fprintf(stderr,
                     "gemm_wgmma_m64n8k16: requires M %% 64 == 0, N %% 8 == 0, K %% 16 == 0. Got M=%d N=%d K=%d\n",
                     M, N, K);
        std::exit(EXIT_FAILURE);
    }
    dim3 block(WG_THREADS);
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    gemm_wgmma_m64n8k16_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
