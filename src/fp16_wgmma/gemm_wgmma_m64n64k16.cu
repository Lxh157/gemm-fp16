// Hopper WGMMA teaching kernel: one warp-group computes one 64x64xK tile.
// This version builds the wide tile from eight m64n8 WGMMA atoms. It validates
// the wider-N warp-group mapping before switching to a single m64n64 atom.
#include "utils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {
constexpr int WG_THREADS = 128;
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int A_ELEMS = BLOCK_M * BLOCK_K;
constexpr int B_ELEMS = BLOCK_K * BLOCK_N;

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t addr;
    asm volatile("{ .reg .u64 smem_ptr64;              \n"
                 "  cvta.to.shared.u64 smem_ptr64, %1; \n"
                 "  cvt.u32.u64 %0, smem_ptr64;        \n"
                 "}\n" : "=r"(addr) : "l"(ptr));
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
    return desc;
}

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async_shared() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_m64n8k16_atom_scale0(
    float& d0, float& d1, float& d2, float& d3, uint64_t desc_a, uint64_t desc_b) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                 "{%0, %1, %2, %3}, %4, %5, 0, 1, 1, 0, 1;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ void wgmma_m64n8k16_atom_scale1(
    float& d0, float& d1, float& d2, float& d3, uint64_t desc_a, uint64_t desc_b) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
                 "{%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 1;\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "l"(desc_a), "l"(desc_b));
}

__device__ __forceinline__ void store_accum_m64n64(float* C, int ldc,
                                                    int block_row, int block_col,
                                                    int tid,
                                                    float d0, float d1, float d2, float d3, float d4, float d5, float d6, float d7, float d8, float d9, float d10, float d11, float d12, float d13, float d14, float d15, float d16, float d17, float d18, float d19, float d20, float d21, float d22, float d23, float d24, float d25, float d26, float d27, float d28, float d29, float d30, float d31) {
    const int warpgroup_warp = tid / 32;
    const int lane_in_warp = tid & 31;
    const int row0 = block_row + warpgroup_warp * 16 + lane_in_warp / 4;
    const int row1 = row0 + 8;
    const int col = block_col + (lane_in_warp & 3) * 2;
    *reinterpret_cast<float2*>(&C[row0 * ldc + col + 0]) = make_float2(d0, d1);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col + 0]) = make_float2(d2, d3);
    *reinterpret_cast<float2*>(&C[row0 * ldc + col + 8]) = make_float2(d4, d5);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col + 8]) = make_float2(d6, d7);
    *reinterpret_cast<float2*>(&C[row0 * ldc + col + 16]) = make_float2(d8, d9);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col + 16]) = make_float2(d10, d11);
    *reinterpret_cast<float2*>(&C[row0 * ldc + col + 24]) = make_float2(d12, d13);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col + 24]) = make_float2(d14, d15);
    *reinterpret_cast<float2*>(&C[row0 * ldc + col + 32]) = make_float2(d16, d17);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col + 32]) = make_float2(d18, d19);
    *reinterpret_cast<float2*>(&C[row0 * ldc + col + 40]) = make_float2(d20, d21);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col + 40]) = make_float2(d22, d23);
    *reinterpret_cast<float2*>(&C[row0 * ldc + col + 48]) = make_float2(d24, d25);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col + 48]) = make_float2(d26, d27);
    *reinterpret_cast<float2*>(&C[row0 * ldc + col + 56]) = make_float2(d28, d29);
    *reinterpret_cast<float2*>(&C[row1 * ldc + col + 56]) = make_float2(d30, d31);
}

__global__ void gemm_wgmma_m64n64k16_kernel(const half* __restrict__ A,
                                                   const half* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int M, int N, int K) {
    __shared__ __align__(16) half smem_a[A_ELEMS];
    __shared__ __align__(16) half smem_b[B_ELEMS];
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;
    float d0=0.f;
    float d1=0.f;
    float d2=0.f;
    float d3=0.f;
    float d4=0.f;
    float d5=0.f;
    float d6=0.f;
    float d7=0.f;
    float d8=0.f;
    float d9=0.f;
    float d10=0.f;
    float d11=0.f;
    float d12=0.f;
    float d13=0.f;
    float d14=0.f;
    float d15=0.f;
    float d16=0.f;
    float d17=0.f;
    float d18=0.f;
    float d19=0.f;
    float d20=0.f;
    float d21=0.f;
    float d22=0.f;
    float d23=0.f;
    float d24=0.f;
    float d25=0.f;
    float d26=0.f;
    float d27=0.f;
    float d28=0.f;
    float d29=0.f;
    float d30=0.f;
    float d31=0.f;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        for (int idx = tid; idx < A_ELEMS; idx += WG_THREADS) {
            const int r = idx / BLOCK_K;
            const int k = idx % BLOCK_K;
            const int smem_idx = (k / 8) * 512 + r * 8 + (k % 8);
            smem_a[smem_idx] = A[(block_row + r) * K + (k0 + k)];
        }
        for (int idx = tid; idx < B_ELEMS; idx += WG_THREADS) {
            const int k = idx / BLOCK_N;
            const int n = idx % BLOCK_N;
            smem_b[(n / 8) * 128 + k * 8 + (n % 8)] = B[(k0 + k) * N + (block_col + n)];
        }
        __syncthreads();
        fence_proxy_async_shared();
        uint64_t desc_a = make_wgmma_desc(smem_a, 64, 8);
        wgmma_fence();
        #pragma unroll
        for (int ng = 0; ng < 8; ++ng) {
            uint64_t desc_b = make_wgmma_desc(&smem_b[ng * 128], 8, 1);
            if (k0 == 0) {
                switch (ng) {
                    case 0: wgmma_m64n8k16_atom_scale0(d0,d1,d2,d3, desc_a, desc_b); break;
                    case 1: wgmma_m64n8k16_atom_scale0(d4,d5,d6,d7, desc_a, desc_b); break;
                    case 2: wgmma_m64n8k16_atom_scale0(d8,d9,d10,d11, desc_a, desc_b); break;
                    case 3: wgmma_m64n8k16_atom_scale0(d12,d13,d14,d15, desc_a, desc_b); break;
                    case 4: wgmma_m64n8k16_atom_scale0(d16,d17,d18,d19, desc_a, desc_b); break;
                    case 5: wgmma_m64n8k16_atom_scale0(d20,d21,d22,d23, desc_a, desc_b); break;
                    case 6: wgmma_m64n8k16_atom_scale0(d24,d25,d26,d27, desc_a, desc_b); break;
                    case 7: wgmma_m64n8k16_atom_scale0(d28,d29,d30,d31, desc_a, desc_b); break;
                }
            } else {
                switch (ng) {
                    case 0: wgmma_m64n8k16_atom_scale1(d0,d1,d2,d3, desc_a, desc_b); break;
                    case 1: wgmma_m64n8k16_atom_scale1(d4,d5,d6,d7, desc_a, desc_b); break;
                    case 2: wgmma_m64n8k16_atom_scale1(d8,d9,d10,d11, desc_a, desc_b); break;
                    case 3: wgmma_m64n8k16_atom_scale1(d12,d13,d14,d15, desc_a, desc_b); break;
                    case 4: wgmma_m64n8k16_atom_scale1(d16,d17,d18,d19, desc_a, desc_b); break;
                    case 5: wgmma_m64n8k16_atom_scale1(d20,d21,d22,d23, desc_a, desc_b); break;
                    case 6: wgmma_m64n8k16_atom_scale1(d24,d25,d26,d27, desc_a, desc_b); break;
                    case 7: wgmma_m64n8k16_atom_scale1(d28,d29,d30,d31, desc_a, desc_b); break;
                }
            }
        }
        wgmma_commit();
        wgmma_wait0();
        __syncthreads();
    }
    store_accum_m64n64(C, N, block_row, block_col, tid, d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31);
}

} // namespace

void launch_gemm_wgmma_m64n64k16(const half* dA, const half* dB, float* dC,
                                        int M, int N, int K, cudaStream_t stream) {
    if (M % BLOCK_M != 0 || N % BLOCK_N != 0 || K % BLOCK_K != 0) {
        std::fprintf(stderr,
                     "gemm_wgmma_m64n64k16: requires M %% 64 == 0, N %% 64 == 0, K %% 16 == 0. Got M=%d N=%d K=%d\n",
                     M, N, K);
        std::exit(EXIT_FAILURE);
    }
    dim3 block(WG_THREADS);
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    gemm_wgmma_m64n64k16_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
