// Hopper WGMMA teaching kernel: one warp-group computes one 64x64xK tile.
// This version uses TMA for both A and B tiles. B is still packed as eight
// independent 16x8 panels, so each K tile issues two A TMA copies and eight B
// TMA copies into the existing WGMMA shared-memory layout.
#include "utils.cuh"

#define __cccl_lib_experimental_ctk12_cp_async_exposure
#include <cuda/barrier>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>

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
constexpr int A_TMA_BYTES = A_ELEMS * static_cast<int>(sizeof(half));
constexpr int B_TMA_BYTES = B_ELEMS * static_cast<int>(sizeof(half));
constexpr int AB_TMA_BYTES = A_TMA_BYTES + B_TMA_BYTES;

void check_driver(CUresult result, const char* call) {
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* msg = nullptr;
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &msg);
        std::fprintf(stderr, "%s failed: %s (%s)\n",
                     call, name ? name : "unknown", msg ? msg : "unknown");
        std::exit(EXIT_FAILURE);
    }
}

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

__device__ __forceinline__ void load_ab_tma_stage(
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    half* __restrict__ smem_a, half* __restrict__ smem_b,
    int block_row, int block_col, int k0, int tid,
    cuda::barrier<cuda::thread_scope_block>& bar) {
    if (tid == 0) {
        cuda::device::barrier_expect_tx(bar, AB_TMA_BYTES);
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            smem_a, tma_a, k0, block_row, bar);
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            smem_a + 512, tma_a, k0 + 8, block_row, bar);
        #pragma unroll
        for (int panel = 0; panel < 8; ++panel) {
            cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
                smem_b + panel * 128, tma_b, block_col + panel * 8, k0, bar);
        }
        bar.arrive_and_wait();
    }
}

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

__device__ __forceinline__ void issue_wgmma_m64n64_atoms(
    half* smem_a, half* smem_b, bool first_k,
    float& d0,float& d1,float& d2,float& d3,float& d4,float& d5,float& d6,float& d7,
    float& d8,float& d9,float& d10,float& d11,float& d12,float& d13,float& d14,float& d15,
    float& d16,float& d17,float& d18,float& d19,float& d20,float& d21,float& d22,float& d23,
    float& d24,float& d25,float& d26,float& d27,float& d28,float& d29,float& d30,float& d31) {
    uint64_t desc_a = make_wgmma_desc(smem_a, 64, 8);
    wgmma_fence();
    if (first_k) {
        wgmma_m64n8k16_atom_scale0(d0,d1,d2,d3, desc_a, make_wgmma_desc(&smem_b[0 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale0(d4,d5,d6,d7, desc_a, make_wgmma_desc(&smem_b[1 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale0(d8,d9,d10,d11, desc_a, make_wgmma_desc(&smem_b[2 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale0(d12,d13,d14,d15, desc_a, make_wgmma_desc(&smem_b[3 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale0(d16,d17,d18,d19, desc_a, make_wgmma_desc(&smem_b[4 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale0(d20,d21,d22,d23, desc_a, make_wgmma_desc(&smem_b[5 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale0(d24,d25,d26,d27, desc_a, make_wgmma_desc(&smem_b[6 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale0(d28,d29,d30,d31, desc_a, make_wgmma_desc(&smem_b[7 * 128], 8, 1));
    } else {
        wgmma_m64n8k16_atom_scale1(d0,d1,d2,d3, desc_a, make_wgmma_desc(&smem_b[0 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale1(d4,d5,d6,d7, desc_a, make_wgmma_desc(&smem_b[1 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale1(d8,d9,d10,d11, desc_a, make_wgmma_desc(&smem_b[2 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale1(d12,d13,d14,d15, desc_a, make_wgmma_desc(&smem_b[3 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale1(d16,d17,d18,d19, desc_a, make_wgmma_desc(&smem_b[4 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale1(d20,d21,d22,d23, desc_a, make_wgmma_desc(&smem_b[5 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale1(d24,d25,d26,d27, desc_a, make_wgmma_desc(&smem_b[6 * 128], 8, 1));
        wgmma_m64n8k16_atom_scale1(d28,d29,d30,d31, desc_a, make_wgmma_desc(&smem_b[7 * 128], 8, 1));
    }
    wgmma_commit();
}

__global__ void gemm_wgmma_m64n64k16_tma_ab_kernel(
                                                      __grid_constant__ const CUtensorMap tma_a,
                                                      __grid_constant__ const CUtensorMap tma_b,
                                                      const half* __restrict__ A,
                                                      const half* __restrict__ B,
                                                      float* __restrict__ C,
                                                      int M, int N, int K) {
    __shared__ __align__(16) half smem_a_storage[2 * A_ELEMS];
    __shared__ __align__(16) half smem_b_storage[2 * B_ELEMS];
    __shared__ cuda::barrier<cuda::thread_scope_block> tma_bar[2];
    half* smem_a[2] = {smem_a_storage, smem_a_storage + A_ELEMS};
    half* smem_b[2] = {smem_b_storage, smem_b_storage + B_ELEMS};
    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * BLOCK_M;
    const int block_col = blockIdx.x * BLOCK_N;
    float d0=0.f; float d1=0.f; float d2=0.f; float d3=0.f; float d4=0.f; float d5=0.f; float d6=0.f; float d7=0.f;
    float d8=0.f; float d9=0.f; float d10=0.f; float d11=0.f; float d12=0.f; float d13=0.f; float d14=0.f; float d15=0.f;
    float d16=0.f; float d17=0.f; float d18=0.f; float d19=0.f; float d20=0.f; float d21=0.f; float d22=0.f; float d23=0.f;
    float d24=0.f; float d25=0.f; float d26=0.f; float d27=0.f; float d28=0.f; float d29=0.f; float d30=0.f; float d31=0.f;

    if (tid < 2) {
        init(&tma_bar[tid], 1);
    }
    __syncthreads();

    int read_buf = 0;
    load_ab_tma_stage(&tma_a, &tma_b, smem_a[read_buf], smem_b[read_buf],
                      block_row, block_col, 0, tid, tma_bar[read_buf]);
    __syncthreads();
    fence_proxy_async_shared();

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        const int next_k0 = k0 + BLOCK_K;
        const int write_buf = read_buf ^ 1;
        issue_wgmma_m64n64_atoms(smem_a[read_buf], smem_b[read_buf], k0 == 0,
            d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,
            d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31);
        if (next_k0 < K) {
            load_ab_tma_stage(&tma_a, &tma_b, smem_a[write_buf], smem_b[write_buf],
                              block_row, block_col, next_k0, tid, tma_bar[write_buf]);
        }
        wgmma_wait0();
        __syncthreads();
        if (next_k0 < K) {
            fence_proxy_async_shared();
        }
        read_buf = write_buf;
    }
    store_accum_m64n64(C, N, block_row, block_col, tid,
        d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,
        d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31);
}

} // namespace

void launch_gemm_wgmma_m64n64k16_tma_ab(const half* dA, const half* dB, float* dC,
                                        int M, int N, int K, cudaStream_t stream) {
    if (M % BLOCK_M != 0 || N % BLOCK_N != 0 || K % BLOCK_K != 0) {
        std::fprintf(stderr,
                     "gemm_wgmma_m64n64k16_tma_ab: requires M %% 64 == 0, N %% 64 == 0, K %% 16 == 0. Got M=%d N=%d K=%d\n",
                     M, N, K);
        std::exit(EXIT_FAILURE);
    }
    alignas(64) CUtensorMap tma_a;
    alignas(64) CUtensorMap tma_b;
    const cuuint64_t a_global_dim[2] = {
        static_cast<cuuint64_t>(K),
        static_cast<cuuint64_t>(M)
    };
    const cuuint64_t a_global_stride[1] = {
        static_cast<cuuint64_t>(K * sizeof(half))
    };
    const cuuint32_t a_box_dim[2] = {8, 64};
    const cuuint32_t element_stride[2] = {1, 1};
    check_driver(cuTensorMapEncodeTiled(
        &tma_a, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
        const_cast<half*>(dA),
        a_global_dim, a_global_stride, a_box_dim, element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled(A)");

    const cuuint64_t b_global_dim[2] = {
        static_cast<cuuint64_t>(N),
        static_cast<cuuint64_t>(K)
    };
    const cuuint64_t b_global_stride[1] = {
        static_cast<cuuint64_t>(N * sizeof(half))
    };
    const cuuint32_t b_box_dim[2] = {8, 16};
    check_driver(cuTensorMapEncodeTiled(
        &tma_b, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2,
        const_cast<half*>(dB),
        b_global_dim, b_global_stride, b_box_dim, element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "cuTensorMapEncodeTiled(B)");

    dim3 block(WG_THREADS);
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    gemm_wgmma_m64n64k16_tma_ab_kernel<<<grid, block, 0, stream>>>(
        tma_a, tma_b, dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}
