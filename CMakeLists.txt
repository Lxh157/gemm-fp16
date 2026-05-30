
cmake_minimum_required(VERSION 3.18)
project(gemm_fp16 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 生成 compile_commands.json（方便编辑器/工具）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(bench_gemm
  src/main_bench.cu
  src/fp32/gemm_naive.cu
  src/fp32/gemm_tiled.cu
  src/fp32/gemm_tiled_rb1x4.cu
  src/fp32/gemm_tiled_rb2x4.cu
  src/fp32/gemm_thread_tiled_1d.cu
  src/fp32/gemm_cublas.cu
  src/fp32/cublaslt_baseline.cu


  src/fail/gemm_wmma_fp16acc_staged_db.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_bcol.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_4x2.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_split.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B8.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B16.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B24.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B32.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B32.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_skewA24_B32.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_skewA24_B16.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k32_4x8_skew16.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4.cu
  src/fail/gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle2d_n4.cu


  src/fp16_acc/gemm_tiled_fp16acc.cu
  src/fp16_acc/gemm_tiled_fp16acc_rb1x4.cu
  src/fp16_acc/gemm_tiled_fp16acc_rb2x4.cu
  src/fp16_acc/gemm_cublas_gemmex_fp16acc.cu

  src/fp16_wmma/gemm_wmma_fp16acc.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k32.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k32_skew16.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k32_4x4_skew16.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k64.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k64_skew16.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch.cu
  src/fp16_wmma/gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2.cu

  src/fp16_mma/gemm_mma_fp16acc.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_k32.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_k32_4x2.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_k32_4x4.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k32_4x2.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16.cu
  src/fp16_mma/gemm_mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16.cu

  src/cublaslt_fp16acc.cu
)


target_include_directories(bench_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 开启一点优化（Day 1 足够）
target_compile_options(bench_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

# CUDA Toolkit（提供 CUDA::cudart / CUDA::cublas 等目标）
find_package(CUDAToolkit REQUIRED)

target_link_libraries(bench_gemm PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
