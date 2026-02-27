
cmake_minimum_required(VERSION 3.18)
project(gemm_fp16 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 生成 compile_commands.json（方便编辑器/工具）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_executable(bench_gemm
  src/main_bench.cu
  src/gemm_naive.cu
  src/gemm_tiled.cu
  src/gemm_tiled_rb1x4.cu
  src/gemm_tiled_rb2x4.cu
  src/gemm_cublas.cu
  src/cublaslt_baseline.cu
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