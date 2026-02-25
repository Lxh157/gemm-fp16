
cmake_minimum_required(VERSION 3.18)
project(gemm_fp16 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 生成 compile_commands.json（方便编辑器/工具）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 如果你知道自己的架构，也可以改成 89（Ada / RTX 4060 常见）
# set(CMAKE_CUDA_ARCHITECTURES 89)
# 为了兼容一点，先给一个较宽松的默认值
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

add_executable(bench_gemm
  src/main_bench.cu
  src/gemm_naive.cu
)

target_include_directories(bench_gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 开启一点优化（Day 1 足够）
target_compile_options(bench_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)