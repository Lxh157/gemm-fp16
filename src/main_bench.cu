#include "utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <string>
#include <vector>

// 

// 来自 gemm_naive.cu 的声明
void launch_gemm_naive(const float* dA, const float* dB, float* dC,
                       int M, int N, int K,
                       cudaStream_t stream);
// 来自 gemm_tiled.cu 的声明             
void launch_gemm_tiled(const float* dA, const float* dB, float* dC,
                       int M, int N, int K,
                       cudaStream_t stream);
// 来自 gemm_tiled_rb1x4.cu 的声明             
void launch_gemm_tiled_rb1x4(const float* dA, const float* dB, float* dC,
                             int M, int N, int K,
                             cudaStream_t stream);
// 来自 gemm_tiled_rb2x4.cu 的声明             
void launch_gemm_tiled_rb2x4(const float* dA, const float* dB, float* dC,
                             int M, int N, int K,
                             cudaStream_t stream);
// 来自 gemm_thread_tiled_1d.cu 的声明             
void launch_gemm_thread_tiled_1d(const float* dA, const float* dB, float* dC,
                             int M, int N, int K,
                             cudaStream_t stream);
// 来自 gemm_cublas.cu 的声明             
void launch_gemm_cublas_rowmajor(const float* A, const float* B, float* C,
                          int M, int N, int K, cudaStream_t stream);
// 来自 cublaslt_baseline.cu 的声明
void launch_gemm_cublaslt_rowmajor(const float* A, const float* B, float* C,
                          int M, int N, int K, cudaStream_t stream); 
// 来自 gemm_tiled_fp16acc.cu 的声明
void launch_gemm_tiled_fp16acc(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_tiled_fp16acc_rb1x4.cu 的声明
void launch_gemm_tiled_fp16acc_rb1x4(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_tiled_fp16acc_rb2x4.cu 的声明
void launch_gemm_tiled_fp16acc_rb2x4(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc.cu 的声明
void launch_gemm_wmma_fp16acc(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged.cu 的声明
void launch_gemm_wmma_fp16acc_staged(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_cublas_gemmex_fp16acc.cu 的声明
void launch_gemm_cublas_gemmex_fp16acc_rowmajor(const half* A, const half* B, float* C,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_cublaslt_fp16acc.cu 的声明
void launch_gemm_cublaslt_fp16acc_rowmajor(const half* A, const half* B, float* C,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_db.cu 的声明
void launch_gemm_wmma_fp16acc_staged_db(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_bcol.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_bcol(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_4x2.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_4x2(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_split.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_split(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_skew16.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B16.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_skewA24_B16.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA24_B16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B24.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B24(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B32.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B32(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B32.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B32(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_skewA24_B32.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA24_B32(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B8.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B8(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);


// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_4x4_skew16.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_4x4_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k32_4x8_skew16.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k32_4x8_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64_skew16.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle2d_n4.cu 的声明
void launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle2d_n4(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);

                          // 来自gemm_mma_fp16acc.cu 的声明
void launch_gemm_mma_fp16acc(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32.cu 的声明
void launch_gemm_mma_fp16acc_m16n32(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_k32.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_k32(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_k32_4x2.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_k32_4x2(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_k32_4x4.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_k32_4x4(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k32_4x2.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k32_4x2(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_k32_vs.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_k32_vs(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_k32_s16_vs.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_k32_s16_vs(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_k32_a16_vs.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_k32_a16_vs(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_k32_b16_vs.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_k32_b16_vs(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_k32_4x4_vs.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_k32_4x4_vs(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16.cu 的声明
void launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16.cu 的声明
void launch_gemm_mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
// 来自 gemm_mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16.cu 的声明
void launch_gemm_mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
void launch_gemm_mma_fp16acc_m32n32_k64_4x2(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
void launch_gemm_mma_fp16acc_m32n32_k64_4x2_s8(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
void launch_gemm_mma_fp16acc_m32n32_k128_4x2_s8(const half* dA, const half* dB, float* dC,
                          int M, int N, int K, cudaStream_t stream);
void launch_gemm_wgmma_m64n8k16_sm90a(const half* dA, const half* dB, float* dC,
                                          int M, int N, int K, cudaStream_t stream);
void launch_gemm_wgmma_m64n64k16_sm90a(const half* dA, const half* dB, float* dC,
                                           int M, int N, int K, cudaStream_t stream);
void launch_gemm_wgmma_m64n64k16_db_sm90a(const half* dA, const half* dB, float* dC,
                                              int M, int N, int K, cudaStream_t stream);
// 非严格参数解析
// 支持：
//   ./bench_gemm
//   ./bench_gemm 1024 1024 1024
//   ./bench_gemm --M 1024 --N 1024 --K 1024 --warmup 5 --repeat 20
struct Args {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int warmup = 5;
    int repeat = 20;
    bool check = true;
    std::string impl = "naive";  // "naive" or "tiled"
  };

Args parse_args(int argc, char** argv) {
  Args a;

  // 位置参数：M N K
  if (argc == 4) {
    a.M = std::stoi(argv[1]);
    a.N = std::stoi(argv[2]);
    a.K = std::stoi(argv[3]);
    return a;
  }

  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    auto need_value = [&](int idx) {
      if (idx + 1 >= argc) {
        std::cerr << "Missing value after " << argv[idx] << std::endl;
        std::exit(EXIT_FAILURE);
      }
    };

    if (s == "--M") {
      need_value(i);
      a.M = std::stoi(argv[++i]);
    } else if (s == "--N") {
      need_value(i);
      a.N = std::stoi(argv[++i]);
    } else if (s == "--K") {
      need_value(i);
      a.K = std::stoi(argv[++i]);
    } else if (s == "--warmup") {
      need_value(i);
      a.warmup = std::stoi(argv[++i]);
    } else if (s == "--repeat") {
      need_value(i);
      a.repeat = std::stoi(argv[++i]);
    } else if (s == "--no-check") {
      a.check = false;
    } else if (s == "--help" || s == "-h") {
      std::cout
          << "Usage:\n"
          << "  ./bench_gemm [M N K]\n"
          << "  ./bench_gemm --M 1024 --N 1024 --K 1024 "
             "[--warmup 5 --repeat 20 --no-check]\n";
      std::exit(0);
    } else if (s == "--impl") {
        need_value(i);
        a.impl = argv[++i];
          if (a.impl != "naive" && a.impl != "tiled" && a.impl != "tiled_rb1x4" && a.impl != "tiled_rb2x4" && a.impl != "thread_tiled_1d" && a.impl != "cublas" && a.impl != "cublaslt" && a.impl != "tiled_fp16acc" && a.impl != "tiled_fp16acc_rb1x4" && a.impl != "tiled_fp16acc_rb2x4" && a.impl != "wmma_fp16acc" && a.impl != "wmma_fp16acc_staged" && a.impl != "wmma_fp16acc_staged_db" && a.impl != "wmma_fp16acc_staged_cpasync" && a.impl != "wmma_fp16acc_staged_cpasync_k32" && a.impl != "wmma_fp16acc_staged_cpasync_k32_bcol" && a.impl != "wmma_fp16acc_staged_cpasync_k32_4x2" && a.impl != "wmma_fp16acc_staged_cpasync_k32_split" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skew16" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skewA16_B8" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skewA8_B16" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skewA24_B16" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skewA8_B12" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skewA8_B24" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skewA8_B32" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skewA16_B32" && a.impl != "wmma_fp16acc_staged_cpasync_k32_skewA24_B32" && a.impl != "wmma_fp16acc_staged_cpasync_k32_4x4_skew16"  && a.impl != "wmma_fp16acc_staged_cpasync_k32_4x8_skew16" && a.impl != "wmma_fp16acc_staged_cpasync_k64" && a.impl != "wmma_fp16acc_staged_cpasync_k64_skew16" && a.impl != "wmma_fp16acc_staged_cpasync_k64_4x4_skew16" && a.impl != "wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16" && a.impl != "wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32" && a.impl != "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch" && a.impl != "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2" && a.impl != "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4" && a.impl != "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle2d_n4"&& a.impl != "mma_fp16acc" && a.impl != "mma_fp16acc_m16n32" && a.impl != "mma_fp16acc_m16n32_k32" && a.impl != "mma_fp16acc_m16n32_k32_4x2" && a.impl != "mma_fp16acc_m16n32_k32_4x4" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k32_4x2" && a.impl != "mma_fp16acc_m16n32_k32_vs" && a.impl != "mma_fp16acc_m16n32_k32_s16_vs" && a.impl != "mma_fp16acc_m16n32_k32_a16_vs" && a.impl != "mma_fp16acc_m16n32_k32_b16_vs" && a.impl != "mma_fp16acc_m16n32_k32_4x4_vs" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k64_4x2" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32" && a.impl != "mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16" && a.impl != "mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16" && a.impl != "mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16" && a.impl != "mma_fp16acc_m32n32_k64_4x2" && a.impl != "mma_fp16acc_m32n32_k64_4x2_s8" && a.impl != "mma_fp16acc_m32n32_k128_4x2_s8" && a.impl != "wgmma_m64n8k16_sm90a" && a.impl != "wgmma_m64n64k16_sm90a" && a.impl != "wgmma_m64n64k16_db_sm90a" && a.impl != "cublas_gemmex_fp16acc" && a.impl != "cublaslt_fp16acc") {
          std::cerr << "Invalid --impl: " << a.impl
                    << " (expected naive, tiled, tiled_rb1x4, tiled_rb2x4, thread_tiled_1d, cublas, cublaslt, tiled_fp16acc, tiled_fp16acc_rb1x4, tiled_fp16acc_rb2x4, wmma_fp16acc, wmma_fp16acc_staged, wmma_fp16acc_staged_db, wmma_fp16acc_staged_cpasync, wmma_fp16acc_staged_cpasync_k32, wmma_fp16acc_staged_cpasync_k32_bcol, wmma_fp16acc_staged_cpasync_k32_4x2, wmma_fp16acc_staged_cpasync_k32_split, wmma_fp16acc_staged_cpasync_k32_skew16, wmma_fp16acc_staged_cpasync_k32_skewA16_B8, wmma_fp16acc_staged_cpasync_k32_skewA8_B16, wmma_fp16acc_staged_cpasync_k32_skewA24_B16, wmma_fp16acc_staged_cpasync_k32_skewA8_B12, wmma_fp16acc_staged_cpasync_k32_skewA8_B24, wmma_fp16acc_staged_cpasync_k32_skewA8_B32, wmma_fp16acc_staged_cpasync_k32_skewA16_B32, wmma_fp16acc_staged_cpasync_k32_skewA24_B32, wmma_fp16acc_staged_cpasync_k32_4x4_skew16, wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16, wmma_fp16acc_staged_cpasync_k32_4x8_skew16, wmma_fp16acc_staged_cpasync_k64, wmma_fp16acc_staged_cpasync_k64_skew16, wmma_fp16acc_staged_cpasync_k64_4x4_skew16, wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32, wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch, wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2, wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4, wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle2d_n4, mma_fp16acc, mma_fp16acc_m16n32, mma_fp16acc_m16n32_k32, mma_fp16acc_m16n32_k32_4x2, mma_fp16acc_m16n32_k32_4x4, mma_fp16acc_m16n32_staged_cpasync_k32_4x2, mma_fp16acc_m16n32_k32_vs, mma_fp16acc_m16n32_k32_s16_vs, mma_fp16acc_m16n32_k32_a16_vs, mma_fp16acc_m16n32_k32_b16_vs, mma_fp16acc_m16n32_k32_4x4_vs, mma_fp16acc_m16n32_staged_cpasync_k64_4x2, mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16, mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16, mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore, mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync, mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch, mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck, mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32, mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16, mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16, mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16, mma_fp16acc_m32n32_k64_4x2, mma_fp16acc_m32n32_k64_4x2_s8, mma_fp16acc_m32n32_k128_4x2_s8, wgmma_m64n8k16_sm90a, wgmma_m64n64k16_sm90a, wgmma_m64n64k16_db_sm90a, cublas_gemmex_fp16acc, or cublaslt_fp16acc)" << std::endl;
          std::exit(EXIT_FAILURE);
        }
    } else {
      std::cerr << "Unknown argument: " << s << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  return a;
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  const int M = args.M;
  const int N = args.N;
  const int K = args.K;

  std::cout << "=== bench_gemm (" << args.impl << ", " << "FP16" << ") ===\n";
  std::cout << "M=" << M << ", N=" << N << ", K=" << K
            << ", warmup=" << args.warmup
            << ", repeat=" << args.repeat
            << ", check=" << (args.check ? "true" : "false") << "\n";

  CHECK_CUDA(cudaSetDevice(0));

  // Host buffers
  std::vector<float> hA(static_cast<size_t>(M) * K);
  std::vector<float> hB(static_cast<size_t>(K) * N);
  std::vector<float> hC(static_cast<size_t>(M) * N, 0.0f);
  std::vector<float> hC_ref;

  fill_random(hA, 123);
  fill_random(hB, 456);

  bool use_fp16_inputs = (args.impl == "tiled_fp16acc" ||
                          args.impl == "tiled_fp16acc_rb1x4" ||
                          args.impl == "tiled_fp16acc_rb2x4"||
                          args.impl == "wmma_fp16acc" ||
                          args.impl == "wmma_fp16acc_staged" ||
                          args.impl == "wmma_fp16acc_staged_db" ||
                          args.impl == "wmma_fp16acc_staged_cpasync" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_bcol" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_4x2" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_split" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_skew16" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA16_B8" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA8_B16" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA24_B16" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA8_B24" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA8_B32" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA16_B32" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA24_B32" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_4x4_skew16" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k32_4x8_skew16" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64_skew16" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4" ||
                          args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle2d_n4" ||

                          args.impl == "mma_fp16acc" ||
                          args.impl == "mma_fp16acc_m16n32" ||
                          args.impl == "mma_fp16acc_m16n32_k32" ||
                          args.impl == "mma_fp16acc_m16n32_k32_4x2" ||
                          args.impl == "mma_fp16acc_m16n32_k32_4x4" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k32_4x2" ||
                          args.impl == "mma_fp16acc_m16n32_k32_vs" ||
                          args.impl == "mma_fp16acc_m16n32_k32_s16_vs" ||
                          args.impl == "mma_fp16acc_m16n32_k32_a16_vs" ||
                          args.impl == "mma_fp16acc_m16n32_k32_b16_vs" ||
                          args.impl == "mma_fp16acc_m16n32_k32_4x4_vs" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32" ||
                          args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16" ||
                          args.impl == "mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16" ||
                          args.impl == "mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16" ||
                          args.impl == "mma_fp16acc_m32n32_k64_4x2" ||
                          args.impl == "mma_fp16acc_m32n32_k64_4x2_s8" ||
                          args.impl == "mma_fp16acc_m32n32_k128_4x2_s8" ||
                          args.impl == "wgmma_m64n8k16_sm90a" ||
                          args.impl == "wgmma_m64n64k16_sm90a" ||
                          args.impl == "wgmma_m64n64k16_db_sm90a" ||
                          args.impl == "cublas_gemmex_fp16acc" ||
                          args.impl == "cublaslt_fp16acc");

  if (args.check) {
    hC_ref.resize(static_cast<size_t>(M) * N, 0.0f);
    std::cout << "[check] running CPU reference..." << std::endl;
    if (use_fp16_inputs) {
      std::vector<float> hA_ref_in(hA.size());
      std::vector<float> hB_ref_in(hB.size());
      for (size_t i = 0; i < hA.size(); ++i) {
        hA_ref_in[i] = __half2float(__float2half(hA[i]));
      }
      for (size_t i = 0; i < hB.size(); ++i) {
        hB_ref_in[i] = __half2float(__float2half(hB[i]));
      }
      cpu_gemm_ref(hA_ref_in.data(), hB_ref_in.data(), hC_ref.data(), M, N, K);
    } else {
      cpu_gemm_ref(hA.data(), hB.data(), hC_ref.data(), M, N, K);
    }
  }

  // Device buffers
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  half  *dA16 = nullptr, *dB16 = nullptr;

  size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

  size_t bytesA16 = static_cast<size_t>(M) * K * sizeof(half);
  size_t bytesB16 = static_cast<size_t>(K) * N * sizeof(half);

  if (use_fp16_inputs) {
    std::vector<half> hA16(static_cast<size_t>(M) * K);
    std::vector<half> hB16(static_cast<size_t>(K) * N);

    for (size_t i = 0; i < hA.size(); ++i) {
      hA16[i] = __float2half(hA[i]);
    }
    for (size_t i = 0; i < hB.size(); ++i) {
      hB16[i] = __float2half(hB[i]);
    }

    CHECK_CUDA(cudaMalloc(&dA16, bytesA16));
    CHECK_CUDA(cudaMalloc(&dB16, bytesB16));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA16, hA16.data(), bytesA16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB16, hB16.data(), bytesB16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));
  } else {
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));
  }

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // launcher 分发函数
  auto launch_selected = [&](cudaStream_t stream) {
    if (args.impl == "naive") {
      launch_gemm_naive(dA, dB, dC, M, N, K, stream);
    } else if (args.impl == "tiled") {
      launch_gemm_tiled(dA, dB, dC, M, N, K, stream);
    } else if (args.impl == "tiled_rb1x4") {
      launch_gemm_tiled_rb1x4(dA, dB, dC, M, N, K, stream);
    } else if (args.impl == "tiled_rb2x4") {
      launch_gemm_tiled_rb2x4(dA, dB, dC, M, N, K, stream);
    } else if (args.impl == "thread_tiled_1d") {
      launch_gemm_thread_tiled_1d(dA, dB, dC, M, N, K, stream);
    } else if (args.impl == "cublas") {
      launch_gemm_cublas_rowmajor(dA, dB, dC, M, N, K, stream);
    } else if (args.impl == "cublaslt") {
      launch_gemm_cublaslt_rowmajor(dA, dB, dC, M, N, K, stream);
    } else if (args.impl == "tiled_fp16acc") {
      launch_gemm_tiled_fp16acc(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "tiled_fp16acc_rb1x4") {
      launch_gemm_tiled_fp16acc_rb1x4(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "tiled_fp16acc_rb2x4") {
      launch_gemm_tiled_fp16acc_rb2x4(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc") {
      launch_gemm_wmma_fp16acc(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged") {
      launch_gemm_wmma_fp16acc_staged(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_db") {
      launch_gemm_wmma_fp16acc_staged_db(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync") {
      launch_gemm_wmma_fp16acc_staged_cpasync(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_bcol") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_bcol(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_4x2") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_4x2(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_split") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_split(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_skew16") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA16_B8") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B8(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA8_B16") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA24_B16") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA24_B16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA8_B24") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B24(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA8_B32") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA8_B32(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA16_B32") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA16_B32(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_skewA24_B32") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_skewA24_B32(dA16, dB16, dC, M, N, K, stream);

    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_4x4_skew16") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_4x4_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k32_4x8_skew16") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k32_4x8_skew16(dA16, dB16, dC, M, N, K, stream);

    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64_skew16") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle2d_n4") {
      launch_gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle2d_n4(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc") {
      launch_gemm_mma_fp16acc(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32") {
      launch_gemm_mma_fp16acc_m16n32(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_k32") {
      launch_gemm_mma_fp16acc_m16n32_k32(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_k32_4x2") {
      launch_gemm_mma_fp16acc_m16n32_k32_4x2(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_k32_4x4") {
      launch_gemm_mma_fp16acc_m16n32_k32_4x4(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k32_4x2") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k32_4x2(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_k32_vs") {
      launch_gemm_mma_fp16acc_m16n32_k32_vs(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_k32_s16_vs") {
      launch_gemm_mma_fp16acc_m16n32_k32_s16_vs(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_k32_a16_vs") {
      launch_gemm_mma_fp16acc_m16n32_k32_a16_vs(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_k32_b16_vs") {
      launch_gemm_mma_fp16acc_m16n32_k32_b16_vs(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_k32_4x4_vs") {
      launch_gemm_mma_fp16acc_m16n32_k32_4x4_vs(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16") {
      launch_gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16") {
      launch_gemm_mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16") {
      launch_gemm_mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m32n32_k64_4x2") {
      launch_gemm_mma_fp16acc_m32n32_k64_4x2(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m32n32_k64_4x2_s8") {
      launch_gemm_mma_fp16acc_m32n32_k64_4x2_s8(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "mma_fp16acc_m32n32_k128_4x2_s8") {
      launch_gemm_mma_fp16acc_m32n32_k128_4x2_s8(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wgmma_m64n8k16_sm90a") {
      launch_gemm_wgmma_m64n8k16_sm90a(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wgmma_m64n64k16_sm90a") {
      launch_gemm_wgmma_m64n64k16_sm90a(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "wgmma_m64n64k16_db_sm90a") {
      launch_gemm_wgmma_m64n64k16_db_sm90a(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "cublas_gemmex_fp16acc") {
      launch_gemm_cublas_gemmex_fp16acc_rowmajor(dA16, dB16, dC, M, N, K, stream);
    } else if (args.impl == "cublaslt_fp16acc") {
      launch_gemm_cublaslt_fp16acc_rowmajor(dA16, dB16, dC, M, N, K, stream);
    } else {
      std::cerr << "Unknown impl: " << args.impl << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };

  // Correctness check (single run)
  if (args.check) {
    launch_selected(stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    float atol = use_fp16_inputs ? 2e-2f : 1e-3f;
    float rtol = use_fp16_inputs ? 2e-2f : 1e-3f;
    auto chk = check_allclose(hC, hC_ref, atol, rtol);

    if (!chk.ok) {
      std::cerr << "[check] FAILED"
                << " max_abs_err=" << chk.max_abs_err
                << " idx=" << chk.max_idx
                << " got=" << chk.got
                << " ref=" << chk.ref << std::endl;

      if (dA)   CHECK_CUDA(cudaFree(dA));
      if (dB)   CHECK_CUDA(cudaFree(dB));
      if (dA16) CHECK_CUDA(cudaFree(dA16));
      if (dB16) CHECK_CUDA(cudaFree(dB16));
      if (dC)   CHECK_CUDA(cudaFree(dC));
      CHECK_CUDA(cudaStreamDestroy(stream));
      return EXIT_FAILURE;
    } else {
      std::cout << "[check] PASS"
                << " max_abs_err=" << chk.max_abs_err << std::endl;
    }
  }

  // Warmup
  for (int i = 0; i < args.warmup; ++i) {
    launch_selected(stream);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timing via CUDA events
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> times_ms;
  times_ms.reserve(args.repeat);

  for (int i = 0; i < args.repeat; ++i) {
    CHECK_CUDA(cudaEventRecord(start, stream));
    launch_selected(stream);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    times_ms.push_back(ms);
  }

  auto stats = summarize_ms(times_ms);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "[time] min=" << stats.min_ms
            << " ms, median=" << stats.median_ms
            << " ms, avg=" << stats.avg_ms << " ms\n";
  std::cout << "[perf] median=" << gflops_gemm(M, N, K, stats.median_ms)
            << " GFLOP/s\n";

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  if (dA)   CHECK_CUDA(cudaFree(dA));
  if (dB)   CHECK_CUDA(cudaFree(dB));
  if (dA16) CHECK_CUDA(cudaFree(dA16));
  if (dB16) CHECK_CUDA(cudaFree(dB16));
  if (dC)   CHECK_CUDA(cudaFree(dC));

  CHECK_CUDA(cudaStreamDestroy(stream));

  return 0;
}
