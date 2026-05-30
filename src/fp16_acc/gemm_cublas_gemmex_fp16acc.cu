// src/gemm_cublas_gemmex_fp16acc.cu
#include "utils.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do {                                   \
  cublasStatus_t st = (call);                                     \
  if (st != CUBLAS_STATUS_SUCCESS) {                              \
    std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",       \
                 __FILE__, __LINE__, (int)st);                    \
    std::exit(EXIT_FAILURE);                                      \
  }                                                               \
} while(0)
#endif

// Row-major mapping:
// C_row(M,N) = A_row(M,K) * B_row(K,N)
// Treat row-major buffers as transposed col-major:
//   A_row memory == A_col(K,M)
//   B_row memory == B_col(N,K)
//   C_row memory == C_col(N,M)
// So compute in cuBLAS col-major world:
//   C_col(N,M) = B_col(N,K) * A_col(K,M)

void launch_gemm_cublas_gemmex_fp16acc_rowmajor(const half* A, const half* B, float* C,
                                                int M, int N, int K,
                                                cudaStream_t stream) {
  static cublasHandle_t handle = nullptr;
  static bool inited = false;
  if (!inited) {
    CHECK_CUBLAS(cublasCreate(&handle));
    // 允许 Tensor Core / 默认 Tensor Op 路径
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    inited = true;
  }

  CHECK_CUBLAS(cublasSetStream(handle, stream));

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // mapped col-major dims
  const int m = N;
  const int n = M;
  const int k = K;

  // C_col(N,M) = B_col(N,K) * A_col(K,M)
  CHECK_CUBLAS(cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      &alpha,
      B, CUDA_R_16F, m,   // A operand in cuBLAS API = B buffer
      A, CUDA_R_16F, k,   // B operand in cuBLAS API = A buffer
      &beta,
      C, CUDA_R_32F, m,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}