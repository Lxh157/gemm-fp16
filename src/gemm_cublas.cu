// src/gemm_cublas.cu
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <mutex>

#define CHECK_CUBLAS(call) do {                                  \
  cublasStatus_t st = (call);                                    \
  if (st != CUBLAS_STATUS_SUCCESS) {                             \
    std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",      \
                 __FILE__, __LINE__, (int)st);                   \
    std::exit(1);                                                \
  }                                                              \
} while(0)

namespace {
cublasHandle_t g_handle = nullptr;
std::once_flag g_init_flag;

void init_handle_once() {
  CHECK_CUBLAS(cublasCreate(&g_handle));
  // 为了和 FP32 kernel 在数值上更可比，先用默认 FP32 路径：
  // 若想用 TF32 拉满 cuBLAS 性能，可改成：
  // cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  CHECK_CUBLAS(cublasSetMathMode(g_handle, CUBLAS_DEFAULT_MATH));
}

} // namespace

// row-major: C(MxN) = A(MxK) * B(KxN)
// uses column-major trick: C^T(NxM) = B^T(NxK) * A^T(KxM)
void launch_gemm_cublas_rowmajor(const float* A, const float* B, float* C,
                         int M, int N, int K, cudaStream_t stream) {
  std::call_once(g_init_flag, init_handle_once);
  CHECK_CUBLAS(cublasSetStream(g_handle, stream));

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // Column-major GEMM: C_col(NxM) = B_col(NxK) * A_col(KxM)
  // lda = leading dimension of B_col = N
  // ldb = leading dimension of A_col = K
  // ldc = leading dimension of C_col = N
  CHECK_CUBLAS(cublasSgemm(g_handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           /*m=*/N, /*n=*/M, /*k=*/K,
                           &alpha,
                           /*A=*/B, /*lda=*/N,
                           /*B=*/A, /*ldb=*/K,
                           &beta,
                           /*C=*/C, /*ldc=*/N));
}