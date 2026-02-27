// src/cublaslt_baseline.cu
#include "utils.cuh"

#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdio>
#include <cstdlib>

#ifndef CHECK_CUBLASLT
#define CHECK_CUBLASLT(call) do {                          \
  cublasStatus_t st = (call);                              \
  if (st != CUBLAS_STATUS_SUCCESS) {                       \
    std::fprintf(stderr, "cuBLASLt error %s:%d: %d\n",     \
      __FILE__, __LINE__, (int)st);                        \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
} while(0)
#endif

// Row-major: C[M,N] = A[M,K] * B[K,N]
// Mapping trick (same spirit as你 cublas row-major->col-major 映射):
// Treat row-major buffers as transposed col-major:
//   A_row(M,K) memory == A_col(K,M)
//   B_row(K,N) memory == B_col(N,K)
//   C_row(M,N) memory == C_col(N,M)
// Then compute in col-major space:
//   C_col(N,M) = B_col(N,K) * A_col(K,M)
// So in Lt call we pass: A_operand = B, B_operand = A, with dims (m=N, n=M, k=K).

void launch_gemm_cublaslt_rowmajor(const float* A, const float* B, float* C,
                                  int M, int N, int K, cudaStream_t stream) {
  struct LtCache {
    cublasLtHandle_t handle = nullptr;
    bool inited = false;

    int lastM = -1, lastN = -1, lastK = -1;

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t aLayout = nullptr, bLayout = nullptr;
    cublasLtMatrixLayout_t cLayout = nullptr, dLayout = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    cublasLtMatmulHeuristicResult_t heur{};
    bool hasAlgo = false;

    void* workspace = nullptr;
    size_t workspaceBytes = 64 * 1024 * 1024; // 64MB 常驻

    void destroy_plan() {
      if (workspace) { CHECK_CUDA(cudaFree(workspace)); workspace = nullptr; }
      if (pref) { CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(pref)); pref = nullptr; }
      if (aLayout) { CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(aLayout)); aLayout = nullptr; }
      if (bLayout) { CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(bLayout)); bLayout = nullptr; }
      if (cLayout) { CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(cLayout)); cLayout = nullptr; }
      if (dLayout) { CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(dLayout)); dLayout = nullptr; }
      if (opDesc) { CHECK_CUBLASLT(cublasLtMatmulDescDestroy(opDesc)); opDesc = nullptr; }
      hasAlgo = false;
    }

    void ensure_handle() {
      if (!inited) {
        CHECK_CUBLASLT(cublasLtCreate(&handle));
        inited = true;
      }
    }

    void build_plan_if_needed(int M_, int N_, int K_) {
      if (M_ == lastM && N_ == lastN && K_ == lastK && hasAlgo) return;

      // rebuild
      destroy_plan();
      lastM = M_; lastN = N_; lastK = K_;

      ensure_handle();

      // row-major -> col-major mapping:
      // compute C_col(N,M) = B_col(N,K) * A_col(K,M)
      const int m = N_;
      const int n = M_;
      const int k = K_;

      // 允许 TF32 tensor op（对齐 cublas 默认口径）
      CHECK_CUBLASLT(cublasLtMatmulDescCreate(
          &opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

      cublasOperation_t opN = CUBLAS_OP_N;
      CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
          opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
      CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
          opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

      // Lt operands:
      // A_operand = B buffer as B_col (m x k), ld=m
      // B_operand = A buffer as A_col (k x n), ld=k
      CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_32F, m, k, m));
      CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_32F, k, n, k));
      CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32F, m, n, m));
      CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&dLayout, CUDA_R_32F, m, n, m));

      CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&pref));
      CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
          pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspaceBytes, sizeof(workspaceBytes)));

      if (workspaceBytes > 0) CHECK_CUDA(cudaMalloc(&workspace, workspaceBytes));

      int returned = 0;
      CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
          handle, opDesc,
          aLayout, bLayout, cLayout, dLayout,
          pref, 1, &heur, &returned));

      if (returned == 0) {
        // fallback: 0 workspace
        if (workspace) { CHECK_CUDA(cudaFree(workspace)); workspace = nullptr; }
        size_t zero = 0;
        CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &zero, sizeof(zero)));

        CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
            handle, opDesc, aLayout, bLayout, cLayout, dLayout,
            pref, 1, &heur, &returned));
        workspaceBytes = 0;
      }

      if (returned == 0) {
        std::fprintf(stderr, "cuBLASLt: no heuristic algorithm found.\n");
        std::exit(EXIT_FAILURE);
      }

      hasAlgo = true;
    }
  };

  static LtCache cache;
  cache.build_plan_if_needed(M, N, K);

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  // Execute: D = alpha * A_operand * B_operand + beta * C
  // Here: C_col = B_col * A_col  => pass (B, aLayout) then (A, bLayout)
  CHECK_CUBLASLT(cublasLtMatmul(
      cache.handle,
      cache.opDesc,
      &alpha,
      B, cache.aLayout,
      A, cache.bLayout,
      &beta,
      C, cache.cLayout,
      C, cache.dLayout,
      &cache.heur.algo,
      cache.workspace,
      cache.workspaceBytes,
      stream));
}