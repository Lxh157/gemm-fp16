#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "   \
                << cudaGetErrorString(err__) << std::endl;                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

inline void fill_random(std::vector<float>& v, uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = dist(rng);
}

// Row-major GEMM: C[M, N] = A[M, K] * B[K, N]
inline void cpu_gemm_ref(const float* A, const float* B, float* C,
                         int M, int N, int K) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}

struct CheckResult {
  bool ok;
  float max_abs_err;
  int max_idx;
  float got;
  float ref;
};

inline CheckResult check_allclose(const std::vector<float>& got,
                                  const std::vector<float>& ref,
                                  float atol = 1e-3f,
                                  float rtol = 1e-3f) {
  if (got.size() != ref.size()) {
    return {false, std::numeric_limits<float>::infinity(), -1, 0.0f, 0.0f};
  }

  float max_abs_err = 0.0f;
  int max_idx = -1;
  for (size_t i = 0; i < got.size(); ++i) {
    float abs_err = std::fabs(got[i] - ref[i]);
    float tol = atol + rtol * std::fabs(ref[i]);
    if (abs_err > max_abs_err) {
      max_abs_err = abs_err;
      max_idx = static_cast<int>(i);
    }
    if (abs_err > tol) {
      return {false, max_abs_err, max_idx, got[i], ref[i]};
    }
  }

  float got_v = (max_idx >= 0) ? got[max_idx] : 0.0f;
  float ref_v = (max_idx >= 0) ? ref[max_idx] : 0.0f;
  return {true, max_abs_err, max_idx, got_v, ref_v};
}

inline double gflops_gemm(int M, int N, int K, double ms) {
  // GEMM FLOPs ~= 2*M*N*K
  double flops = 2.0 * static_cast<double>(M) * N * K;
  double sec = ms / 1000.0;
  return flops / sec / 1e9;
}

struct BenchStats {
  float min_ms = 0.0f;
  float median_ms = 0.0f;
  float avg_ms = 0.0f;
};

inline BenchStats summarize_ms(std::vector<float> ms) {
  BenchStats s{};
  if (ms.empty()) return s;

  std::sort(ms.begin(), ms.end());
  s.min_ms = ms.front();

  size_t n = ms.size();
  if (n % 2 == 0) {
    s.median_ms = 0.5f * (ms[n / 2 - 1] + ms[n / 2]);
  } else {
    s.median_ms = ms[n / 2];
  }

  double sum = 0.0;
  for (float x : ms) sum += x;
  s.avg_ms = static_cast<float>(sum / ms.size());
  return s;
}