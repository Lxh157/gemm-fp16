#include "utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

// 来自 gemm_naive.cu 的声明
void launch_gemm_naive(const float* dA, const float* dB, float* dC,
                       int M, int N, int K,
                       cudaStream_t stream);
// 来自 gemm_tiled.cu 的声明             
void launch_gemm_tiled(const float* dA, const float* dB, float* dC,
                       int M, int N, int K,
                       cudaStream_t stream);

// 非严格参数解析（Day 1 够用）
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
        if (a.impl != "naive" && a.impl != "tiled") {
          std::cerr << "Invalid --impl: " << a.impl
                    << " (expected naive or tiled)" << std::endl;
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

  std::cout << "=== bench_gemm (" << args.impl << ", FP32) ===\n";
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

  if (args.check) {
    hC_ref.resize(static_cast<size_t>(M) * N, 0.0f);
    std::cout << "[check] running CPU reference..." << std::endl;
    cpu_gemm_ref(hA.data(), hB.data(), hC_ref.data(), M, N, K);
  }

  // Device buffers
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
  size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
  size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

  CHECK_CUDA(cudaMalloc(&dA, bytesA));
  CHECK_CUDA(cudaMalloc(&dB, bytesB));
  CHECK_CUDA(cudaMalloc(&dC, bytesC));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC, 0, bytesC));
    // launcher 分发函数
  auto launch_selected = [&](cudaStream_t stream = nullptr) {
    if (args.impl == "naive") {
      launch_gemm_naive(dA, dB, dC, M, N, K, stream);
    } else if (args.impl == "tiled") {
      launch_gemm_tiled(dA, dB, dC, M, N, K, stream);
    } else {
      std::cerr << "Unknown impl: " << args.impl << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };

  // Correctness check (single run)
  if (args.check) {
    launch_selected(nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    auto chk = check_allclose(hC, hC_ref, 1e-3f, 1e-3f);

    if (!chk.ok) {
      std::cerr << "[check] FAILED"
                << " max_abs_err=" << chk.max_abs_err
                << " idx=" << chk.max_idx
                << " got=" << chk.got
                << " ref=" << chk.ref << std::endl;
      CHECK_CUDA(cudaFree(dA));
      CHECK_CUDA(cudaFree(dB));
      CHECK_CUDA(cudaFree(dC));
      return EXIT_FAILURE;
    } else {
      std::cout << "[check] PASS"
                << " max_abs_err=" << chk.max_abs_err << std::endl;
    }
  }

  // Warmup
  for (int i = 0; i < args.warmup; ++i) {
    launch_selected(nullptr);
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
    CHECK_CUDA(cudaEventRecord(start));
    launch_selected(nullptr);
    CHECK_CUDA(cudaEventRecord(stop));
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
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));

  return 0;
}