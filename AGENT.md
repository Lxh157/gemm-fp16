# Agent Notes

## Project Overview

This repository is a CUDA GEMM optimization lab focused on FP32, FP16, WMMA, Tensor Core, and profiling-driven iteration.

The benchmark entrypoint is `src/main_bench.cu`, which dispatches kernels by `--impl`. The build target is `bench_gemm`.

Current main direction:

- Phase 1: RTX 4060 Laptop / WSL2, full learning chain from naive FP32 to tiled, register blocking, FP16 input, WMMA, staged WMMA, and cp.async.
- Phase 2: RTX 4090 / CUDA 11.8, Tensor Core mainline optimization. The current best custom kernel is:
  - file: `src/gemm_wmma_fp16acc_staged_cpasync_k64_4x4_skew16.cu`
  - impl: `wmma_fp16acc_staged_cpasync_k64_4x4_skew16`

The project is intentionally experimental. Some implementations under `src/fail/` are kept as failed or superseded variants for comparison and should not be cleaned up casually.

## Environment

Primary target architecture is Ada / SM89.

`CMakeLists.txt` currently sets:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

Phase 2 server environment usually uses CUDA 11.8:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Build

From repository root:

```bash
cmake -S . -B build
cmake --build build -j
```

Equivalent existing workflow:

```bash
cd build
cmake ..
make -j
```

Build output:

```text
build/bench_gemm
```

## Run Single Benchmark

Example:

```bash
./build/bench_gemm --impl wmma_fp16acc_staged_cpasync_k64_4x4_skew16 --M 1024 --N 1024 --K 1024 --warmup 3 --repeat 10
```

For larger sizes, correctness checking can be skipped to avoid slow CPU reference:

```bash
./build/bench_gemm --impl wmma_fp16acc_staged_cpasync_k64_4x4_skew16 --M 4096 --N 4096 --K 4096 --warmup 3 --repeat 10 --no-check
```

Output format of interest:

```text
[check] PASS max_abs_err=...
[time] min=... ms, median=... ms, avg=... ms
[perf] median=... GFLOP/s
```

Use median GFLOP/s as the stable performance metric.

## Batch Benchmark

Default batch script:

```bash
PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
```

Useful overrides:

```bash
PROFILE_SET=phase2_4090_tc SIZES_OVERRIDE="1024 2048 4096" bash scripts/run_bench.sh
CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
CUDA_VISIBLE_DEVICES=1 CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
```

Raw outputs are written under `results/raw/`. This directory is ignored except for `.gitkeep`.

## Plotting

```bash
python3 scripts/plot.py
```

The plotting script reads the latest matching raw benchmark files from `results/raw/` and writes PNGs under `results/plots/`.

Use:

```bash
python3 -m pip install --user matplotlib
```

if plotting dependencies are missing.

## Profiling

Nsight Compute runs perturb timing. Do not use NCU-reported runtime or GFLOP/s as final performance conclusions. Use NCU for stall and pipeline diagnosis only.

Typical command:

```bash
ncu --set full --target-processes all --force-overwrite \
  ./build/bench_gemm --impl wmma_fp16acc_staged_cpasync_k64_4x4_skew16 \
  --M 2048 --N 2048 --K 2048 --warmup 0 --repeat 1 --no-check
```

Profile artifacts belong under:

```text
profiles/ncu/
profiles/nsys/
```

These directories are ignored except for `.gitkeep`.

## Source Layout

Important files:

- `src/main_bench.cu`: argument parsing, CPU reference, FP32/FP16 input setup, launcher dispatch, timing.
- `src/utils.cuh`: CUDA error macro, random initialization, CPU reference GEMM, allclose check, GFLOP/s and timing stats.
- `src/gemm_*.cu`: custom GEMM implementations.
- `src/cublas*.cu`: cuBLAS and cuBLASLt baselines.
- `scripts/run_bench.sh`: batch runner.
- `scripts/plot.py`: raw result parser and plot generator.
- `scripts/collect_env.sh`: basic environment capture.

## Adding A New Kernel

When adding a new implementation:

1. Add a new `src/*.cu` file with a launcher matching the local pattern:

```cpp
void launch_gemm_<name>(const half* dA, const half* dB, float* dC,
                        int M, int N, int K, cudaStream_t stream);
```

or for FP32 kernels:

```cpp
void launch_gemm_<name>(const float* dA, const float* dB, float* dC,
                        int M, int N, int K, cudaStream_t stream);
```

2. Add the file to `add_executable(bench_gemm ...)` in `CMakeLists.txt`.
3. Add a forward declaration in `src/main_bench.cu`.
4. Add the impl string to the `--impl` validation list.
5. Add the impl string to `use_fp16_inputs` if it consumes half inputs.
6. Add a dispatch branch in `launch_selected`.
7. Add it to `scripts/run_bench.sh` if it should be part of a batch profile set.
8. Add it to `scripts/plot.py` if it should appear in plots.

Keep the impl string, launcher name, source filename, batch script entry, and plot entry consistent.

## Correctness Policy

The benchmark compares against CPU reference by default.

Current tolerances:

- FP32 kernels: `atol=1e-3`, `rtol=1e-3`
- FP16-input kernels: `atol=2e-2`, `rtol=2e-2`

For large benchmark sweeps, use `--no-check` or set `CHECK_MAX_SIZE` in `scripts/run_bench.sh` so the CPU reference does not dominate runtime.

Many WMMA kernels require dimensions to be multiples of tile sizes. Preserve explicit guard checks in launchers and print clear errors when shape requirements are violated.

## Current Technical Conclusions

The current best custom 4090 kernel is `wmma_fp16acc_staged_cpasync_k64_4x4_skew16`.

Working conclusions from the README:

- `skew16` improved the old `k32` path through better readiness, barrier behavior, and memory feed.
- `k64` helps small shapes mainly by reducing K-loop, synchronization, and control overhead.
- `k64_4x4_skew16` is strongest for larger shapes because of tile/work organization and resource rebalance, not because every low-level shared-memory metric becomes cleaner.
- The gap to cuBLASLt is not likely to be solved by blind pitch or tile-shape sweeps alone. The vendor kernel appears to operate in a deeper tensor-pipeline-dominated regime.

Next work should focus on source-level stall diagnosis around `wmma_fp16acc_staged_cpasync_k64_4x4_skew16` and on moving from WMMA toward lower-level MMA / ldmatrix data paths.

## Repository Hygiene

- Do not commit build outputs, raw logs, NCU/NSYS reports, or generated caches.
- `results/plots/*.png` are currently present and may be used in README documentation.
- Avoid broad cleanup of `src/fail/`; these files document explored variants.
- Keep changes scoped. This repository values benchmark comparability, so do not silently change timing, correctness tolerances, input generation, or output parsing formats.

