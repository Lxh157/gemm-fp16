# Agent Notes

## Current H100 Mainline

This checkout is currently used for the H100 / SM90a WGMMA route. The current best custom H100 kernel is:

```text
file:
src/fp16_wgmma/gemm_wgmma_m64n128k32_tma_ab_mbar.cu

impl:
wgmma_m64n128k32_tma_ab_mbar
```

Use same-device, same-run comparisons against `cublaslt_fp16acc`. On H100, use `PROFILE_SET=h100_wgmma` for batch runs and `BEST_IMPL=wgmma_m64n128k32_tma_ab_mbar` for NCU comparison. Nsight Compute is for bottleneck diagnosis only; benchmark timing should come from `bench_gemm` runs without NCU instrumentation.

Current H100 evidence: the mbar variant is only a small step over `wgmma_m64n128k32_tma_ab`; it reduces synchronization overhead by using a block-wide TMA mbarrier and grouped WGMMA commit. Failed or slower routes in this round included wider single-WG N tiles, 2-WG CTA sharing, K64 staging, ordinary B global-load packing, split TMA wait, fence-once, and direct wide WGMMA without the correct B core-matrix layout. The next substantial route should focus on correct B shared-memory layout/swizzle for wider WGMMA instructions such as m64n16/m64n64.

## Project Overview

This repository is a CUDA GEMM optimization lab focused on FP32, FP16-input FP32-accumulate, WMMA, inline MMA, Tensor Core, and profiling-driven iteration.

The benchmark entrypoint is `src/main_bench.cu`, which dispatches kernels by `--impl`. The build target is `bench_gemm`.

Historical main direction was FP16-input FP32-accumulate Tensor Core optimization using inline PTX MMA, `ldmatrix`, and `cp.async`. The active H100 direction is WGMMA plus TMA under `src/fp16_wgmma/`.

Current best custom kernel on the local RTX 4060 Laptop:

```text
file:
src/fp16_mma/gemm_mma_fp16acc_m16n32_k32_vs.cu

impl:
mma_fp16acc_m16n32_k32_vs
```

Latest local 4060 comparison data is in:

```text
results/table/bench_phase2_4090_tc_20260605_012329.csv
```

At 4096 in the latest local 4060 comparison, `mma_fp16acc_m16n32_k32_vs` reaches about `15.7 TFLOP/s`, roughly `88.2%` of the same-run `cublaslt_fp16acc` baseline. Do not compare this directly with historical RTX 4090 results.

The project is intentionally experimental. Some implementations under `src/fail/` are kept as failed or superseded variants for comparison and should not be cleaned up casually.

## Collaboration Workflow

The active development workflow is a direct local optimization loop:

1. Read the latest same-device benchmark and NCU data.
2. Choose one focused optimization direction.
3. Add and wire a comparison kernel.
4. Build, run correctness checks, benchmark, and profile locally on the RTX 4060 Laptop.
5. Analyze the result, commit the completed round, and push it.
6. Start the next round from the new evidence.

Use only same-device, same-run comparisons for performance conclusions. RTX 4090 results remain useful historical references but must not be compared numerically with local RTX 4060 runs.

When reporting new kernel changes, the preferred response shape is:

```text
已新增并接入 ...
新增文件:
...
已更新:
...
新 impl:
...
对照变量:
...
服务器上先单点:
...
通过后跑全量:
...
```

Keep kernel filenames reasonably short. Avoid endlessly appending long suffixes. Preferred short suffixes:

```text
vs  = vector store
sls = skip last sync
bpf = B fragment prefetch
nc  = no check
```

## Environment

Primary target architecture for this branch is Hopper / SM90a.

`CMakeLists.txt` currently sets:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 90a)
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

Build output:

```text
build/bench_gemm
```

## Run Single Benchmark

Current best local custom kernel:

```bash
./build/bench_gemm \
  --impl mma_fp16acc_m16n32_k32_vs \
  --M 4096 --N 4096 --K 4096 \
  --warmup 3 --repeat 10 --no-check
```

cuBLASLt baseline:

```bash
./build/bench_gemm \
  --impl cublaslt_fp16acc \
  --M 4096 --N 4096 --K 4096 \
  --warmup 3 --repeat 10 --no-check
```

Output format of interest:

```text
[check] PASS max_abs_err=...
[time] min=... ms, median=... ms, avg=... ms
[perf] median=... GFLOP/s
```

Use median GFLOP/s as the stable performance metric.

## Batch Benchmark

Default Phase 2 batch script:

```bash
CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
```

Useful overrides:

```bash
PROFILE_SET=phase2_4090_tc SIZES_OVERRIDE="1024 2048 4096" bash scripts/run_bench.sh
PROFILE_SET=phase2_4090_tc IMPLS_OVERRIDE="mma_fp16acc_m16n32_k32_vs cublaslt_fp16acc" bash scripts/run_bench.sh
CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
CUDA_VISIBLE_DEVICES=1 CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
```

Raw outputs are written under `results/raw/`.

CSV extraction:

```bash
python3 scripts/raw_to_csv.py
```

CSV outputs are written under `results/table/`.

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

Current Phase 2 MMA plots:

```text
results/plots/gflops_phase2_4090_tc_mma.png
results/plots/rel_to_cublaslt_phase2_4090_tc_mma.png
```

## One-Command Server Workflow

`scripts/run_all_scripts.sh` automatically chooses a free RTX 4090 and runs:

```text
cmake configure
cmake build
benchmark sweep
raw_to_csv
plot
NCU compare
```

Default:

```bash
bash scripts/run_all_scripts.sh
```

Skip NCU:

```bash
RUN_NCU=0 bash scripts/run_all_scripts.sh
```

Choose NCU sizes:

```bash
NCU_SIZES="2048 4096" bash scripts/run_all_scripts.sh
```

## Profiling

Nsight Compute runs perturb timing. Do not use NCU-reported runtime or GFLOP/s as final performance conclusions. Use NCU for stall and pipeline diagnosis only.

Use the text-only script:

```bash
bash scripts/run_ncu_compare.sh
```

It compares by default:

```text
mma_fp16acc_m16n32_k32_vs
cublaslt_fp16acc
```

Default output:

```text
profiles/ncu/ncu_mma_best/*.txt
profiles/ncu/ncu_cublaslt/*.txt
```

The script intentionally does not pass `--export`, so it should not generate `.ncu-rep` files.

Common overrides:

```bash
NCU_SIZES=1024 bash scripts/run_ncu_compare.sh
MMA_BEST_IMPL=mma_fp16acc_m16n32_k32_vs NCU_SIZES=1024 bash scripts/run_ncu_compare.sh
NCU_SIZES="2048 3072 4096" bash scripts/run_ncu_compare.sh
NCU_SET=full NCU_PAGE=raw bash scripts/run_ncu_compare.sh
NCU_BIN=/path/to/ncu bash scripts/run_ncu_compare.sh
```

On the local WSL + RTX 4060 setup, full NCU at 4096 has failed before. Use 1024 for routine diagnosis unless a larger profile is specifically needed.

On shared servers, Nsight Compute may fail on `/tmp/nsight-compute-lock`. The script checks the lock owner and reports it. Do not delete another user's lock without confirming it is stale or getting permission.

First NCU comparison should focus on:

- SM / Tensor Core utilization
- achieved occupancy / active warps
- eligible warps per scheduler
- warp stall reasons
- shared memory throughput, bank conflict, ldmatrix pressure
- global memory and cp.async behavior
- instruction mix

## Source Layout

Important files and directories:

- `src/main_bench.cu`: argument parsing, CPU reference, FP32/FP16 input setup, launcher dispatch, timing.
- `src/utils.cuh`: CUDA error macro, random initialization, CPU reference GEMM, allclose check, GFLOP/s and timing stats.
- `src/fp32/`: FP32 custom kernels and FP32 cuBLAS/cuBLASLt baselines.
- `src/fp16_acc/`: non-Tensor-Core FP16-input FP32-accumulate kernels and cuBLAS GEMMEx baseline.
- `src/fp16_wmma/`: WMMA Tensor Core kernels.
- `src/fp16_mma/`: inline MMA / ldmatrix Tensor Core kernels.
- `src/fail/`: failed or superseded WMMA variants kept for comparison.
- `src/cublaslt_fp16acc.cu`: cuBLASLt FP16-input FP32-accumulate baseline.
- `scripts/run_bench.sh`: batch benchmark runner.
- `scripts/raw_to_csv.py`: parse latest raw result into CSV.
- `scripts/plot.py`: raw result parser and plot generator.
- `scripts/run_ncu_compare.sh`: text-only NCU comparison.
- `scripts/run_all_scripts.sh`: build + bench + csv + plot + NCU orchestration.
- `scripts/collect_env.sh`: basic environment capture.

## Adding A New Kernel

When adding a new implementation:

1. Add a new `.cu` file under the appropriate source directory:

```text
src/fp32/
src/fp16_acc/
src/fp16_wmma/
src/fp16_mma/
```

2. Add the file to `add_executable(bench_gemm ...)` in `CMakeLists.txt`.
3. Add a forward declaration in `src/main_bench.cu`.
4. Add the impl string to the `--impl` validation list.
5. Add the impl string to `use_fp16_inputs` if it consumes half inputs.
6. Add a dispatch branch in `main_bench.cu`.
7. Add it to `scripts/run_bench.sh` if it should be part of a batch profile set.
8. Add it to `scripts/plot.py` if it should appear in plots.

Keep the impl string, launcher name, source filename, batch script entry, and plot entry consistent.

## Correctness Policy

The benchmark compares against CPU reference by default.

Current tolerances:

- FP32 kernels: `atol=1e-3`, `rtol=1e-3`
- FP16-input kernels: `atol=2e-2`, `rtol=2e-2`

For large benchmark sweeps, use `--no-check` or set `CHECK_MAX_SIZE` in `scripts/run_bench.sh` so the CPU reference does not dominate runtime.

Many Tensor Core kernels require dimensions to be multiples of tile sizes. Preserve explicit guard checks in launchers and print clear errors when shape requirements are violated.

## Current Technical Conclusions

Current best local 4060 kernel:

```text
mma_fp16acc_m16n32_k32_vs
```

Working conclusions:

- Moving from WMMA to inline MMA / ldmatrix gave a major improvement.
- K32 is the current local 4060 mainline depth; its smaller shared-memory footprint beats the previous K64 mainline across 512-4096.
- The historical 4090 best remains the K64 `skew16` vector-store/skip-last-sync variant until K32 is validated there.
- `cp.async.cg` is better than the tested `cp.async.ca` variant.
- `float2` vectorized C store was a clear win.
- Skipping the final unnecessary sync is a small but stable large-shape win.
- `nocheck` store and current B-fragment prefetch are not better mainlines.
- `m16n64` variants tested so far are not better than the current `m16n32 4x2` mainline.
- The K32 `4x4` CTA comparison reduces short-scoreboard stalls but does not reduce barrier stalls. Its 512-thread block lowers Tensor Core utilization and is slower than K32 `4x2`.
- K32 symmetric `skew16` is worse than `skew8`: it nearly doubles shared-load pressure and increases short-scoreboard stalls.
- K32 A-only `skew16` is also worse than the base A `skew8`, increasing shared-load pressure and short-scoreboard stalls.
- K32 B-only `skew16` is worse as well and increases shared-load pressure more than A-only `skew16`. Keep both A and B at `skew8`; further padding tuning is low value.

NCU shows that K32 reduces shared-load pressure and short-scoreboard stalls while increasing barrier stalls. Keep the `4x2` CTA shape; the next high-value direction is to reduce synchronization cost or improve pipeline overlap without increasing the block warp count.

## Repository Hygiene

- Do not commit build outputs, raw logs unless intentionally documenting an experiment, `.ncu-rep`, `.nsys-rep`, or generated caches.
- `results/raw/`, `results/table/`, and `results/plots/` are used in the local/server sync workflow.
- Avoid broad cleanup of `src/fail/`; these files document explored variants.
- Keep changes scoped. This repository values benchmark comparability, so do not silently change timing, correctness tolerances, input generation, or output parsing formats.
