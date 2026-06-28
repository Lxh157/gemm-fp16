# CUDA GEMM Optimization Practice

## H100 / SM90a Current Mainline

This checkout is currently used for the H100 WGMMA optimization route. The current best custom H100 kernel is:

```text
impl: wgmma_m64n128k32_tma_ab
file: src/fp16_wgmma/gemm_wgmma_m64n128k32_tma_ab.cu
```

Use same-device, same-run comparisons against `cublaslt_fp16acc`; do not compare H100 results numerically with historical RTX 4060/4090 runs. For H100 batch runs, prefer:

```bash
PROFILE_SET=h100_wgmma SIZES_OVERRIDE="1024 2048 4096" CHECK_MAX_SIZE=256 bash scripts/run_bench.sh
BEST_IMPL=wgmma_m64n128k32_tma_ab NCU_SIZES=2048 bash scripts/run_ncu_compare.sh
```

`CMakeLists.txt` currently targets `90a`.

这个仓库是一个 CUDA GEMM 优化实验项目，目标是把 GEMM 从基础 FP32 kernel 推进到 FP16 input / FP32 accumulate / Tensor Core，并用 benchmark、图表和 Nsight Compute 文本指标驱动后续优化。

当前主线按 FP16/Tensor Core 优化路线组织：

- FP32 naive / tiled / register-blocking kernels 用作基础优化练习和框架验证。
- FP16 input / FP32 accumulate 的 non-Tensor-Core kernels 用作对照。
- WMMA kernels 用于建立 Tensor Core staged / cp.async 数据供给路径。
- 当前重点是 inline PTX MMA + `ldmatrix` + `cp.async`，并与 `cublaslt_fp16acc` baseline 对比。

## 当前状态

当前本地 RTX 4060 Laptop 最强 custom kernel 是：

```text
impl:
mma_fp16acc_m16n32_k32_vs

file:
src/fp16_mma/gemm_mma_fp16acc_m16n32_k32_vs.cu
```

它的核心组织：

- FP16 input，FP32 accumulate
- inline PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- `ldmatrix` 从 shared memory 取 A/B fragment
- CTA tile: `64x64`
- warp tile: `16x32`
- warp layout: `4x2`
- K stage depth: `32`
- shared padding: A/B 都为 `skew8`
- global-to-shared: double-buffered `cp.async.cg`
- C 写回: `float2` vectorized store
- main loop: skip final unnecessary `__syncthreads()`

最新本地数据：

```text
results/table/bench_phase2_4090_tc_20260605_012329.csv
```

该数据来自 RTX 4060 Laptop，同轮比较如下，单位为 GFLOP/s，取 benchmark 输出的 median：

| impl | 1024 | 2048 | 3072 | 4096 |
| --- | ---: | ---: | ---: | ---: |
| `mma_fp16acc_m16n32_k32_vs` | 14463 | 15986 | 16197 | 15726 |
| `mma_fp16acc_m16n32_k32_b16_vs` | 14364 | 15843 | 14748 | 15569 |
| `cublaslt_fp16acc` | 15087 | 17839 | 18614 | 17829 |

4096 上当前 best custom kernel 约为：

```text
15.73 TFLOP/s
约为同轮 cublasLt baseline 的 88.2%
```

不要把本地 4060 结果和历史 RTX 4090 结果直接数值比较。历史 4090 数据仍可作为实验记录和方向参考，但结论必须基于同设备、同环境、同轮 benchmark。

## 最新技术结论

已经确认有效的方向：

- 从 WMMA 下沉到 inline MMA / `ldmatrix` 是大幅提升方向。
- 在本地 RTX 4060 上，K32 staged 主线优于历史 K64 主线；较小 shared footprint 明显降低 shared-load 压力。
- `cp.async.cg` 优于当前试过的 `cp.async.ca` 对照。
- `float2` vectorized C store 是明确收益。
- 跳过最后一轮无必要 CTA sync 是稳定小收益。

已经确认不是当前主线的方向：

- K32 `4x4` CTA：short-scoreboard 降低，但 barrier stall 基本不降，512-thread block 降低 Tensor Core 利用率，整体慢于 `4x2`。
- K32 symmetric `skew16`：shared-load 压力和 short-scoreboard 明显恶化。
- K32 A-only `skew16`：慢于 A `skew8`，shared-load 压力上升。
- K32 B-only `skew16`：也慢于 B `skew8`，shared-load 压力恶化更明显。
- `skew32`
- `cp.async.ca`
- `m16n64 4x1 / 4x2` 作为替代主线
- `nocheck` store
- 当前写法的 B fragment prefetch

下一步高价值方向：

1. 保持 K32、`4x2` CTA、A/B `skew8`。
2. 在不增加 block warp 数的前提下，降低同步成本或改善 pipeline overlap。
3. 继续用 NCU 文本指标判断 Tensor Core utilization、barrier stall、short-scoreboard、shared-load 压力和 eligible warps。
4. 避免继续只做 padding 或 CTA 扩大这类已经证伪的 sweep。

## 环境

主要目标架构是 Ada / SM89。

`CMakeLists.txt` 当前设置：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

常用依赖：

- CMake 3.18+
- CUDA Toolkit
- cuBLAS / cuBLASLt
- Python 3
- matplotlib，用于画图
- Nsight Compute CLI `ncu`，用于 profiling

## 目录结构

```text
gemm-fp16/
  CMakeLists.txt
  README.md
  AGENT.md

  src/
    main_bench.cu
    utils.cuh
    cublaslt_fp16acc.cu

    fp32/
    fp16_acc/
    fp16_wmma/
    fp16_mma/
    fail/

  scripts/
    collect_env.sh
    run_bench.sh
    raw_to_csv.py
    plot.py
    run_ncu_compare.sh
    run_all_scripts.sh

  results/
    raw/
    table/
    plots/

  profiles/
    ncu/
      ncu_mma_best/
      ncu_cublaslt/
    nsys/

  logs/
```

`src/fail/` 中的文件是已经尝试过但失败或被替代的实验版本，保留用于比较，不建议随手清理。

## 构建

从仓库根目录：

```bash
cmake -S . -B build
cmake --build build -j
```

本地内存紧张时：

```bash
cmake --build build -j2
```

构建产物：

```text
build/bench_gemm
```

## 单点运行

当前 best custom kernel：

```bash
./build/bench_gemm \
  --impl mma_fp16acc_m16n32_k32_vs \
  --M 4096 --N 4096 --K 4096 \
  --warmup 3 --repeat 10 --no-check
```

cuBLASLt baseline：

```bash
./build/bench_gemm \
  --impl cublaslt_fp16acc \
  --M 4096 --N 4096 --K 4096 \
  --warmup 3 --repeat 10 --no-check
```

输出中主要看：

```text
[time] min=... ms, median=... ms, avg=... ms
[perf] median=... GFLOP/s
```

项目统一用 median GFLOP/s 作为性能比较指标。

## 批量 Benchmark

当前 Tensor Core sweep：

```bash
CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
```

只跑指定尺寸：

```bash
SIZES_OVERRIDE="1024 2048 4096" \
CHECK_MAX_SIZE=256 \
PROFILE_SET=phase2_4090_tc \
bash scripts/run_bench.sh
```

只跑指定实现，适合每轮小对照：

```bash
IMPLS_OVERRIDE="mma_fp16acc_m16n32_k32_vs cublaslt_fp16acc" \
CHECK_MAX_SIZE=256 \
PROFILE_SET=phase2_4090_tc \
bash scripts/run_bench.sh
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 \
CHECK_MAX_SIZE=256 \
PROFILE_SET=phase2_4090_tc \
bash scripts/run_bench.sh
```

raw 输出写到：

```text
results/raw/
```

## CSV 与图表

把最新 raw 结果提取成 CSV：

```bash
python3 scripts/raw_to_csv.py
```

输出目录：

```text
results/table/
```

画图：

```bash
python3 scripts/plot.py
```

输出目录：

```text
results/plots/
```

## 一键脚本

`scripts/run_all_scripts.sh` 会自动选择空闲 RTX 4090，并执行：

```text
cmake configure
cmake build
benchmark sweep
raw_to_csv
plot
NCU compare
```

默认运行：

```bash
bash scripts/run_all_scripts.sh
```

跳过 NCU：

```bash
RUN_NCU=0 bash scripts/run_all_scripts.sh
```

选择 NCU 尺寸：

```bash
NCU_SIZES="1024 2048" bash scripts/run_all_scripts.sh
```

注意：这个脚本的自动选卡逻辑仍按共用 RTX 4090 服务器设计。本地 4060 开发时通常直接用 `run_bench.sh` 和 `run_ncu_compare.sh`。

## Nsight Compute 文本 Profiling

NCU 脚本：

```bash
bash scripts/run_ncu_compare.sh
```

默认比较：

```text
mma_fp16acc_m16n32_k32_vs
cublaslt_fp16acc
```

默认尺寸：

```text
4096
```

本地 WSL 上 `NCU_SET=full` 跑 4096 可能失败或耗时过长，当前实践中常用 1024 做结构诊断：

```bash
NCU_SIZES=1024 bash scripts/run_ncu_compare.sh
```

输出为 `.txt`，不生成 `.ncu-rep`：

```text
profiles/ncu/ncu_mma_best/
profiles/ncu/ncu_cublaslt/
```

常用覆盖：

```bash
MMA_BEST_IMPL=mma_fp16acc_m16n32_k32_vs NCU_SIZES=1024 bash scripts/run_ncu_compare.sh
NCU_SET=full NCU_PAGE=raw bash scripts/run_ncu_compare.sh
NCU_BIN=/path/to/ncu bash scripts/run_ncu_compare.sh
```

NCU 注意事项：

- NCU 会显著扰动运行时间，不用 NCU 输出的耗时/GFLOP/s 做最终性能结论。
- NCU 只用于判断 stall、occupancy、pipeline、shared memory、global memory、instruction mix。
- 共用服务器上可能遇到 `/tmp/nsight-compute-lock` stale lock。脚本会提前检查并提示 owner。不要在未确认的情况下删除别人的 lock。

重点关注：

- SM / Tensor Core 利用率
- achieved occupancy / active warps
- eligible warps per scheduler
- warp stall reasons，尤其 barrier、short scoreboard、math pipe throttle
- shared memory throughput / bank conflict / `ldmatrix` 相关压力
- global memory 与 `cp.async` 行为
- instruction mix

## 实验口径

- 计时方式：CUDA events
- 默认 warmup: `3`
- 默认 repeat: `10`
- 性能指标：median GFLOP/s
- correctness check 默认开启
- 大尺寸 sweep 通常用 `CHECK_MAX_SIZE=256`，超过该尺寸自动加 `--no-check`

当前 tolerance：

```text
FP32 kernels:       atol=1e-3, rtol=1e-3
FP16 input kernels: atol=2e-2, rtol=2e-2
```

很多 Tensor Core kernel 要求 M/N/K 满足 tile 对齐，launcher 中保留显式 guard。

## 添加新 Kernel

1. 在对应目录新增 `.cu` 文件：

```text
src/fp32/
src/fp16_acc/
src/fp16_wmma/
src/fp16_mma/
```

2. 在 `CMakeLists.txt` 的 `bench_gemm` 源文件列表中加入新文件。

3. 在 `src/main_bench.cu` 中加入：

- launcher forward declaration
- `--impl` validation
- `use_fp16_inputs` 判断，若使用 half input
- dispatch branch

4. 如果需要参与批量测试，加入：

```text
scripts/run_bench.sh
scripts/plot.py
```

5. 命名建议：

- 文件名和 impl 保持可读，但不要无限叠长后缀。
- 短后缀约定：
  - `vs`: vector store
  - `sls`: skip last sync
  - `bpf`: B fragment prefetch
  - `nc`: no check

## 仓库卫生

- 不提交 build 输出、临时日志、`.ncu-rep`、`.nsys-rep`。
- `results/raw/`、`results/table/`、`results/plots/` 用于实验数据与图表输出。
- 保持 benchmark 输出格式稳定，因为 `raw_to_csv.py` 和 `plot.py` 依赖它。
- 不要随意改 correctness tolerance、输入生成、计时口径，否则历史结果不可比。
- 避免 broad cleanup `src/fail/`，这些文件记录已经探索过的失败路径。
