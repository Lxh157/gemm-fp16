# CUDA GEMM Optimization Practice

这个仓库是一个 CUDA GEMM 优化实验项目，目标是把 GEMM 从基础 FP32 kernel 一路推进到 FP16 input / FP32 accumulate / Tensor Core，并用 benchmark、图表和 Nsight Compute 指标驱动后续优化。

当前主线按 FP16/Tensor Core 优化路线组织：

- FP32 naive / tiled / register-blocking kernels 用作基础优化练习和框架验证。
- FP16 input / FP32 accumulate 的 non-Tensor-Core kernels 用作对照。
- WMMA kernels 用于建立 Tensor Core staged / cp.async 数据供给路径。
- 当前重点是 inline PTX MMA + ldmatrix + cp.async 主线，并与 cuBLASLt FP16acc baseline 对比。

## 当前状态

当前最强 custom kernel 是：

```text
impl:
mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync

file:
src/fp16_mma/gemm_mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync.cu
```

它的核心组织：

- FP16 input，FP32 accumulate
- inline PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- `ldmatrix` 从 shared memory 取 A/B fragment
- CTA tile: `64x64`
- warp tile: `16x32`
- warp layout: `4x2`
- K stage depth: `64`
- shared padding: `skew16`
- global-to-shared: double-buffered `cp.async.cg`
- C 写回: `float2` vectorized store
- main loop: skip final unnecessary `__syncthreads()`

最新阶段性结论：

- `skew16` 仍是当前 padding 参数里的最优点；更大的 `skew32` 明显回退。
- `cp.async.cg` 优于当前试过的 `cp.async.ca` 对照。
- `K64 + 4x2 warp` 是当前 MMA 主线基础。
- `float2` 写回是一次明确有效的提升。
- `skiplastsync` 在大尺寸上稳定带来约 `0.4% ~ 0.6%` 的小幅收益。
- `nocheck store` 和 B fragment prefetch 对大尺寸没有成为新主线。
- 当前已经进入小幅局部优化区间，下一步重点转向 Nsight Compute 对比 custom kernel 与 cuBLASLt 的瓶颈差异。

## 最新结果摘录

数据来自：

```text
results/table/bench_phase2_4090_tc_20260530_195030.csv
```

测试环境为 RTX 4090 / CUDA 11.8。单位：GFLOP/s，取 benchmark 输出的 median。

| impl | 1024 | 2048 | 3072 | 4096 |
| --- | ---: | ---: | ---: | ---: |
| `wmma_fp16acc_staged_cpasync_k64_4x4_skew16` | 87381 | 105954 | 109735 | 111113 |
| `mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16` | 80660 | 111107 | 120219 | 123623 |
| `mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore` | 95325 | 118908 | 124446 | 126026 |
| `mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync` | 95529 | 119225 | 125272 | 126619 |
| `mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck` | 95325 | 118737 | 124446 | 125700 |
| `mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch` | 95325 | 119411 | 124785 | 126102 |
| `mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16` | 80660 | 108943 | 117232 | 121634 |
| `cublaslt_fp16acc` | 67650 | 145889 | 155558 | 160996 |

4096 上当前 best custom kernel 达到约：

```text
126.6 TFLOP/s
约为当前 cublasLt baseline 的 78.6%
```

注意：`1024` 上 custom kernel 高于当前仓库口径的 `cublaslt_fp16acc`，但大尺寸上 cuBLASLt 明显领先。后续需要 NCU 判断差距来自 tensor pipe 利用率、shared/ldmatrix、warp readiness、occupancy，还是更深层的 pipeline 组织差异。

## 环境

主要目标架构是 Ada / SM89。

`CMakeLists.txt` 当前设置：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

当前主要测试服务器环境：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

依赖：

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
      gemm_naive.cu
      gemm_tiled.cu
      gemm_tiled_rb1x4.cu
      gemm_tiled_rb2x4.cu
      gemm_thread_tiled_1d.cu
      gemm_cublas.cu
      cublaslt_baseline.cu

    fp16_acc/
      gemm_tiled_fp16acc.cu
      gemm_tiled_fp16acc_rb1x4.cu
      gemm_tiled_fp16acc_rb2x4.cu
      gemm_cublas_gemmex_fp16acc.cu

    fp16_wmma/
      gemm_wmma_fp16acc*.cu

    fp16_mma/
      gemm_mma_fp16acc*.cu

    fail/
      failed_or_superseded_wmma_variants.cu

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

构建产物：

```text
build/bench_gemm
```

## 单点运行

当前 best custom kernel：

```bash
./build/bench_gemm \
  --impl mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync \
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

## 批量 benchmark

当前 Tensor Core sweep：

```bash
CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
```

覆盖尺寸：

```bash
SIZES_OVERRIDE="1024 2048 4096" \
CHECK_MAX_SIZE=256 \
PROFILE_SET=phase2_4090_tc \
bash scripts/run_bench.sh
```

指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 \
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

当前主要图表：

```text
results/plots/gflops_phase2_4090_tc_mma.png
results/plots/rel_to_cublaslt_phase2_4090_tc_mma.png
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
NCU_SIZES="2048 4096" bash scripts/run_all_scripts.sh
```

GPU 空闲判定逻辑：

- GPU 名称包含 `4090`
- 没有 compute process
- `memory.used <= EMPTY_MEM_MB`，默认 `1024 MiB`

可覆盖：

```bash
EMPTY_MEM_MB=2048 bash scripts/run_all_scripts.sh
```

## Nsight Compute 文本 profiling

NCU 脚本：

```bash
bash scripts/run_ncu_compare.sh
```

默认比较：

```text
mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync
cublaslt_fp16acc
```

默认尺寸：

```text
4096
```

输出为 `.txt`，不生成 `.ncu-rep`：

```text
profiles/ncu/ncu_mma_best/
profiles/ncu/ncu_cublaslt/
```

常用覆盖：

```bash
NCU_SIZES="2048 3072 4096" bash scripts/run_ncu_compare.sh
NCU_SET=full NCU_PAGE=raw bash scripts/run_ncu_compare.sh
NCU_BIN=/path/to/ncu bash scripts/run_ncu_compare.sh
```

NCU 注意事项：

- NCU 会显著扰动运行时间，不用 NCU 输出的耗时/GFLOP/s 做最终性能结论。
- NCU 只用于判断 stall、occupancy、pipeline、shared memory、global memory、instruction mix。
- 共用服务器上可能遇到 `/tmp/nsight-compute-lock` stale lock。脚本会提前检查并提示 owner。不要在未确认的情况下删除别人的 lock。

第一轮 NCU 重点关注：

- SM / Tensor Core 利用率
- achieved occupancy / active warps
- eligible warps per scheduler
- warp stall reasons
- shared memory throughput / bank conflict / ldmatrix 相关压力
- global memory 与 cp.async 行为
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

## 添加新 kernel

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
- 后续短后缀约定：
  - `vs`: vector store
  - `sls`: skip last sync
  - `bpf`: B fragment prefetch
  - `nc`: no check

## 技术路线记录

已经确认有效的方向：

- 从 WMMA 下沉到 inline MMA/ldmatrix 是大幅提升方向。
- K64 比 K32 更适合作为当前 MMA 主线。
- `skew16` 是当前 padding sweep 中的优选。
- `float2` 写回明显优于标量写回。
- 去掉最后一轮无必要同步有小幅稳定收益。

已经确认不是当前主线的方向：

- `skew32`
- `cp.async.ca`
- `m16n64 4x1 / 4x2` 作为替代主线
- `nocheck` store
- B fragment prefetch 当前写法

下一步：

1. 等待 Nsight Compute 文本 profile，比较当前 best custom kernel 与 `cublaslt_fp16acc`。
2. 根据 NCU 判断下一轮方向：
   - shared layout / swizzle
   - cp.async stage depth
   - warp tile / CTA tile 重组
   - occupancy / register pressure
   - epilogue / store path
3. 避免继续只靠盲 sweep 做小幅随机波动优化。

## 仓库卫生

- 不提交 build 输出、临时日志、`.ncu-rep`、`.nsys-rep`。
- `results/raw/` 和 `results/table/` 用于实验数据同步。
- `results/plots/` 用于图表输出。
- 保持 benchmark 输出格式稳定，因为 `raw_to_csv.py` 和 `plot.py` 依赖它。
- 不要随意改 correctness tolerance、输入生成、计时口径，否则历史结果不可比。
