# CUDA GEMM Optimization Practice（FP32 主线，后续扩展 FP16）

## 项目目标

本项目用于练习 CUDA / GPU 性能优化与 profiling，围绕 GEMM（矩阵乘法）实现一条清晰的优化链路，并用 **benchmark 数据 + Nsight Compute 指标**形成可复现实验结论。

当前主线聚焦 **FP32 GEMM**（后续再扩展到 FP16 / Tensor Core）：

- naive GEMM baseline
    
- shared memory tiling 优化
    
- thread coarsening / register blocking（每线程计算 1x4 输出）
    
- 统一口径 benchmark（稳态性能对比）
    
- 初步 NCU 指标分析（瓶颈解释）
    

## 当前进展

-  搭建 benchmark 框架：参数化 M/N/K，CUDA events 计时，输出 min/median/avg 与 GFLOP/s
    
-  naive GEMM kernel（CPU reference correctness check）
    
-  tiled GEMM kernel（shared memory tiling）
    
-  tiled_rb1x4 GEMM kernel（thread coarsening / register blocking）
    
-  统一口径 batch benchmark & 性能对比表（见下文）
    
-  Nsight Compute 初步对比（tiled vs tiled_rb1x4，见下文）
    
-  接入 cuBLAS / cuBLASLt baseline（进行中）
    
-  更系统的 NCU 指标/访存事务/指令级分析（进行中）
    
-  完整图表与报告式总结（进行中）
    

## 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
    
- OS: WSL2 Ubuntu
    
- CUDA: 13.1
    
- Compiler: g++ / nvcc 12.8
    
- Tools: Nsight Compute / Nsight Systems（基础使用）
    


## 目录结构

```text
gemm-fp16/
  src/
    main_bench.cu            # benchmark 入口（--impl/--M/--N/--K/--warmup/--repeat/--no-check）
    gemm_naive.cu            # naive GEMM
    gemm_tiled.cu            # tiled GEMM（shared memory）
    gemm_tiled_rb1x4.cu      # tiled + register blocking（每线程 1x4 输出）
    cublaslt_baseline.cu     # cuBLASLt baseline（进行中）
    utils.cuh                # 工具函数 / 校验 / 计时辅助
  scripts/
    run_bench.sh             # 批量跑 benchmark（统一口径）
    collect_env.sh           # 导出环境信息
    plot.py                  # 画图脚本（可选）
  results/
    raw/                     # 原始结果（日志/CSV）
    plots/                   # 图表
  profiles/
    nsys/                    # Nsight Systems traces
    ncu/                     # Nsight Compute reports
  logs/
```


## 复现方式（Build / Run）

### 1) 构建（WSL2 Ubuntu）

```bash
cd ~/gemm-fp16/build  
cmake ..  
make -j
```

构建产物：`build/bench_gemm`

### 2) 单点运行

下面命令会执行 CPU reference 校验并给出 GFLOP/s（使用 CUDA events 计时）：

```bash
./bench_gemm --impl tiled_rb1x4 --M 256 --N 256 --K 256 --warmup 3 --repeat 10  
./bench_gemm --impl tiled_rb1x4 --M 512 --N 512 --K 512 --warmup 3 --repeat 10
```

### 3) 批量 benchmark

运行脚本（默认 `warmup=3, repeat=10`，对 `256/512/1024`，实现 `naive/tiled/tiled_rb1x4` 共 9 个点）：

```bash
bash scripts/run_bench.sh
```

脚本输出文件路径与命名规则：

- 输出目录：`results/raw/`
    
- 文件名：`bench_fp32_YYYYmmdd_HHMMSS.txt`  
    例如：`results/raw/bench_fp32_20260225_214132.txt`
    

也可以自定义参数：

```bash
BUILD_DIR=build WARMUP=5 REPEAT=20 bash scripts/run_bench.sh
```


## 实验口径说明

### 1) 性能对比

- 计时方式：CUDA events
    
- `warmup >= 3`，`repeat >= 10`
    
- 使用 **median** 作为稳定性能指标
    
- correctness check：对 CPU reference 做校验（可通过 `--no-check` 关闭，用于纯 profiling 或批量跑更快）
    

### 2) NCU profiling

- NCU 会显著扰动运行时间，因此 **NCU 输出的 ms/GFLOP/s 不用于性能结论**
    
- 建议口径：`--no-check --warmup 0 --repeat 1`（或只看 repeat 对应的那次 kernel launch）
    

## 当前结果（阶段性，FP32）

### Benchmark results（CUDA events，warmup=3，repeat=10，取 median；全部 correctness PASS）

|Impl|256³ GFLOP/s|512³ GFLOP/s|1024³ GFLOP/s|
|---|---|---|---|
|naive|618.264|689.853|693.388|
|tiled|762.047|868.026|897.765|
|tiled_rb1x4|1212.928|1833.175|1843.701|
| cublas (sgemm) | 2337.962 | 4854.519 | 6307.224 |

> cuBLAS baseline：使用 `cublasSgemm`，并通过 row-major→column-major 的等价映射实现 `C = A × B`（row-major 语义），math mode = `CUBLAS_DEFAULT_MATH`。

### Speedup summary

|Size|tiled / naive|rb1x4 / tiled|rb1x4 / naive|
|---|---|---|---|
|256³|1.23x|1.59x|1.96x|
|512³|1.26x|2.11x|2.66x|
|1024³|1.29x|2.05x|2.66x|

### Relative to cuBLAS (rb1x4 / cublas)

| Size | rb1x4 / cublas |
|---|---:|
| 256³ | 51.9% |
| 512³ | 37.8% |
| 1024³ | 29.2% |

### 初步观察

- `tiled` 相比 `naive` 稳定提升约 1.23x~1.29x，说明 shared memory tiling 已能减少部分冗余访存并提升吞吐。
- 最大收益来自 `tiled_rb1x4`：在 512³/1024³ 上相对 `tiled` 稳定在 ~2x（bench 结果），说明 thread coarsening / register blocking 能显著摊薄 per-output 的 shared load / sync / address calc 等开销。
- 与 `cublasSgemm` 相比，当前 `tiled_rb1x4` 在 1024³ 上达到约 29.2% 的吞吐，仍有较大优化空间（例如更深的寄存器分块、向量化 load、提高数据复用、减少指令与同步开销、以及进一步靠近 tensor-core 路径等）。
    

## Nsight Compute 初步瓶颈解释（tiled vs tiled_rb1x4，1024³，profiling-only）

> 说明：下表指标用于解释瓶颈方向，不作为最终性能结论。

| 指标                                         | tiled     | tiled_rb1x4     | 结论                                        |
| ------------------------------------------ | --------- | --------------- | ----------------------------------------- |
| Achieved Occupancy (%)                     | 98.55     | 80.00（理论 83.33） | rb1x4 降低 occupancy（符合寄存器/每线程工作量上升的预期）     |
| Compute (SM) Throughput (SOL, %)           | 96.67     | 64.29           | SOL% 跨 kernel 不宜直接当“更快/更慢”证据；主要用来辅助判断资源侧重 |
| Memory Throughput (SOL, %)                 | 96.67     | 89.81           | 两者都偏高；tiled 更接近 memory-side SOL 上限        |
| L2 Hit Rate (%)                            | 98.17     | 96.14           | 都很高；差异不大                                  |
| Warp Cycles per Issued Instruction (cycle) | 38.13     | 26.50           | rb1x4 的 issue 间隔更短（更少“发不出指令”的空转）          |
| **Stall MIO Throttle (cycles/inst)**       | **20.02** | **12.99**       | rb1x4 显著降低 MIO 队列压力/相关 stall              |
| **Stall Barrier (cycles/inst)**            | **5.97**  | **3.99**        |  rb1x4 显著降低 barrier/sync 相关 stall         |

在 1024³ 的 profiling-only 口径下，`tiled_rb1x4` 相比 `tiled` 的 Achieved Occupancy 从 98.55% 降至 80%（理论 83.33%），这符合 register blocking / 每线程计算更多输出导致寄存器与线程资源占用上升的预期。但更关键的是，warp-level stall 明显下降：

- `Stall MIO Throttle`：20.02 → 12.99 cycles/inst（约 -35%）
    
- `Stall Barrier`：5.97 → 3.99 cycles/inst（约 -33%）
    
- `Warp Cycles per Issued Instruction`：38.13 → 26.50 cycles（issue 间隔显著缩短）
    

这说明 `tiled` 版本主要受 **MIO 指令队列压力（包含 shared memory 相关指令）** 与 **barrier/sync 开销**影响，warp 在“发不出下一条指令”的状态上花费较多周期。`tiled_rb1x4` 通过 thread coarsening / register blocking（每线程计算 1x4 输出）提高了每次加载/同步所能覆盖的有效计算量，从而摊薄 shared-memory 指令与同步的固定成本，降低 MIO throttle 与 barrier stall，最终在 bench 稳态测试中带来约 2× 的吞吐提升（1024³：~1844 vs ~898 GFLOP/s）。


## 下一步计划（短期）

1. 接入 `cublasSgemm` / cuBLASLt baseline（FP32）作为对照（同口径 benchmark）
    
2. 更系统化的 NCU 分析：补充指令/内存事务（global/shared）与 roofline 视角的解释
    
3. 视时间进行 block shape sweep（如 16×16、32×8、64×8 等）验证性能敏感性
    
4. 主线收束后再扩展 FP16 / Tensor Core 版本（作为进阶）