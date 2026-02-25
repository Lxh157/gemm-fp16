# CUDA GEMM Optimization Practice (FP16, RTX 4060)

## 项目目标
本项目用于练习 CUDA / GPU 性能优化基础，围绕 GEMM（矩阵乘法）实现与 profiling 展开。  
当前目标是从基础实现出发，逐步完成：
- naive GEMM baseline
- shared memory tiling 优化
- benchmark 与性能对比
- 初步 profiling（Nsight Compute / Nsight Systems）

---

## 当前进展
- [x] 搭建基础 benchmark 框架（CUDA events 计时）
- [x] 实现 naive GEMM kernel（正确性校验）
- [x] 实现优化版 GEMM kernel（shared memory tiling）
- [x] 跑通基础性能对比（naive vs optimized）
- [ ] 接入 cuBLAS / cuBLASLt baseline（进行中）
- [ ] 系统化 Nsight Compute 指标分析（进行中）
- [ ] 完整图表与报告式总结（进行中）

---

## 环境
- GPU: NVIDIA GeForce RTX 4060
- OS: WSL2 Ubuntu
- CUDA: 13.1
- Compiler: (g++ / nvcc 12.8)
- Tools: Nsight Compute / Nsight Systems（基础使用）

---

## 目录结构
```text
gemm-fp16/
  src/
    main_bench.cu          # benchmark 入口
    gemm_naive.cu          # naive GEMM
    gemm_tiled.cu          # tiled GEMM（优化版）
    gemm_regblock.cu       # reg-blocking 版本
    cublaslt_baseline.cu   # cuBLASLt baseline
    utils.cuh              # 工具函数 / 校验 / 计时辅助
  scripts/
    run_bench.sh           # 批量跑 benchmark
    collect_env.sh         # 导出环境信息
    plot.py                # 画图脚本（可选）
  results/
    raw/                   # 原始结果（CSV/日志）
    plots/                 # 图表
  profiles/
    nsys/                  # Nsight Systems traces
    ncu/                   # Nsight Compute reports

---

## 当前结果（阶段性，FP32）

环境：RTX 4060 + WSL2 Ubuntu + CUDA 12.8  
实现：naive vs shared-memory tiled（16x16）  
计时方式：CUDA events（warmup + repeat，取 median）

| impl  | M=N=K | median time (ms) | GFLOP/s |
|---|---:|---:|---:|
| naive | 256  | 0.054 | 618.264 |
| tiled | 256  | 0.046 | 762.047 |
| naive | 512  | 0.394 | 681.779 |
| tiled | 512  | 0.297 | 903.945 |
| naive | 1024 | 3.109 | 690.776 |
| tiled | 1024 | 2.393 | 897.399 |

### 初步观察
- 相比 naive，tiled 版本在 256 、512 和 1024 尺寸上均有性能提升。
- 在更大尺寸下（如 512），shared memory tiling 带来的收益更明显（当前阶段性结果约 `1.4x`）。
- 下一步将补充 1024+ 尺寸、Nsight Compute 指标分析，并接入 cuBLAS/cuBLASLt baseline。