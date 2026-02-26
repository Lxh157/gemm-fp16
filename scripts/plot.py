#!/usr/bin/env python3
import re
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt

# 自动选择 results/raw/ 下最新的 bench_fp32_*.txt
RAW_DIR = Path("results/raw/")
candidates = sorted(RAW_DIR.glob("bench_fp32_*.txt"))
if not candidates:
    raise FileNotFoundError("No results/raw/bench_fp32_*.txt found. Run scripts/run_bench.sh first.")
INPUT = candidates[-1]

OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 解析：抓 impl 和 M=N=K 和 perf median GFLOP/s
pat_case = re.compile(r"===== impl=(\w+), M=N=K=(\d+) =====")
pat_perf = re.compile(r"\[perf\]\s+median=([0-9.]+)\s+GFLOP/s")

data = {}  # impl -> size -> gflops
cur_impl = None
cur_size = None

with open(INPUT, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = pat_case.search(line)
        if m:
            cur_impl = m.group(1)
            cur_size = int(m.group(2))
            data.setdefault(cur_impl, {})
            continue
        m = pat_perf.search(line)
        if m and cur_impl is not None and cur_size is not None:
            g = float(m.group(1))
            data[cur_impl][cur_size] = g
            cur_impl = None
            cur_size = None

# 需要的 impl 顺序（按你 README 展示）
order = ["naive", "tiled", "tiled_rb1x4", "tiled_rb2x4", "cublas"]

# sizes 取交集并排序
all_sizes = set()
for impl in order:
    if impl in data:
        all_sizes |= set(data[impl].keys())
sizes = sorted(all_sizes)

# --- 图1：GFLOP/s vs size ---
plt.figure()
for impl in order:
    if impl not in data:
        continue
    ys = [data[impl].get(s, float("nan")) for s in sizes]
    plt.plot(sizes, ys, marker="o", label=impl)
plt.xlabel("Matrix size (M=N=K)")
plt.ylabel("GFLOP/s (median)")
plt.title("FP32 GEMM Throughput (CUDA events, median)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
out1 = OUT_DIR / "gflops_fp32.png"
plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.close()

# --- 图2：Relative to cuBLAS ---
if "cublas" in data:
    plt.figure()
    cublas = data["cublas"]
    for impl in ["tiled_rb1x4", "tiled_rb2x4"]:
        if impl not in data:
            continue
        ys = []
        for s in sizes:
            if s in data[impl] and s in cublas and cublas[s] != 0:
                ys.append(100.0 * data[impl][s] / cublas[s])
            else:
                ys.append(float("nan"))
        plt.plot(sizes, ys, marker="o", label=f"{impl}/cublas")
    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel("Percent of cuBLAS (%)")
    plt.title("Relative Throughput vs cuBLAS (FP32)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    out2 = OUT_DIR / "rel_to_cublas_fp32.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()

print(f"[OK] Parsed: {INPUT}")
print(f"[OK] Wrote: {out1}")
print(f"[OK] Wrote: {OUT_DIR / 'rel_to_cublas_fp32.png'}")