#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt

# 用法：
#   python3 scripts/plot.py
#
# 默认行为：
#   - Phase 1（4060）优先读取：
#       1) results/raw/bench_phase1_4060_all_*.txt
#       2) 若没有，再回退到历史文件 results/raw/bench_fp16_*.txt
#   - Phase 2（4090）读取：
#       results/raw/bench_phase2_4090_tc_*.txt
#
# 输出：
#   results/plots/gflops_phase1_fp32.png
#   results/plots/gflops_phase1_fp16.png
#   results/plots/rel_to_cublaslt_phase1_fp16.png
#   results/plots/gflops_phase2_4090_tc.png
#   results/plots/rel_to_cublaslt_phase2_4090_tc.png
#
# 补充：
# 后面统一这样：
# 安装包：
# python3 -m pip install --user 包名
# 运行脚本：
# python3 脚本名.py
# 查看包：
# python3 -m pip show 包名
# 别再混用 python。

RAW_DIR = Path("results/raw")
OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAT_CASE = re.compile(r"===== impl=([\w_]+), M=N=K=(\d+) =====")
PAT_PERF = re.compile(r"\[perf\]\s+median=([0-9.]+)\s+GFLOP/s")

PHASE1_FP32_IMPLS = [
    "naive",
    "tiled",
    "tiled_rb1x4",
    "tiled_rb2x4",
    "thread_tiled_1d",
    "cublas",
    "cublaslt",
]

PHASE1_FP16_IMPLS = [
    "tiled_fp16acc",
    "tiled_fp16acc_rb1x4",
    "tiled_fp16acc_rb2x4",
    "wmma_fp16acc",
    "wmma_fp16acc_staged",
    "wmma_fp16acc_staged_cpasync",
    "cublas_gemmex_fp16acc",
    "cublaslt_fp16acc",
]

PHASE2_4090_IMPLS = [
    # "wmma_fp16acc_staged_cpasync",
    # "wmma_fp16acc_staged_cpasync_k32",
    # "wmma_fp16acc_staged_cpasync_k32_4x2",
    # "wmma_fp16acc_staged_cpasync_k32_split",
    # "wmma_fp16acc_staged_cpasync_k32_skew16",
    # "wmma_fp16acc_staged_cpasync_k32_4x4_skew16",
    # "wmma_fp16acc_staged_cpasync_k32_4x8_skew16",
    # "wmma_fp16acc_staged_cpasync_k32_skewA16_B8",
    # "wmma_fp16acc_staged_cpasync_k32_skewA8_B16",
    # "wmma_fp16acc_staged_cpasync_k32_skewA24_B16",
    # "wmma_fp16acc_staged_cpasync_k32_skewA8_B24",
    # "wmma_fp16acc_staged_cpasync_k32_skewA8_B32",
    # "wmma_fp16acc_staged_cpasync_k32_skewA16_B32",
    # "wmma_fp16acc_staged_cpasync_k32_skewA24_B32",
    # "wmma_fp16acc_staged_cpasync_k64",
    "wmma_fp16acc_staged_cpasync_k64_skew16",
    "wmma_fp16acc_staged_cpasync_k64_4x4_skew16",
    "wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16",
    # "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4",
    # "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch",
    # "wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2",
    # "wmma_fp16acc_staged_cpasync_k64_4x8_skew16",

    # "mma_fp16acc",
    # "mma_fp16acc_m16n32",
    # "mma_fp16acc_m16n32_k32",
    # "mma_fp16acc_m16n32_k32_4x2",
    # "mma_fp16acc_m16n32_k32_4x4",
    # "mma_fp16acc_m16n32_staged_cpasync_k32_4x2",
    "mma_fp16acc_m16n32_k32_vs",
    # "mma_fp16acc_m16n32_staged_cpasync_k64_4x2",
    # "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16",
    # "mma_fp16acc_m16n32_staged_cpasync_ca_k64_4x2_skew16",
    # "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore",
    "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync",
    "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_bprefetch",
    # "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync_nocheck",
    # "mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32",
    # "mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16",
    # "mma_fp16acc_m16n64_staged_cpasync_k64_4x1_skew16",
    # "mma_fp16acc_m16n64_staged_cpasync_k64_4x2_skew16",
    "mma_fp16acc_m32n32_k64_4x2",
    # "cublas_gemmex_fp16acc",
    "cublaslt_fp16acc",
]


def pick_latest(candidates):
    return candidates[-1] if candidates else None


def parse_bench_file(path: Path):
    data = {}
    cur_impl = None
    cur_size = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_case = PAT_CASE.search(line)
            if m_case:
                cur_impl = m_case.group(1)
                cur_size = int(m_case.group(2))
                data.setdefault(cur_impl, {})
                continue

            m_perf = PAT_PERF.search(line)
            if m_perf and cur_impl is not None and cur_size is not None:
                data[cur_impl][cur_size] = float(m_perf.group(1))
                cur_impl = None
                cur_size = None

    return data


def collect_sizes(data, impls):
    sizes = set()
    for impl in impls:
        if impl in data:
            sizes |= set(data[impl].keys())
    return sorted(sizes)


def plot_gflops(data, impls, sizes, title, out_path):
    if not sizes:
        print(f"[WARN] no sizes found, skip {out_path.name}")
        return

    plt.figure()
    for impl in impls:
        if impl not in data:
            continue
        ys = [data[impl].get(s, float("nan")) for s in sizes]
        plt.plot(sizes, ys, marker="o", label=impl)

    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel("GFLOP/s (median)")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_relative(data, impls, baseline_impl, sizes, title, ylabel, out_path):
    if not sizes:
        print(f"[WARN] no sizes found, skip {out_path.name}")
        return

    if baseline_impl not in data:
        print(f"[WARN] baseline {baseline_impl} not found, skip {out_path.name}")
        return

    baseline = data[baseline_impl]

    plt.figure()
    for impl in impls:
        if impl not in data or impl == baseline_impl:
            continue

        ys = []
        for s in sizes:
            if s in data[impl] and s in baseline and baseline[s] != 0:
                ys.append(100.0 * data[impl][s] / baseline[s])
            else:
                ys.append(float("nan"))

        plt.plot(sizes, ys, marker="o", label=f"{impl}/{baseline_impl}")

    plt.xlabel("Matrix size (M=N=K)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


phase1_explicit = sorted(RAW_DIR.glob("bench_phase1_4060_all_*.txt"))
phase1_legacy = sorted(RAW_DIR.glob("bench_fp16_*.txt"))
phase2_4090 = sorted(RAW_DIR.glob("bench_phase2_4090_tc_*.txt"))

input_phase1 = pick_latest(phase1_explicit) or pick_latest(phase1_legacy)
input_phase2 = pick_latest(phase2_4090)

if input_phase1 is None and input_phase2 is None:
    raise FileNotFoundError(
        "No phase1/phase2 raw result files found under results/raw/. "
        "Run scripts/run_bench.sh first."
    )

if input_phase1 is not None:
    data_phase1 = parse_bench_file(input_phase1)

    phase1_fp32_sizes = collect_sizes(data_phase1, PHASE1_FP32_IMPLS)
    phase1_fp16_sizes = collect_sizes(data_phase1, PHASE1_FP16_IMPLS)

    out_phase1_fp32 = OUT_DIR / "gflops_phase1_fp32.png"
    out_phase1_fp16 = OUT_DIR / "gflops_phase1_fp16.png"
    out_phase1_rel_fp16 = OUT_DIR / "rel_to_cublaslt_phase1_fp16.png"

    plot_gflops(
        data_phase1,
        PHASE1_FP32_IMPLS,
        phase1_fp32_sizes,
        "Phase 1 (RTX 4060 Laptop) - FP32 GEMM Throughput",
        out_phase1_fp32,
    )

    plot_gflops(
        data_phase1,
        PHASE1_FP16_IMPLS,
        phase1_fp16_sizes,
        "Phase 1 (RTX 4060 Laptop) - FP16 / Tensor Core Throughput",
        out_phase1_fp16,
    )

    plot_relative(
        data_phase1,
        PHASE1_FP16_IMPLS,
        "cublaslt_fp16acc",
        phase1_fp16_sizes,
        "Phase 1 (RTX 4060 Laptop) - Relative Throughput vs cuBLASLt FP16acc",
        "Percent of cuBLASLt FP16acc (%)",
        out_phase1_rel_fp16,
    )

    print(f"[OK] Parsed Phase 1 raw: {input_phase1}")
    print(f"[OK] Wrote: {out_phase1_fp32}")
    print(f"[OK] Wrote: {out_phase1_fp16}")
    print(f"[OK] Wrote: {out_phase1_rel_fp16}")
else:
    print("[WARN] No Phase 1 raw file found, skip Phase 1 plots.")

if input_phase2 is not None:
    data_phase2 = parse_bench_file(input_phase2)
    phase2_sizes = collect_sizes(data_phase2, PHASE2_4090_IMPLS)

    out_phase2_gflops = OUT_DIR / "gflops_phase2_4090_tc_prefetch2.png"
    out_phase2_rel = OUT_DIR / "rel_to_cublaslt_phase2_4090_tc_prefetch2.png"

    plot_gflops(
        data_phase2,
        PHASE2_4090_IMPLS,
        phase2_sizes,
        "Phase 2 (RTX 4090 Server) - Tensor Core Mainline Throughput",
        out_phase2_gflops,
    )

    plot_relative(
        data_phase2,
        PHASE2_4090_IMPLS,
        "cublaslt_fp16acc",
        phase2_sizes,
        "Phase 2 (RTX 4090 Server) - Relative Throughput vs cuBLASLt FP16acc",
        "Percent of cuBLASLt FP16acc (%)",
        out_phase2_rel,
    )

    print(f"[OK] Parsed Phase 2 raw: {input_phase2}")
    print(f"[OK] Wrote: {out_phase2_gflops}")
    print(f"[OK] Wrote: {out_phase2_rel}")
else:
    print("[WARN] No Phase 2 raw file found, skip Phase 2 plots.")
