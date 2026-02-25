#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录运行：
#   bash scripts/run_bench.sh
# 输出：
#   results/raw/bench_fp32_naive_tiled.csv

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/bench_gemm"
OUT_DIR="${ROOT_DIR}/results/raw"
OUT_CSV="${OUT_DIR}/bench_fp32_naive_tiled.csv"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] binary not found: ${BIN}"
  echo "Please build first:"
  echo "  mkdir -p build && cd build && cmake .. && make -j"
  exit 1
fi

# 你可以按时间调整这些尺寸
SIZES=(256 512 1024)
WARMUP=3
REPEAT=10

echo "impl,M,N,K,median_ms,gflops" > "${OUT_CSV}"

run_one() {
  local impl="$1"
  local m="$2"
  local n="$3"
  local k="$4"

  echo "[RUN] impl=${impl}, M=${m}, N=${n}, K=${k}"

  # 捕获输出
  local output
  output="$("${BIN}" --impl "${impl}" --M "${m}" --N "${n}" --K "${k}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}")"

  echo "${output}"

  # 解析 median time 和 gflops（基于当前输出格式）
  local median_ms
  local gflops

  median_ms="$(echo "${output}" | grep '^\[time\]' | sed -E 's/.*median=([0-9.]+) ms.*/\1/')"
  gflops="$(echo "${output}" | grep '^\[perf\]' | sed -E 's/.*median=([0-9.]+) GFLOP\/s.*/\1/')"

  if [[ -z "${median_ms}" || -z "${gflops}" ]]; then
    echo "[ERROR] failed to parse output for impl=${impl}, size=${m}"
    exit 1
  fi

  echo "${impl},${m},${n},${k},${median_ms},${gflops}" >> "${OUT_CSV}"
}

for s in "${SIZES[@]}"; do
  run_one naive "${s}" "${s}" "${s}"
  run_one tiled "${s}" "${s}" "${s}"
done

echo
echo "[DONE] CSV saved to: ${OUT_CSV}"