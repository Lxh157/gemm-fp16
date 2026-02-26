#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/run_bench.sh
# 或
#   BUILD_DIR=build WARMUP=3 REPEAT=10 bash scripts/run_bench.sh

BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BUILD_DIR}/bench_gemm"
OUT_DIR="results/raw"
LOG_DIR="logs"

WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-10}"

# 先做最小闭环：9个点
SIZES=(256 512 1024)
IMPLS=(naive tiled tiled_rb1x4 cublas)

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

TS=$(date +%Y%m%d_%H%M%S)
OUT_TXT="${OUT_DIR}/bench_fp32_${TS}.txt"

echo "# bench_gemm batch run" | tee "${OUT_TXT}"
echo "# time: $(date)" | tee -a "${OUT_TXT}"
echo "# warmup=${WARMUP}, repeat=${REPEAT}, metric=median" | tee -a "${OUT_TXT}"
echo | tee -a "${OUT_TXT}"

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] binary not found: ${BIN}"
  echo "Please build first: cmake -S . -B build && cmake --build build -j"
  exit 1
fi

for s in "${SIZES[@]}"; do
  for impl in "${IMPLS[@]}"; do
    echo "===== impl=${impl}, M=N=K=${s} =====" | tee -a "${OUT_TXT}"
    "${BIN}" \
      --impl "${impl}" \
      --M "${s}" --N "${s}" --K "${s}" \
      --warmup "${WARMUP}" --repeat "${REPEAT}" \
      2>&1 | tee -a "${OUT_TXT}"
    echo | tee -a "${OUT_TXT}"
  done
done

echo "[DONE] results saved to ${OUT_TXT}"