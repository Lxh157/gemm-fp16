#!/usr/bin/env bash
set -euo pipefail

# Profile current best MMA kernel against cuBLASLt with Nsight Compute CLI.
# Text output only: this script intentionally does not pass --export, so no
# .ncu-rep file is generated.
#
# Usage:
#   bash scripts/run_ncu_compare.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_ncu_compare.sh
#   NCU_SIZES="2048 4096" bash scripts/run_ncu_compare.sh
#   NCU_SET=full NCU_PAGE=raw bash scripts/run_ncu_compare.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BUILD_DIR}/bench_gemm"

NCU_BIN="${NCU_BIN:-ncu}"
NCU_SET="${NCU_SET:-full}"
NCU_PAGE="${NCU_PAGE:-raw}"
NCU_PRINT_SUMMARY="${NCU_PRINT_SUMMARY:-per-kernel}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
NCU_SIZES="${NCU_SIZES:-4096}"
NCU_WARMUP="${NCU_WARMUP:-0}"
NCU_REPEAT="${NCU_REPEAT:-1}"
NCU_TMPDIR="${NCU_TMPDIR:-${OUT_ROOT:-profiles/ncu}/tmp}"

MMA_BEST_IMPL="${MMA_BEST_IMPL:-mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16_vstore_skiplastsync}"
CUBLASLT_IMPL="${CUBLASLT_IMPL:-cublaslt_fp16acc}"

OUT_ROOT="${OUT_ROOT:-profiles/ncu}"
MMA_OUT_DIR="${MMA_OUT_DIR:-${OUT_ROOT}/ncu_mma_best}"
CUBLASLT_OUT_DIR="${CUBLASLT_OUT_DIR:-${OUT_ROOT}/ncu_cublaslt}"

mkdir -p "${MMA_OUT_DIR}" "${CUBLASLT_OUT_DIR}" "${NCU_TMPDIR}"
export TMPDIR="${NCU_TMPDIR}"

NCU_LOCK_FILE="/tmp/nsight-compute-lock"
if [[ -e "${NCU_LOCK_FILE}" && ! -w "${NCU_LOCK_FILE}" ]]; then
  echo "[ERROR] Nsight Compute lock file exists but is not writable: ${NCU_LOCK_FILE}"
  echo "Current state:"
  ls -l "${NCU_LOCK_FILE}" || true
  echo
  echo "Fix on the server, then rerun:"
  echo "  ls -l ${NCU_LOCK_FILE}"
  echo "  rm -f ${NCU_LOCK_FILE}      # if you own it"
  echo "  sudo rm -f ${NCU_LOCK_FILE} # if it is owned by another user/root"
  exit 1
fi

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] binary not found: ${BIN}"
  echo "Please build first: cmake -S . -B ${BUILD_DIR} && cmake --build ${BUILD_DIR} -j"
  exit 1
fi

if ! command -v "${NCU_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Nsight Compute CLI not found: ${NCU_BIN}"
  echo "Set NCU_BIN=/path/to/ncu if it is not in PATH."
  exit 1
fi

run_one() {
  local impl="$1"
  local size="$2"
  local out_dir="$3"
  local tag="$4"
  local ts
  local out_txt

  ts="$(date +%Y%m%d_%H%M%S)"
  out_txt="${out_dir}/${tag}_${size}_${ts}.txt"

  echo "=== ncu impl=${impl}, size=${size} ==="
  echo "[info] output: ${out_txt}"

  "${NCU_BIN}" \
    --target-processes "${NCU_TARGET_PROCESSES}" \
    --set "${NCU_SET}" \
    --page "${NCU_PAGE}" \
    --print-summary "${NCU_PRINT_SUMMARY}" \
    --log-file "${out_txt}" \
    "${BIN}" \
      --impl "${impl}" \
      --M "${size}" --N "${size}" --K "${size}" \
      --warmup "${NCU_WARMUP}" --repeat "${NCU_REPEAT}" \
      --no-check
}

echo "# Nsight Compute text profile"
echo "# binary=${BIN}"
echo "# ncu_bin=${NCU_BIN}"
echo "# ncu_set=${NCU_SET}"
echo "# ncu_page=${NCU_PAGE}"
echo "# sizes=${NCU_SIZES}"
echo "# tmpdir=${TMPDIR}"
echo "# mma_best_impl=${MMA_BEST_IMPL}"
echo "# cublaslt_impl=${CUBLASLT_IMPL}"
echo

for size in ${NCU_SIZES}; do
  run_one "${MMA_BEST_IMPL}" "${size}" "${MMA_OUT_DIR}" "mma_best"
  echo
  run_one "${CUBLASLT_IMPL}" "${size}" "${CUBLASLT_OUT_DIR}" "cublaslt"
  echo
done

echo "[DONE] NCU text outputs saved under ${MMA_OUT_DIR} and ${CUBLASLT_OUT_DIR}"
