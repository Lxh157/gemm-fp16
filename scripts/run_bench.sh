#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/run_bench.sh
#   PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
#   PROFILE_SET=phase1_4060_all bash scripts/run_bench.sh
#   BUILD_DIR=build WARMUP=3 REPEAT=10 bash scripts/run_bench.sh
#   SIZES_OVERRIDE="1024 2048 4096" PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
#   CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh
#   CUDA_VISIBLE_DEVICES=1 CHECK_MAX_SIZE=256 PROFILE_SET=phase2_4090_tc bash scripts/run_bench.sh

BUILD_DIR="${BUILD_DIR:-build}"
BIN="${BUILD_DIR}/bench_gemm"
OUT_DIR="${OUT_DIR:-results/raw}"
LOG_DIR="${LOG_DIR:-logs}"

WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-10}"
PROFILE_SET="${PROFILE_SET:-phase2_4090_tc}"
CHECK_MAX_SIZE="${CHECK_MAX_SIZE:-2048}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if [[ ! -x "${BIN}" ]]; then
  echo "[ERROR] binary not found: ${BIN}"
  echo "Please build first: cmake -S . -B build && cmake --build build -j"
  exit 1
fi

case "${PROFILE_SET}" in
  phase1_4060_all)
    OUT_PREFIX="bench_phase1_4060_all"
    if [[ -n "${SIZES_OVERRIDE:-}" ]]; then
      read -r -a SIZES <<< "${SIZES_OVERRIDE}"
    else
      SIZES=(256 512 768 1024 1280 1536 1792 2048 2304 2560 2816 3072 3328 3584 3840 4096)
    fi
    IMPLS=(
      naive
      tiled
      tiled_rb1x4
      tiled_rb2x4
      thread_tiled_1d
      cublas
      cublaslt
      # tiled_fp16acc
      # tiled_fp16acc_rb1x4
      # tiled_fp16acc_rb2x4
      # wmma_fp16acc
      # wmma_fp16acc_staged
      # wmma_fp16acc_staged_db
      # wmma_fp16acc_staged_cpasync
      # cublas_gemmex_fp16acc
      # cublaslt_fp16acc
    )
    ;;
  phase2_4090_tc)
    OUT_PREFIX="bench_phase2_4090_tc"
    if [[ -n "${SIZES_OVERRIDE:-}" ]]; then
      read -r -a SIZES <<< "${SIZES_OVERRIDE}"
    else
      SIZES=(256 512 768 1024 1280 1536 1792 2048 2304 2560 2816 3072 3328 3584 3840 4096)
    fi
    IMPLS=(
      # wmma_fp16acc_staged_cpasync
      # wmma_fp16acc_staged_cpasync_k32
      # wmma_fp16acc_staged_cpasync_k32_4x2
      # wmma_fp16acc_staged_cpasync_k32_split
      # wmma_fp16acc_staged_cpasync_k32_skew16
      # wmma_fp16acc_staged_cpasync_k32_4x4_skew16
      # wmma_fp16acc_staged_cpasync_k32_4x8_skew16
      # wmma_fp16acc_staged_cpasync_k32_skewA16_B8
      # wmma_fp16acc_staged_cpasync_k32_skewA8_B16
      # wmma_fp16acc_staged_cpasync_k32_skewA24_B16
      # wmma_fp16acc_staged_cpasync_k32_skewA8_B24
      # wmma_fp16acc_staged_cpasync_k32_skewA8_B32
      # wmma_fp16acc_staged_cpasync_k32_skewA16_B32
      # wmma_fp16acc_staged_cpasync_k32_skewA24_B32

      # wmma_fp16acc_staged_cpasync_k64
      # wmma_fp16acc_staged_cpasync_k64
      wmma_fp16acc_staged_cpasync_k64_skew16
      wmma_fp16acc_staged_cpasync_k64_4x4_skew16
      # wmma_fp16acc_staged_cpasync_k64_4x4_128B_skew16
      # wmma_fp16acc_staged_cpasync_k64_4x4_skew16_swizzle_n4
      # wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch
      # wmma_fp16acc_staged_cpasync_k64_4x4_skew16_prefetch2
      # wmma_fp16acc_staged_cpasync_k64_4x8_skew16
      # wmma_fp16acc_staged_cpasync_k64_4x4_skewA16_B32

      mma_fp16acc
      mma_fp16acc_m16n32
      mma_fp16acc_m16n32_k32
      mma_fp16acc_m16n32_k32_4x2
      mma_fp16acc_m16n32_k32_4x4
      mma_fp16acc_m16n32_staged_cpasync_k32_4x2
      mma_fp16acc_m16n32_staged_cpasync_k64_4x2
      mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew16
      mma_fp16acc_m16n32_staged_cpasync_k64_4x2_skew32
      mma_fp16acc_m16n32_staged_cpasync_k64_4x4_skew16

      # cublas_gemmex_fp16acc
      cublaslt_fp16acc
    )
    ;;
  *)
    echo "[ERROR] unknown PROFILE_SET=${PROFILE_SET}"
    echo "Supported PROFILE_SET values: phase1_4060_all, phase2_4090_tc"
    exit 1
    ;;
esac

TS=$(date +%Y%m%d_%H%M%S)
OUT_TXT="${OUT_DIR}/${OUT_PREFIX}_${TS}.txt"

{
  echo "# bench_gemm batch run"
  echo "# time: $(date)"
  echo "# profile_set=${PROFILE_SET}"
  echo "# build_dir=${BUILD_DIR}"
  echo "# binary=${BIN}"
  echo "# warmup=${WARMUP}, repeat=${REPEAT}, metric=median"
  echo "# check_max_size=${CHECK_MAX_SIZE}"
  echo "# continue_on_error=${CONTINUE_ON_ERROR}"
  echo "# extra_bench_args=${EXTRA_BENCH_ARGS}"
  echo "# sizes=${SIZES[*]}"
  echo "# impls=${IMPLS[*]}"
  echo
} | tee "${OUT_TXT}"

FAIL_COUNT=0

for s in "${SIZES[@]}"; do
  for impl in "${IMPLS[@]}"; do
    echo "===== impl=${impl}, M=N=K=${s} =====" | tee -a "${OUT_TXT}"

    cmd=(
      "${BIN}"
      --impl "${impl}"
      --M "${s}" --N "${s}" --K "${s}"
      --warmup "${WARMUP}" --repeat "${REPEAT}"
    )

    if (( CHECK_MAX_SIZE >= 0 && s > CHECK_MAX_SIZE )); then
      echo "[info] size ${s} exceeds CHECK_MAX_SIZE=${CHECK_MAX_SIZE}, append --no-check" | tee -a "${OUT_TXT}"
      cmd+=(--no-check)
    fi

    if [[ -n "${EXTRA_BENCH_ARGS}" ]]; then
      read -r -a extra_args <<< "${EXTRA_BENCH_ARGS}"
      cmd+=("${extra_args[@]}")
    fi

    if "${cmd[@]}" 2>&1 | tee -a "${OUT_TXT}"; then
      :
    else
      FAIL_COUNT=$((FAIL_COUNT + 1))
      echo "[warn] impl=${impl}, size=${s} failed" | tee -a "${OUT_TXT}"
      if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        exit 1
      fi
    fi

    echo | tee -a "${OUT_TXT}"
  done
done


if (( FAIL_COUNT > 0 )); then
  echo "[WARN] completed with ${FAIL_COUNT} failed case(s); see ${OUT_TXT}"
  exit 1
fi

echo "[DONE] results saved to ${OUT_TXT}"
