#!/usr/bin/env bash
set -euo pipefail

# 一次性构建 + 跑 run_bench.sh + raw_to_csv.py + plot.py + NCU。
# 自动选择一张空闲 RTX 4090。判定口径：
#   1) GPU 名称包含 4090
#   2) 没有 compute process
#   3) memory.used <= EMPTY_MEM_MB，默认 1024 MiB

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EMPTY_MEM_MB="${EMPTY_MEM_MB:-1024}"
BUILD_DIR="${BUILD_DIR:-build}"
RUN_NCU="${RUN_NCU:-1}"
GPU_NAME_FILTER="${GPU_NAME_FILTER:-H100}"
PROFILE_SET="${PROFILE_SET:-h100_wgmma}"
CHECK_MAX_SIZE="${CHECK_MAX_SIZE:-256}"

echo "=== nvidia-smi ==="
nvidia-smi
echo

mapfile -t GPU_LINES < <(
  nvidia-smi --query-gpu=index,name,memory.used,pci.bus_id \
    --format=csv,noheader,nounits
)

mapfile -t BUSY_BUSES < <(
  nvidia-smi --query-compute-apps=gpu_bus_id \
    --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | sort -u || true
)

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "${s}"
}

bus_is_busy() {
  local bus="$1"
  local busy_bus
  for busy_bus in "${BUSY_BUSES[@]}"; do
    if [[ "$(trim "${busy_bus}")" == "${bus}" ]]; then
      return 0
    fi
  done
  return 1
}

SELECTED_GPU=""

echo "=== GPU availability check ==="
for line in "${GPU_LINES[@]}"; do
  IFS=',' read -r gpu_idx gpu_name mem_used bus_id <<< "${line}"
  gpu_idx="$(trim "${gpu_idx}")"
  gpu_name="$(trim "${gpu_name}")"
  mem_used="$(trim "${mem_used}")"
  bus_id="$(trim "${bus_id}")"

  if [[ -n "${GPU_NAME_FILTER}" && "${gpu_name}" != *"${GPU_NAME_FILTER}"* ]]; then
    echo "[skip] gpu=${gpu_idx}, name=${gpu_name}, reason=name does not contain ${GPU_NAME_FILTER}"
    continue
  fi

  if bus_is_busy "${bus_id}"; then
    echo "[busy] gpu=${gpu_idx}, name=${gpu_name}, memory.used=${mem_used} MiB, reason=compute process"
    continue
  fi

  if (( mem_used > EMPTY_MEM_MB )); then
    echo "[busy] gpu=${gpu_idx}, name=${gpu_name}, memory.used=${mem_used} MiB, reason=memory>${EMPTY_MEM_MB}MiB"
    continue
  fi

  echo "[free] gpu=${gpu_idx}, name=${gpu_name}, memory.used=${mem_used} MiB"
  SELECTED_GPU="${gpu_idx}"
  break
done

if [[ -z "${SELECTED_GPU}" ]]; then
  echo "[ERROR] no free GPU matched GPU_NAME_FILTER=${GPU_NAME_FILTER}; benchmark cannot run now."
  exit 1
fi

echo
echo "=== build bench_gemm ==="
cmake -S . -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j

echo
echo "=== run benchmark on physical GPU ${SELECTED_GPU} ==="
CUDA_VISIBLE_DEVICES="${SELECTED_GPU}" \
BUILD_DIR="${BUILD_DIR}" \
CHECK_MAX_SIZE="${CHECK_MAX_SIZE}" \
PROFILE_SET="${PROFILE_SET}" \
bash scripts/run_bench.sh

echo
echo "=== extract latest raw result to csv ==="
python3 scripts/raw_to_csv.py

echo
echo "=== generate plots ==="
python3 scripts/plot.py

if [[ "${RUN_NCU}" == "1" ]]; then
  echo
  echo "=== run NCU comparison on physical GPU ${SELECTED_GPU} ==="
  CUDA_VISIBLE_DEVICES="${SELECTED_GPU}" \
  BUILD_DIR="${BUILD_DIR}" \
  bash scripts/run_ncu_compare.sh
else
  echo
  echo "=== skip NCU comparison because RUN_NCU=${RUN_NCU} ==="
fi
