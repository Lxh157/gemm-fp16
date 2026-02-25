#!/usr/bin/env bash
set -euo pipefail

echo "=== date ==="
date

echo "=== uname ==="
uname -a

echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== nvcc ==="
nvcc --version || true

echo "=== gcc ==="
gcc --version | head -n 1 || true
