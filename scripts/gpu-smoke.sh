#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${repo_root}/scripts/gpu-preflight.sh"
docker run --rm --gpus all \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${repo_root}:/workspace" -w /workspace \
  brt-dev:26.06-cuda13 \
  bash -lc 'cmake -S . -B build/gpu -G Ninja -DBRT_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build/gpu --target brt-smoke && ./build/gpu/cpp/brt-smoke'
