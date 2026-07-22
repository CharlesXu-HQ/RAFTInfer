#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
cmake -S "${repo_root}" -B "${repo_root}/build/host" -G Ninja -DBRT_ENABLE_CUDA=OFF
cmake --build "${repo_root}/build/host"
ctest --test-dir "${repo_root}/build/host" --output-on-failure
cargo fmt --check
cargo test --workspace
