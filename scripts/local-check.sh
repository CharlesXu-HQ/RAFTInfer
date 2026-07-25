#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
cmake -S "${repo_root}" -B "${repo_root}/build/host" -G Ninja -DBRT_ENABLE_CUDA=OFF
cmake --build "${repo_root}/build/host"
ctest --test-dir "${repo_root}/build/host" --output-on-failure
"${repo_root}/tests/native-library-type-test.sh"
"${repo_root}/tests/parity-script-test.sh"
"${repo_root}/tests/benchmark-script-test.sh"
"${repo_root}/tests/prepare-qwen35-gguf-test.sh"
cargo fmt --check
cargo test --workspace
