#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
cmake -S "${repo_root}" -B "${repo_root}/build/host" -G Ninja -DRAFTINFER_ENABLE_CUDA=OFF
cmake --build "${repo_root}/build/host"
ctest --test-dir "${repo_root}/build/host" --output-on-failure
"${repo_root}/tests/native-library-type-test.sh"
"${repo_root}/tests/parity-script-test.sh"
"${repo_root}/tests/benchmark-script-test.sh"
"${repo_root}/tests/bf16-gate-script-test.sh"
python3 "${repo_root}/tests/benchmark-chart-test.py"
"${repo_root}/tests/benchmark-asset-test.sh"
"${repo_root}/tests/prepare-qwen35-gguf-test.sh"
"${repo_root}/tests/public-surface-test.sh"
"${repo_root}/scripts/check-project-brand.sh"
python3 "${repo_root}/tests/readme-links-test.py"
cargo fmt --check
cargo test --workspace
