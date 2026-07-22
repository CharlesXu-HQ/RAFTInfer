#!/usr/bin/env bash
set -euo pipefail
cmake -S . -B build/host -G Ninja -DBRT_ENABLE_CUDA=OFF
cmake --build build/host
ctest --test-dir build/host --output-on-failure
cargo fmt --check
cargo test --workspace
