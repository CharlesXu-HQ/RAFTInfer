#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/raftinfer-public-surface.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

build_dir="${work_dir}/build"
cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=OFF \
  -DRAFTINFER_BUILD_TESTS=OFF
for expected_cache_entry in \
  'RAFTINFER_ENABLE_CUDA:BOOL=OFF' \
  'RAFTINFER_BUILD_TESTS:BOOL=OFF'; do
  grep -Fx "${expected_cache_entry}" "${build_dir}/CMakeCache.txt" >/dev/null || {
    printf 'public-surface: CMake did not consume %s\n' \
      "${expected_cache_entry%%:*}" >&2
    exit 1
  }
done
cmake --build "${build_dir}" --target raftinfer_cpp
install_prefix="${work_dir}/install"
cmake --install "${build_dir}" --prefix "${install_prefix}"

consumer_dir="${work_dir}/consumer"
mkdir -p "${consumer_dir}"
cat >"${consumer_dir}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(raftinfer_public_consumer LANGUAGES C CXX)

find_path(RAFTINFER_INCLUDE_DIR raftinfer/c_api.h
  PATHS "${RAFTINFER_PREFIX}/include"
  NO_DEFAULT_PATH
  REQUIRED)
find_library(RAFTINFER_LIBRARY raftinfer_cpp
  PATHS "${RAFTINFER_PREFIX}/lib"
  NO_DEFAULT_PATH
  REQUIRED)

add_executable(raftinfer_public_consumer consumer.c)
target_include_directories(raftinfer_public_consumer PRIVATE
  "${RAFTINFER_INCLUDE_DIR}")
target_link_libraries(raftinfer_public_consumer PRIVATE "${RAFTINFER_LIBRARY}")
set_property(TARGET raftinfer_public_consumer PROPERTY LINKER_LANGUAGE CXX)
EOF

cat >"${consumer_dir}/consumer.c" <<'EOF'
#include <raftinfer/c_api.h>

#include <stddef.h>

int main(void) {
  RaftInferEngineConfig config = {
      .struct_size = sizeof(config),
      .device_id = 0,
      .initial_pool_bytes = 0,
  };
  RaftInferEngineHandle *engine = NULL;
  RaftInferStatus status = raftinfer_engine_create(&config, &engine);
  if (status.code != RAFTINFER_STATUS_OK || engine == NULL) {
    return 1;
  }
  raftinfer_engine_destroy(engine);
  return 0;
}
EOF

cmake -S "${consumer_dir}" -B "${consumer_dir}/build" -G Ninja \
  -DRAFTINFER_PREFIX="${install_prefix}"
cmake --build "${consumer_dir}/build"
"${consumer_dir}/build/raftinfer_public_consumer"

CARGO_TARGET_DIR="${work_dir}/cargo-target" \
  cargo metadata --manifest-path "${repo_root}/Cargo.toml" \
  --format-version 1 --no-deps | python3 -c '
import json
import sys

names = {package["name"] for package in json.load(sys.stdin)["packages"]}
expected = {"raftinfer-sys", "raftinfer-runtime", "raftinfer-cli"}
if names != expected:
    raise SystemExit(f"unexpected workspace packages: {sorted(names)}")
'

CARGO_TARGET_DIR="${work_dir}/cargo-target" \
  cargo build --manifest-path "${repo_root}/Cargo.toml" -p raftinfer-cli
"${work_dir}/cargo-target/debug/raftinfer" info | grep -Fx 'backend=host'
