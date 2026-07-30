#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/raftinfer-public-surface.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

off_build_dir="${work_dir}/build-off"
cmake -S "${repo_root}" -B "${off_build_dir}" -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=OFF \
  -DRAFTINFER_BUILD_TESTS=OFF
if ! ctest --test-dir "${off_build_dir}" -N | grep -Fq 'Total Tests: 0'; then
  printf 'public-surface: RAFTINFER_BUILD_TESTS=OFF registered CTest tests\n' \
    >&2
  exit 1
fi
if cmake --build "${off_build_dir}" --target help | \
  grep -Fq 'raftinfer_c_api_test'; then
  printf 'public-surface: RAFTINFER_BUILD_TESTS=OFF exposed raftinfer_c_api_test\n' \
    >&2
  exit 1
fi
if cmake --build "${off_build_dir}" --target help | \
  grep -Fq 'raftinfer-qwen35-logits'; then
  printf 'public-surface: RAFTINFER_ENABLE_CUDA=OFF exposed CUDA target\n' >&2
  exit 1
fi

cuda_on_build_dir="${work_dir}/build-cuda-on"
cuda_on_log="${work_dir}/cuda-on-configure.log"
set +e
cmake -S "${repo_root}" -B "${cuda_on_build_dir}" -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=ON \
  -DRAFTINFER_BUILD_TESTS=OFF >"${cuda_on_log}" 2>&1
cuda_on_status=$?
set -e
if [[ "${cuda_on_status}" -eq 0 ]]; then
  if ! cmake --build "${cuda_on_build_dir}" --target help | \
    grep -Fq 'raftinfer-qwen35-logits'; then
    cat "${cuda_on_log}" >&2
    printf 'public-surface: RAFTINFER_ENABLE_CUDA=ON configured without CUDA target\n' \
      >&2
    exit 1
  fi
else
  if grep -Fq 'Manually-specified variables were not used' "${cuda_on_log}" && \
    grep -Fq 'RAFTINFER_ENABLE_CUDA' "${cuda_on_log}"; then
    cat "${cuda_on_log}" >&2
    printf 'public-surface: RAFTINFER_ENABLE_CUDA=ON was not consumed\n' >&2
    exit 1
  fi
  if grep -Eiq 'unknown (CMake )?option|unknown option.*RAFTINFER_ENABLE_CUDA|RAFTINFER_ENABLE_CUDA.*unknown option' \
    "${cuda_on_log}"; then
    cat "${cuda_on_log}" >&2
    printf 'public-surface: RAFTINFER_ENABLE_CUDA=ON was rejected as an unknown option\n' \
      >&2
    exit 1
  fi
  if ! grep -Eiq 'Could not find CMAKE_CUDA_COMPILER|No CMAKE_CUDA_COMPILER could be found|Failed to find nvcc|Could NOT find CUDAToolkit|Could not find a package configuration file provided by "(CUDAToolkit|raft|rmm)"|Could NOT find (raft|rmm)' \
    "${cuda_on_log}"; then
    cat "${cuda_on_log}" >&2
    printf 'public-surface: RAFTINFER_ENABLE_CUDA=ON failed for an unexpected reason\n' \
      >&2
    exit 1
  fi
fi

on_build_dir="${work_dir}/build-on"
cmake -S "${repo_root}" -B "${on_build_dir}" -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=OFF \
  -DRAFTINFER_BUILD_TESTS=ON
if ! ctest --test-dir "${on_build_dir}" -N | \
  grep -Fq 'raftinfer_c_api_test'; then
  printf 'public-surface: RAFTINFER_BUILD_TESTS=ON did not register raftinfer_c_api_test\n' \
    >&2
  exit 1
fi
if ! cmake --build "${on_build_dir}" --target help | \
  grep -Fq 'raftinfer_c_api_test'; then
  printf 'public-surface: RAFTINFER_BUILD_TESTS=ON did not expose raftinfer_c_api_test\n' \
    >&2
  exit 1
fi

cmake --build "${off_build_dir}" --target raftinfer_cpp
install_prefix="${work_dir}/install"
cmake --install "${off_build_dir}" --prefix "${install_prefix}"

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
