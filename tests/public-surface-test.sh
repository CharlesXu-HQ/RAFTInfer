#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/raftinfer-public-surface.XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

build_dir="${work_dir}/build"
cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=OFF \
  -DRAFTINFER_BUILD_TESTS=OFF
cmake --build "${build_dir}" --target raftinfer_cpp

cat >"${work_dir}/consumer.c" <<'EOF'
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

cc -std=c11 -I"${repo_root}/cpp/include" -c "${work_dir}/consumer.c" \
  -o "${work_dir}/consumer.o"
c++ "${work_dir}/consumer.o" "${build_dir}/cpp/libraftinfer_cpp.a" \
  -o "${work_dir}/consumer"
"${work_dir}/consumer"

cargo metadata --format-version 1 --no-deps | python3 -c '
import json
import sys

names = {package["name"] for package in json.load(sys.stdin)["packages"]}
expected = {"raftinfer-sys", "raftinfer-runtime", "raftinfer-cli"}
if names != expected:
    raise SystemExit(f"unexpected workspace packages: {sorted(names)}")
'

cargo build -p raftinfer-cli
"${repo_root}/target/debug/raftinfer" info | grep -Fx 'backend=host'
