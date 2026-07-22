#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
golden_result="${repo_root}/tests/golden/smoke_result.json"
lock_dir="${repo_root}/build"
mkdir -p "${lock_dir}"
exec 9>"${lock_dir}/brt-gpu-smoke.lock"
if ! flock -n 9; then
  echo "BRT GPU smoke refused: another BRT GPU smoke run holds the project lock" >&2
  exit 30
fi

minimum_free_mib="${BRT_MIN_FREE_MIB:-2048}"
maximum_utilization="${BRT_MAX_UTILIZATION_PERCENT:-5}"
BRT_MIN_FREE_MIB="${minimum_free_mib}" \
  BRT_MAX_UTILIZATION_PERCENT="${maximum_utilization}" \
  "${repo_root}/scripts/gpu-preflight.sh" >/dev/null
result_file="$(mktemp "${lock_dir}/brt-smoke-result.XXXXXX")"
trap 'rm -f "${result_file}"' EXIT

if docker run --rm --gpus device=0 \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${repo_root}:/workspace" -w /workspace \
  -e "BRT_MIN_FREE_MIB=${minimum_free_mib}" \
  -e "BRT_MAX_UTILIZATION_PERCENT=${maximum_utilization}" \
  brt-dev:26.06-cuda13 \
  bash -lc 'cmake -S . -B build/gpu -G Ninja -DBRT_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release >&2 && cmake --build build/gpu --target brt-smoke >&2 && scripts/gpu-preflight.sh >/dev/null && ./build/gpu/cpp/brt-smoke' >"${result_file}"; then
  :
else
  docker_status=$?
  exit "${docker_status}"
fi
if ! cmp -s "${golden_result}" "${result_file}"; then
  diff -u "${golden_result}" "${result_file}" >&2 || true
  exit 31
fi
cat "${result_file}"
