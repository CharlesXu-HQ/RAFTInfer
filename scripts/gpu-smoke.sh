#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
golden_result="${repo_root}/tests/golden/smoke_result.json"
lock_dir="${repo_root}/build"
host_uid="$(id -u)"
host_gid="$(id -g)"
if ! [[ "${host_uid}" =~ ^[1-9][0-9]*$ && "${host_gid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BRT GPU smoke refused: host UID/GID must be numeric and non-root" >&2
  exit 32
fi
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
if ! container_group_gid="$(docker run --rm --entrypoint id brt-dev:26.06-cuda13 -g)"; then
  echo "BRT GPU smoke refused: unable to resolve the image default group" >&2
  exit 33
fi
if ! [[ "${container_group_gid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BRT GPU smoke refused: invalid image default group" >&2
  exit 33
fi
run_dir="$(mktemp -d "${lock_dir}/brt-gpu-run.XXXXXX")"
result_file="$(mktemp "${run_dir}/brt-smoke-result.XXXXXX")"
trap 'rm -rf "${run_dir}"' EXIT

if docker run --rm --gpus device=0 \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${repo_root}:/workspace" -w /workspace \
  -v "${run_dir}:/brt-run" \
  --user "${host_uid}:${host_gid}" \
  --group-add "${container_group_gid}" \
  -e HOME=/tmp/brt-home \
  -e XDG_CACHE_HOME=/tmp/brt-home/.cache \
  -e "BRT_MIN_FREE_MIB=${minimum_free_mib}" \
  -e "BRT_MAX_UTILIZATION_PERCENT=${maximum_utilization}" \
  brt-dev:26.06-cuda13 \
  bash -c 'mkdir -p "${HOME}" "${XDG_CACHE_HOME}" && nvcc_path="$(command -v nvcc)" && test -x "${nvcc_path}" && cmake -S . -B /brt-run/build -G Ninja -DBRT_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="${nvcc_path}" >&2 && cmake --build /brt-run/build --target brt-smoke >&2 && scripts/gpu-preflight.sh >/dev/null && /brt-run/build/cpp/brt-smoke' >"${result_file}"; then
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
