#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_dir="$(mktemp -d)"
host_uid=1000
host_gid=1000
golden_json='{"device_id":0,"element_count":1024,"checksum":523776}'
golden_output="${golden_json}"$'\n'
trap 'rm -rf "${fixture_dir}"' EXIT

cat >"${fixture_dir}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${RAFTINFER_TEST_NVIDIA_LOG}"
case "$*" in
  *query-compute-apps*)
    if [[ "${RAFTINFER_TEST_NVIDIA_MODE:-}" == fail_compute ]]; then
      exit 70
    fi
    printf '%s' "${RAFTINFER_TEST_COMPUTE_APPS:-}"
    ;;
  *query-gpu*)
    if [[ "${RAFTINFER_TEST_NVIDIA_MODE:-}" == fail_gpu ]]; then
      exit 71
    fi
    printf '%s' "${RAFTINFER_TEST_GPU_ROW}"
    ;;
  *) exit 99 ;;
esac
EOF

cat >"${fixture_dir}/flock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${RAFTINFER_TEST_FLOCK_LOG}"
exit "${RAFTINFER_TEST_FLOCK_STATUS:-0}"
EOF

cat >"${fixture_dir}/id" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  -u) printf '%s\n' "${RAFTINFER_TEST_HOST_UID}" ;;
  -g) printf '%s\n' "${RAFTINFER_TEST_HOST_GID}" ;;
  *) exit 99 ;;
esac
EOF

cat >"${fixture_dir}/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >>"${RAFTINFER_TEST_DOCKER_LOG}"
arguments=("$@")
has_gpu=0
for argument in "${arguments[@]}"; do
  if [[ "${argument}" == --gpus ]]; then
    has_gpu=1
    break
  fi
done
if [[ " ${arguments[*]} " == *' --entrypoint stat '* ]]; then
  printf '%s\n' "${RAFTINFER_TEST_CONDA_GID}"
  exit "${RAFTINFER_TEST_CONDA_GID_STATUS:-0}"
fi
if [[ "${has_gpu}" -eq 1 ]]; then
  printf '%s\n' "$@" >>"${RAFTINFER_TEST_DOCKER_GPU_LOG}"
fi
for ((index = 0; index < ${#arguments[@]}; index++)); do
  if [[ "${arguments[index]}" == -e ]]; then
    index=$((index + 1))
    case "${arguments[index]}" in
      RAFTINFER_MIN_FREE_MIB=*|RAFTINFER_MAX_UTILIZATION_PERCENT=*)
        export "${arguments[index]}"
        ;;
    esac
  fi
done
for argument in "${arguments[@]}"; do
  if [[ "${argument}" == *'scripts/gpu-preflight.sh >/dev/null && /raftinfer-run/build/cpp/raftinfer-smoke'* ]]; then
    RAFTINFER_TEST_COMPUTE_APPS="${RAFTINFER_TEST_INNER_COMPUTE_APPS:-${RAFTINFER_TEST_COMPUTE_APPS:-}}" \
      "${RAFTINFER_TEST_REPO_ROOT}/scripts/gpu-preflight.sh" >/dev/null
    break
  fi
done
if [[ "${RAFTINFER_TEST_DOCKER_STATUS:-0}" -ne 0 ]]; then
  exit "${RAFTINFER_TEST_DOCKER_STATUS}"
fi
printf '%s' "${RAFTINFER_TEST_DOCKER_OUTPUT}"
EOF

cat >"${fixture_dir}/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >>"${RAFTINFER_TEST_SSH_LOG}"
EOF

cat >"${fixture_dir}/rsync" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >>"${RAFTINFER_TEST_RSYNC_LOG}"
EOF

chmod +x "${fixture_dir}"/{nvidia-smi,flock,id,docker,ssh,rsync}

reset_logs() {
  : >"${fixture_dir}/nvidia.log"
  : >"${fixture_dir}/flock.log"
  : >"${fixture_dir}/docker.log"
  : >"${fixture_dir}/docker-gpu.log"
  : >"${fixture_dir}/ssh.log"
  : >"${fixture_dir}/rsync.log"
}

assert_status() {
  local expected="$1"
  shift
  set +e
  "$@" >"${fixture_dir}/stdout" 2>"${fixture_dir}/stderr"
  local actual=$?
  set -e
  [[ "${actual}" -eq "${expected}" ]] || {
    printf 'expected exit %s, got %s\n' "${expected}" "${actual}" >&2
    cat "${fixture_dir}/stdout" "${fixture_dir}/stderr" >&2
    exit 1
  }
}

run_preflight() {
  local compute_apps="$1"
  local gpu_row="$2"
  local nvidia_mode="${3:-}"
  local min_free="${4:-}"
  local max_utilization="${5:-}"
  PATH="${fixture_dir}:${PATH}" \
    RAFTINFER_TEST_NVIDIA_LOG="${fixture_dir}/nvidia.log" \
    RAFTINFER_TEST_COMPUTE_APPS="${compute_apps}" \
    RAFTINFER_TEST_GPU_ROW="${gpu_row}" \
    RAFTINFER_TEST_NVIDIA_MODE="${nvidia_mode}" \
    RAFTINFER_MIN_FREE_MIB="${min_free}" \
    RAFTINFER_MAX_UTILIZATION_PERCENT="${max_utilization}" \
    "${repo_root}/scripts/gpu-preflight.sh"
}

assert_preflight() {
  local expected="$1"
  shift
  reset_logs
  assert_status "${expected}" run_preflight "$@"
}

assert_preflight 0 '' '8192, 0, 42' '' ''
grep -Fx 'gpu_preflight=ok free_mib=8192 utilization=0 temperature=42' "${fixture_dir}/stdout"
gpu_zero_query_count="$(grep -Fc -- '--id=0' "${fixture_dir}/nvidia.log" || true)"
[[ "${gpu_zero_query_count}" -eq 2 ]] || {
  printf 'expected both nvidia-smi queries to select GPU 0, found %s\n' "${gpu_zero_query_count}" >&2
  exit 1
}

assert_preflight 20 '1234, training, 1024' '8192, 0, 42' '' ''
grep -Fqx 'RAFTINFER GPU preflight refused: active compute applications detected' "${fixture_dir}/stderr"
assert_preflight 21 '' '1024, 0, 42' '' ''
grep -Fqx 'RAFTINFER GPU preflight refused: free=1024MiB required=2048MiB' "${fixture_dir}/stderr"
assert_preflight 22 '' '8192, 6, 42' '' ''
grep -Fqx 'RAFTINFER GPU preflight refused: utilization=6%' "${fixture_dir}/stderr"

for malformed_row in '' '8192, 0' '8192, 0, 42,' '8192, 0, 42, 1' $'8192, 0, 42\n8192, 0, 43' 'abc, 0, 42' '-1, 0, 42' '8192, -1, 42' '8192, 0, -1'; do
  assert_preflight 23 '' "${malformed_row}" '' ''
done
assert_preflight 23 '' '8192, 0, 42' fail_compute '' ''
assert_preflight 23 '' '8192, 0, 42' fail_gpu '' ''
assert_preflight 23 '' '8192, 0, 42' '' invalid ''
assert_preflight 23 '' '8192, 0, 42' '' -1 ''
assert_preflight 23 '' '8192, 0, 42' '' 0 ''
assert_preflight 23 '' '8192, 0, 42' '' '' invalid
assert_preflight 23 '' '8192, 0, 42' '' '' -1
assert_preflight 23 '' '8192, 0, 42' '' '' 101

run_smoke() {
  PATH="${fixture_dir}:${PATH}" \
    RAFTINFER_TEST_NVIDIA_LOG="${fixture_dir}/nvidia.log" \
    RAFTINFER_TEST_FLOCK_LOG="${fixture_dir}/flock.log" \
    RAFTINFER_TEST_DOCKER_LOG="${fixture_dir}/docker.log" \
    RAFTINFER_TEST_DOCKER_GPU_LOG="${fixture_dir}/docker-gpu.log" \
    RAFTINFER_TEST_REPO_ROOT="${repo_root}" \
    RAFTINFER_TEST_HOST_UID="${RAFTINFER_TEST_HOST_UID:-${host_uid}}" \
    RAFTINFER_TEST_HOST_GID="${RAFTINFER_TEST_HOST_GID:-${host_gid}}" \
    RAFTINFER_TEST_CONDA_GID="${RAFTINFER_TEST_CONDA_GID:-1001}" \
    RAFTINFER_TEST_CONDA_GID_STATUS="${RAFTINFER_TEST_CONDA_GID_STATUS:-0}" \
    RAFTINFER_TEST_COMPUTE_APPS="${RAFTINFER_TEST_COMPUTE_APPS:-}" \
    RAFTINFER_TEST_INNER_COMPUTE_APPS="${RAFTINFER_TEST_INNER_COMPUTE_APPS:-}" \
    RAFTINFER_TEST_GPU_ROW="${RAFTINFER_TEST_GPU_ROW:-8192, 0, 42}" \
    RAFTINFER_TEST_FLOCK_STATUS="${RAFTINFER_TEST_FLOCK_STATUS:-0}" \
    RAFTINFER_TEST_DOCKER_OUTPUT="${RAFTINFER_TEST_DOCKER_OUTPUT:-${golden_output}}" \
    RAFTINFER_TEST_DOCKER_STATUS="${RAFTINFER_TEST_DOCKER_STATUS:-0}" \
    "${repo_root}/scripts/gpu-smoke.sh"
}

assert_smoke() {
  local expected="$1"
  shift
  reset_logs
  set +e
  (
    for setting in "$@"; do
      export "${setting}"
    done
    run_smoke
  ) >"${fixture_dir}/stdout" 2>"${fixture_dir}/stderr"
  local actual=$?
  set -e
  [[ "${actual}" -eq "${expected}" ]] || {
    printf 'expected exit %s, got %s\n' "${expected}" "${actual}" >&2
    cat "${fixture_dir}/stdout" "${fixture_dir}/stderr" >&2
    exit 1
  }
}

assert_docker_flag_value() {
  local log_file="$1"
  local flag="$2"
  local expected_value="$3"
  local previous=''
  local argument=''
  while IFS= read -r argument; do
    if [[ "${previous}" == "${flag}" ]]; then
      [[ "${argument}" == "${expected_value}" ]] && return 0
      break
    fi
    previous="${argument}"
  done <"${log_file}"
  printf 'expected Docker flag %s to be immediately followed by %s\n' \
    "${flag}" "${expected_value}" >&2
  exit 1
}

# A lock refusal must not launch Docker.
assert_smoke 30 RAFTINFER_TEST_FLOCK_STATUS=1
[[ ! -s "${fixture_dir}/docker.log" ]]

# A preflight refusal must not launch Docker.
assert_smoke 20 RAFTINFER_TEST_COMPUTE_APPS='1234, training, 1024'
[[ ! -s "${fixture_dir}/docker.log" ]]

# Invalid policy values must be rejected by the outer preflight before Docker.
assert_smoke 23 RAFTINFER_MIN_FREE_MIB=0
[[ ! -s "${fixture_dir}/docker.log" ]]

assert_smoke 0
cmp -s tests/golden/smoke_result.json "${fixture_dir}/stdout"
gpu_docker_log="${fixture_dir}/docker-gpu.log"
assert_docker_flag_value "${fixture_dir}/docker.log" --entrypoint stat
grep -Fx -- '-c' "${fixture_dir}/docker.log"
grep -Fx -- '%g' "${fixture_dir}/docker.log"
grep -Fx -- '/opt/conda' "${fixture_dir}/docker.log"
grep -Fx -- '--gpus' "${gpu_docker_log}"
assert_docker_flag_value "${gpu_docker_log}" --gpus device=0
grep -Fx -- '--user' "${gpu_docker_log}" || {
  echo "Docker smoke must run with the host UID/GID" >&2
  exit 1
}
assert_docker_flag_value "${gpu_docker_log}" --user "${host_uid}:${host_gid}"
grep -Fx -- 'HOME=/tmp/raftinfer-home' "${gpu_docker_log}" || {
  echo "Docker smoke must provide a writable container home" >&2
  exit 1
}
grep -Fx -- 'XDG_CACHE_HOME=/tmp/raftinfer-home/.cache' "${gpu_docker_log}" || {
  echo "Docker smoke must provide a writable container cache" >&2
  exit 1
}
grep -Fx -- '-c' "${gpu_docker_log}" || {
  echo "Docker smoke must preserve the image PATH with bash -c" >&2
  exit 1
}
! grep -Fx -- '-lc' "${gpu_docker_log}"
grep -E -- '.+:/raftinfer-run$' "${gpu_docker_log}" >/dev/null || {
  echo "Docker smoke must mount a fresh run directory" >&2
  exit 1
}
grep -Fx -- '--group-add' "${gpu_docker_log}" || {
  echo "Docker smoke must add the image default group" >&2
  exit 1
}
assert_docker_flag_value "${gpu_docker_log}" --group-add 1001
grep -Fx -- '-e' "${gpu_docker_log}"
grep -Fx -- 'RAFTINFER_MIN_FREE_MIB=2048' "${gpu_docker_log}"
grep -Fx -- 'RAFTINFER_MAX_UTILIZATION_PERCENT=5' "${gpu_docker_log}"
grep -F -- 'cmake -S . -B /raftinfer-run/build' "${gpu_docker_log}"
grep -F -- 'nvcc_path="$(command -v nvcc)"' "${gpu_docker_log}"
grep -F -- '-DCMAKE_CUDA_COMPILER="${nvcc_path}"' "${gpu_docker_log}"
grep -F -- 'scripts/gpu-preflight.sh >/dev/null && /raftinfer-run/build/cpp/raftinfer-smoke' "${gpu_docker_log}"
[[ -z "$(find "${repo_root}/build" -maxdepth 1 -name 'raftinfer-gpu-run.*' -print -quit)" ]]

assert_smoke 0 RAFTINFER_MIN_FREE_MIB=4096 RAFTINFER_MAX_UTILIZATION_PERCENT=3
cmp -s tests/golden/smoke_result.json "${fixture_dir}/stdout"
grep -Fx -- 'RAFTINFER_MIN_FREE_MIB=4096' "${fixture_dir}/docker.log"
grep -Fx -- 'RAFTINFER_MAX_UTILIZATION_PERCENT=3' "${fixture_dir}/docker.log"

# The simulated in-container preflight must stop before the smoke result is emitted.
assert_smoke 20 RAFTINFER_TEST_INNER_COMPUTE_APPS='4321, foreign-job, 1024'
[[ ! -s "${fixture_dir}/stdout" ]]

# Unsafe host identities must be refused before Docker starts.
assert_smoke 32 RAFTINFER_TEST_HOST_UID=0
[[ ! -s "${fixture_dir}/docker.log" ]]
assert_smoke 32 RAFTINFER_TEST_HOST_UID=not-a-number
[[ ! -s "${fixture_dir}/docker.log" ]]
assert_smoke 32 RAFTINFER_TEST_HOST_GID=0
[[ ! -s "${fixture_dir}/docker.log" ]]
assert_smoke 33 RAFTINFER_TEST_CONDA_GID=0
[[ ! -s "${fixture_dir}/docker-gpu.log" ]]
assert_smoke 33 RAFTINFER_TEST_CONDA_GID_STATUS=1
[[ ! -s "${fixture_dir}/docker-gpu.log" ]]

# Docker stdout must compare byte-for-byte with the golden file.
assert_smoke 31 RAFTINFER_TEST_DOCKER_OUTPUT="${golden_json}"
[[ ! -s "${fixture_dir}/stdout" ]]
assert_smoke 31 RAFTINFER_TEST_DOCKER_OUTPUT="${golden_output}"$'\n'
[[ ! -s "${fixture_dir}/stdout" ]]
assert_smoke 31 RAFTINFER_TEST_DOCKER_OUTPUT='{"device_id":0,"element_count":1024,"checksum":0}'
[[ ! -s "${fixture_dir}/stdout" ]]
assert_smoke 77 RAFTINFER_TEST_DOCKER_STATUS=77
[[ ! -s "${fixture_dir}/stdout" ]]

run_sync() {
  PATH="${fixture_dir}:${PATH}" \
    RAFTINFER_TEST_SSH_LOG="${fixture_dir}/ssh.log" \
    RAFTINFER_TEST_RSYNC_LOG="${fixture_dir}/rsync.log" \
    "$@" "${repo_root}/scripts/sync-target.sh"
}

reset_logs
assert_status 40 run_sync env RAFTINFER_TARGET='charles@192.168.124.8;bad'
[[ ! -s "${fixture_dir}/ssh.log" && ! -s "${fixture_dir}/rsync.log" ]]
reset_logs
assert_status 41 run_sync env RAFTINFER_TARGET_DIR=/
[[ ! -s "${fixture_dir}/ssh.log" && ! -s "${fixture_dir}/rsync.log" ]]
reset_logs
assert_status 41 run_sync env RAFTINFER_TARGET_DIR=/home/charles/raftinfer-workspace/../outside
[[ ! -s "${fixture_dir}/ssh.log" && ! -s "${fixture_dir}/rsync.log" ]]
reset_logs
assert_status 0 run_sync env
grep -Fx -- "${repo_root}/" "${fixture_dir}/rsync.log"
! grep -F -- '--delete' "${fixture_dir}/rsync.log"
grep -Fx -- 'mkdir -p -- /home/charles/raftinfer-workspace' "${fixture_dir}/ssh.log"
