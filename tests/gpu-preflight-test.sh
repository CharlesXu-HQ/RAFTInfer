#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_dir="$(mktemp -d)"
golden_json='{"device_id":0,"element_count":1024,"checksum":523776}'
golden_output="${golden_json}"$'\n'
trap 'rm -rf "${fixture_dir}"' EXIT

cat >"${fixture_dir}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${BRT_TEST_NVIDIA_LOG}"
case "$*" in
  *query-compute-apps*)
    if [[ "${BRT_TEST_NVIDIA_MODE:-}" == fail_compute ]]; then
      exit 70
    fi
    printf '%s' "${BRT_TEST_COMPUTE_APPS:-}"
    ;;
  *query-gpu*)
    if [[ "${BRT_TEST_NVIDIA_MODE:-}" == fail_gpu ]]; then
      exit 71
    fi
    printf '%s' "${BRT_TEST_GPU_ROW}"
    ;;
  *) exit 99 ;;
esac
EOF

cat >"${fixture_dir}/flock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${BRT_TEST_FLOCK_LOG}"
exit "${BRT_TEST_FLOCK_STATUS:-0}"
EOF

cat >"${fixture_dir}/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >>"${BRT_TEST_DOCKER_LOG}"
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  if [[ "${arguments[index]}" == -e ]]; then
    index=$((index + 1))
    case "${arguments[index]}" in
      BRT_MIN_FREE_MIB=*|BRT_MAX_UTILIZATION_PERCENT=*)
        export "${arguments[index]}"
        ;;
    esac
  fi
done
for argument in "${arguments[@]}"; do
  if [[ "${argument}" == *'scripts/gpu-preflight.sh >/dev/null && ./build/gpu/cpp/brt-smoke'* ]]; then
    BRT_TEST_COMPUTE_APPS="${BRT_TEST_INNER_COMPUTE_APPS:-${BRT_TEST_COMPUTE_APPS:-}}" \
      "${BRT_TEST_REPO_ROOT}/scripts/gpu-preflight.sh" >/dev/null
    break
  fi
done
if [[ "${BRT_TEST_DOCKER_STATUS:-0}" -ne 0 ]]; then
  exit "${BRT_TEST_DOCKER_STATUS}"
fi
printf '%s' "${BRT_TEST_DOCKER_OUTPUT}"
EOF

cat >"${fixture_dir}/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >>"${BRT_TEST_SSH_LOG}"
EOF

cat >"${fixture_dir}/rsync" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >>"${BRT_TEST_RSYNC_LOG}"
EOF

chmod +x "${fixture_dir}"/{nvidia-smi,flock,docker,ssh,rsync}

reset_logs() {
  : >"${fixture_dir}/nvidia.log"
  : >"${fixture_dir}/flock.log"
  : >"${fixture_dir}/docker.log"
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
    BRT_TEST_NVIDIA_LOG="${fixture_dir}/nvidia.log" \
    BRT_TEST_COMPUTE_APPS="${compute_apps}" \
    BRT_TEST_GPU_ROW="${gpu_row}" \
    BRT_TEST_NVIDIA_MODE="${nvidia_mode}" \
    BRT_MIN_FREE_MIB="${min_free}" \
    BRT_MAX_UTILIZATION_PERCENT="${max_utilization}" \
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
grep -Fqx 'BRT GPU preflight refused: active compute applications detected' "${fixture_dir}/stderr"
assert_preflight 21 '' '1024, 0, 42' '' ''
grep -Fqx 'BRT GPU preflight refused: free=1024MiB required=2048MiB' "${fixture_dir}/stderr"
assert_preflight 22 '' '8192, 6, 42' '' ''
grep -Fqx 'BRT GPU preflight refused: utilization=6%' "${fixture_dir}/stderr"

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
    BRT_TEST_NVIDIA_LOG="${fixture_dir}/nvidia.log" \
    BRT_TEST_FLOCK_LOG="${fixture_dir}/flock.log" \
    BRT_TEST_DOCKER_LOG="${fixture_dir}/docker.log" \
    BRT_TEST_REPO_ROOT="${repo_root}" \
    BRT_TEST_COMPUTE_APPS="${BRT_TEST_COMPUTE_APPS:-}" \
    BRT_TEST_INNER_COMPUTE_APPS="${BRT_TEST_INNER_COMPUTE_APPS:-}" \
    BRT_TEST_GPU_ROW="${BRT_TEST_GPU_ROW:-8192, 0, 42}" \
    BRT_TEST_FLOCK_STATUS="${BRT_TEST_FLOCK_STATUS:-0}" \
    BRT_TEST_DOCKER_OUTPUT="${BRT_TEST_DOCKER_OUTPUT:-${golden_output}}" \
    BRT_TEST_DOCKER_STATUS="${BRT_TEST_DOCKER_STATUS:-0}" \
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

# A lock refusal must not launch Docker.
assert_smoke 30 BRT_TEST_FLOCK_STATUS=1
[[ ! -s "${fixture_dir}/docker.log" ]]

# A preflight refusal must not launch Docker.
assert_smoke 20 BRT_TEST_COMPUTE_APPS='1234, training, 1024'
[[ ! -s "${fixture_dir}/docker.log" ]]

# Invalid policy values must be rejected by the outer preflight before Docker.
assert_smoke 23 BRT_MIN_FREE_MIB=0
[[ ! -s "${fixture_dir}/docker.log" ]]

assert_smoke 0
cmp -s tests/golden/smoke_result.json "${fixture_dir}/stdout"
grep -Fx -- '--gpus' "${fixture_dir}/docker.log"
grep -Fx -- 'device=0' "${fixture_dir}/docker.log"
grep -Fx -- '-e' "${fixture_dir}/docker.log"
grep -Fx -- 'BRT_MIN_FREE_MIB=2048' "${fixture_dir}/docker.log"
grep -Fx -- 'BRT_MAX_UTILIZATION_PERCENT=5' "${fixture_dir}/docker.log"
grep -F -- 'scripts/gpu-preflight.sh >/dev/null && ./build/gpu/cpp/brt-smoke' "${fixture_dir}/docker.log"

assert_smoke 0 BRT_MIN_FREE_MIB=4096 BRT_MAX_UTILIZATION_PERCENT=3
cmp -s tests/golden/smoke_result.json "${fixture_dir}/stdout"
grep -Fx -- 'BRT_MIN_FREE_MIB=4096' "${fixture_dir}/docker.log"
grep -Fx -- 'BRT_MAX_UTILIZATION_PERCENT=3' "${fixture_dir}/docker.log"

# The simulated in-container preflight must stop before the smoke result is emitted.
assert_smoke 20 BRT_TEST_INNER_COMPUTE_APPS='4321, foreign-job, 1024'
[[ ! -s "${fixture_dir}/stdout" ]]

# Docker stdout must compare byte-for-byte with the golden file.
assert_smoke 31 BRT_TEST_DOCKER_OUTPUT="${golden_json}"
[[ ! -s "${fixture_dir}/stdout" ]]
assert_smoke 31 BRT_TEST_DOCKER_OUTPUT="${golden_output}"$'\n'
[[ ! -s "${fixture_dir}/stdout" ]]
assert_smoke 31 BRT_TEST_DOCKER_OUTPUT='{"device_id":0,"element_count":1024,"checksum":0}'
[[ ! -s "${fixture_dir}/stdout" ]]
assert_smoke 77 BRT_TEST_DOCKER_STATUS=77
[[ ! -s "${fixture_dir}/stdout" ]]

run_sync() {
  PATH="${fixture_dir}:${PATH}" \
    BRT_TEST_SSH_LOG="${fixture_dir}/ssh.log" \
    BRT_TEST_RSYNC_LOG="${fixture_dir}/rsync.log" \
    "$@" "${repo_root}/scripts/sync-target.sh"
}

reset_logs
assert_status 40 run_sync env BRT_TARGET='charles@192.168.124.8;bad'
[[ ! -s "${fixture_dir}/ssh.log" && ! -s "${fixture_dir}/rsync.log" ]]
reset_logs
assert_status 41 run_sync env BRT_TARGET_DIR=/
[[ ! -s "${fixture_dir}/ssh.log" && ! -s "${fixture_dir}/rsync.log" ]]
reset_logs
assert_status 41 run_sync env BRT_TARGET_DIR=/home/charles/brt-workspace/../outside
[[ ! -s "${fixture_dir}/ssh.log" && ! -s "${fixture_dir}/rsync.log" ]]
reset_logs
assert_status 0 run_sync env
grep -Fx -- "${repo_root}/" "${fixture_dir}/rsync.log"
! grep -F -- '--delete' "${fixture_dir}/rsync.log"
grep -Fx -- 'mkdir -p -- /home/charles/brt-workspace' "${fixture_dir}/ssh.log"
