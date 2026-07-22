#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_dir="$(mktemp -d)"
trap 'rm -rf "${fixture_dir}"' EXIT

cat >"${fixture_dir}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *query-compute-apps*) printf '%s' "${BRT_TEST_COMPUTE_APPS:-}" ;;
  *query-gpu*) printf '%s\n' "${BRT_TEST_GPU_ROW:-8192, 0, 42}" ;;
  *) exit 99 ;;
esac
EOF
chmod +x "${fixture_dir}/nvidia-smi"

assert_preflight() {
  local expected="$1"
  local compute_apps="$2"
  local gpu_row="$3"
  set +e
  PATH="${fixture_dir}:${PATH}" \
    BRT_TEST_COMPUTE_APPS="${compute_apps}" \
    BRT_TEST_GPU_ROW="${gpu_row}" \
    "${repo_root}/scripts/gpu-preflight.sh" >"${fixture_dir}/stdout" 2>"${fixture_dir}/stderr"
  local actual=$?
  set -e
  [[ "${actual}" -eq "${expected}" ]] || {
    printf 'expected exit %s, got %s\n' "${expected}" "${actual}" >&2
    cat "${fixture_dir}/stderr" >&2
    exit 1
  }
}

assert_preflight 0 '' '8192, 0, 42'
grep -Fx 'gpu_preflight=ok free_mib=8192 utilization=0 temperature=42' "${fixture_dir}/stdout"

assert_preflight 20 '1234, training, 1024' '8192, 0, 42'
grep -Fqx 'BRT GPU preflight refused: active compute applications detected' "${fixture_dir}/stderr"

assert_preflight 21 '' '1024, 0, 42'
grep -Fqx 'BRT GPU preflight refused: free=1024MiB required=2048MiB' "${fixture_dir}/stderr"

assert_preflight 22 '' '8192, 6, 42'
grep -Fqx 'BRT GPU preflight refused: utilization=6%' "${fixture_dir}/stderr"
