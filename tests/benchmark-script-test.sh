#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/brt-benchmark-script.XXXXXX")"
trap 'rm -rf "${fixture_root}"' EXIT

fake_bin="${fixture_root}/bin"
mkdir -p "${fake_bin}"
touch "${fixture_root}/brt.gguf" "${fixture_root}/llama.gguf"
cat >"${fixture_root}/parity.jsonl" <<'EOF'
{"schema_version":1,"name":"one","parity_passed":true,"prompt_token_ids":[10,11],"generated_token_ids":[20]}
{"schema_version":1,"name":"two","parity_passed":true,"prompt_token_ids":[12,13],"generated_token_ids":[21]}
EOF

cat >"${fake_bin}/gpu-preflight" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'preflight\n' >>"${BRT_TEST_PREFLIGHT_LOG}"
EOF

cat >"${fake_bin}/flock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF

cat >"${fake_bin}/llama-server" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
trap 'exit 0' TERM INT
while :; do
  sleep 1
done
EOF

cat >"${fake_bin}/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
url="${*: -1}"
payload=''
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
  if [[ "${arguments[index]}" == "-d" ]]; then
    payload="${arguments[index + 1]}"
  fi
done
case "${url}" in
  */health)
    printf '{"status":"ok"}\n'
    ;;
  */completion)
    printf 'completion\n' >>"${BRT_TEST_COMPLETION_LOG}"
    prompt_n="$(jq '.prompt | length' <<<"${payload}")"
    printf '{"timings":{"prompt_n":%s,"prompt_ms":2,"predicted_n":128,"predicted_ms":4}}\n' \
      "${prompt_n}"
    ;;
  *)
    printf 'unexpected URL: %s\n' "${url}" >&2
    exit 1
    ;;
esac
EOF

cat >"${fake_bin}/brt-cli" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
prompt_tokens=''
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "--prompt-tokens" ]]; then
    prompt_tokens="$2"
    shift 2
  else
    shift
  fi
done
if [[ "${BRT_TEST_SLOW:-0}" == "1" ]]; then
  prefill_median=4000
  generation_median=8000
else
  prefill_median=1000
  generation_median=2000
fi
prefill_tps=$((prompt_tokens * 1000000 / prefill_median))
generation_tps=$((128 * 1000000 / generation_median))
printf '{"schema_version":1,"prompt_tokens":%s,"generated_tokens":128,"warmup_iterations":5,"measured_iterations":20,"peak_allocated_gpu_bytes":18000000000,"prefill":{"min_us":%s,"median_us":%s,"p95_us":%s,"max_us":%s,"tokens_per_second":%s},"generation":{"min_us":%s,"median_us":%s,"p95_us":%s,"max_us":%s,"tokens_per_second":%s}}\n' \
  "${prompt_tokens}" "${prefill_median}" "${prefill_median}" \
  "${prefill_median}" "${prefill_median}" "${prefill_tps}" \
  "${generation_median}" "${generation_median}" "${generation_median}" \
  "${generation_median}" "${generation_tps}"
EOF

cat >"${fake_bin}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'NVIDIA GeForce RTX 5090, 580.65, 2100, 14001\n'
EOF

chmod +x "${fake_bin}"/*

run_benchmark() {
  PATH="${fake_bin}:${PATH}" \
    BRT_MODEL="${fixture_root}/brt.gguf" \
    LLAMA_MODEL="${fixture_root}/llama.gguf" \
    BRT_CLI="${fake_bin}/brt-cli" \
    LLAMA_SERVER_BIN="${fake_bin}/llama-server" \
    GPU_PREFLIGHT="${fake_bin}/gpu-preflight" \
    BRT_GPU_LOCK="${fixture_root}/gpu.lock" \
    BRT_TEST_PREFLIGHT_LOG="${fixture_root}/preflight.log" \
    BRT_TEST_COMPLETION_LOG="${fixture_root}/completion.log" \
    BRT_TEST_SLOW="${BRT_TEST_SLOW:-0}" \
    PARITY_REPORT="${fixture_root}/parity.jsonl" \
    BENCHMARK_OUTPUT="${fixture_root}/benchmark.jsonl" \
    "${repo_root}/scripts/qwen35-benchmark.sh"
}

: >"${fixture_root}/preflight.log"
: >"${fixture_root}/completion.log"
run_benchmark

[[ "$(wc -l <"${fixture_root}/benchmark.jsonl" | tr -d ' ')" -eq 2 ]]
jq -e -s '
  length == 2 and
  ([.[].prompt_tokens] == [128,512]) and
  all(.[];
    .warmup_iterations == 5 and
    .measured_iterations == 20 and
    .performance_floor_passed == true and
    .peak_allocated_gpu_bytes == 18000000000 and
    .peak_memory_status == "measured_by_brt_rmm")
' "${fixture_root}/benchmark.jsonl" >/dev/null
[[ "$(wc -l <"${fixture_root}/preflight.log" | tr -d ' ')" -eq 3 ]]
[[ "$(wc -l <"${fixture_root}/completion.log" | tr -d ' ')" -eq 50 ]]

set +e
BRT_TEST_SLOW=1 run_benchmark \
  >"${fixture_root}/slow-stdout" \
  2>"${fixture_root}/slow-stderr"
slow_status=$?
set -e

[[ "${slow_status}" -eq 42 ]]
grep -F 'performance floor failed' "${fixture_root}/slow-stderr"
jq -e -s 'length == 2 and any(.[]; .performance_floor_passed == false)' \
  "${fixture_root}/benchmark.jsonl" >/dev/null

cat >"${fixture_root}/parity.jsonl" <<'EOF'
{"schema_version":1,"name":"bad","parity_passed":false}
EOF
: >"${fixture_root}/preflight.log"
set +e
run_benchmark >"${fixture_root}/blocked-stdout" 2>"${fixture_root}/blocked-stderr"
blocked_status=$?
set -e

[[ "${blocked_status}" -eq 41 ]]
grep -F 'parity report is not fully passing' "${fixture_root}/blocked-stderr"
[[ ! -s "${fixture_root}/preflight.log" ]]
