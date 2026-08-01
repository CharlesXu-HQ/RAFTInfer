#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/raftinfer-benchmark-script.XXXXXX")"
trap 'rm -rf "${fixture_root}"' EXIT

fake_bin="${fixture_root}/bin"
mkdir -p "${fake_bin}"
printf 'same bf16 model bytes\n' >"${fixture_root}/raftinfer.gguf"
cp "${fixture_root}/raftinfer.gguf" "${fixture_root}/llama.gguf"
model_sha="$(shasum -a 256 "${fixture_root}/raftinfer.gguf" | awk '{print $1}')"
llama_revision="1234567890abcdef1234567890abcdef12345678"

write_provenance() {
  local artifact_sha="${1:-${model_sha}}"
  local revision="${2:-${llama_revision}}"
  local schema_version="${3:-2}"
  jq -nc \
    --arg artifact_sha "${artifact_sha}" \
    --arg revision "${revision}" \
    --argjson schema_version "${schema_version}" \
    '{schema_version:$schema_version,conversion:{outtype:"bf16"},
      llama_cpp:{reference_revision:$revision},
      artifact:{path:"/models/qwen35-bf16.gguf",sha256:$artifact_sha}}' \
    >"${fixture_root}/provenance.json"
}

write_parity() {
  local attention="${1:-online_tiled}"
  local dtype="${2:-bf16}"
  local layout="${3:-head-major}"
  local include_execution="${4:-1}"
  : >"${fixture_root}/parity.jsonl"
  for index in 0 1 2 3; do
    start=$((index * 32))
    end=$((start + 32))
    if [[ "${include_execution}" -eq 1 ]]; then
      jq -nc \
        --arg name "case-${index}" \
        --arg attention "${attention}" \
        --arg dtype "${dtype}" \
        --arg layout "${layout}" \
        --argjson start "${start}" \
        --argjson end "${end}" \
        '{schema_version:2,name:$name,parity_passed:true,
          prompt_token_ids:[10,11],generated_token_ids:[range($start;$end)],
          execution:{attention:$attention,kv_cache_dtype:$dtype,
            kv_cache_layout:$layout,decode_graph_enabled:true,
            decode_graph_captured:false,decode_graph_replayed:false,
            attention_workspace_bytes:0}}' >>"${fixture_root}/parity.jsonl"
    else
      jq -nc \
        --arg name "case-${index}" \
        --argjson start "${start}" \
        --argjson end "${end}" \
        '{schema_version:2,name:$name,parity_passed:true,
          prompt_token_ids:[10,11],generated_token_ids:[range($start;$end)]}' \
        >>"${fixture_root}/parity.jsonl"
    fi
  done
}

write_provenance
write_parity

cat >"${fake_bin}/gpu-preflight" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'preflight\n' >>"${RAFTINFER_TEST_PREFLIGHT_LOG}"
call_count="$(wc -l <"${RAFTINFER_TEST_PREFLIGHT_LOG}" | tr -d ' ')"
if [[ "${call_count}" -eq "${RAFTINFER_TEST_PREFLIGHT_FAIL_CALL:-0}" ]]; then
  exit 22
fi
EOF

cat >"${fake_bin}/flock" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exit 0
EOF

cat >"${fake_bin}/llama-server" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  case "${RAFTINFER_TEST_LLAMA_VERSION:-ok}" in
    ok)
      printf 'version: 1 (1234567)\n'
      ;;
    unknown)
      printf 'llama.cpp build unknown\n'
      ;;
    wrong)
      printf 'llama.cpp build ffffffffffff\n'
      ;;
  esac
  exit 0
fi
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
    printf 'completion\n' >>"${RAFTINFER_TEST_COMPLETION_LOG}"
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

cat >"${fake_bin}/raftinfer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
prompt_tokens=''
decode_tokens=''
kv_cache_dtype=''
kv_cache_layout=''
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --prompt-tokens)
      prompt_tokens="$2"
      shift 2
      ;;
    --decode-tokens)
      decode_tokens="$2"
      shift 2
      ;;
    --kv-cache-dtype)
      kv_cache_dtype="$2"
      shift 2
      ;;
    --kv-cache-layout)
      kv_cache_layout="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
printf 'prompt=%s decode=%s dtype=%s layout=%s\n' \
  "${prompt_tokens}" "${decode_tokens}" "${kv_cache_dtype}" "${kv_cache_layout}" \
  >>"${RAFTINFER_TEST_RAFTINFER_ARG_LOG}"
if [[ "${RAFTINFER_TEST_SLOW:-0}" == "1" ]]; then
  prefill_median=4000
  generation_median=8000
else
  prefill_median=1000
  generation_median=2000
fi
prefill_tps=$((prompt_tokens * 1000000 / prefill_median))
generation_tps=$((128 * 1000000 / generation_median))
printf '{"schema_version":2,"prompt_tokens":%s,"generated_tokens":128,"warmup_iterations":5,"measured_iterations":20,"peak_allocated_gpu_bytes":18000000000,"execution":{"attention":"online_tiled","kv_cache_dtype":"%s","kv_cache_layout":"%s","decode_graph_enabled":true,"decode_graph_captured":true,"decode_graph_replayed":true,"attention_workspace_bytes":0},"prefill":{"min_us":%s,"mean_us":%s,"median_us":%s,"p95_us":%s,"max_us":%s,"coefficient_of_variation":0.01,"tokens_per_second":%s},"generation":{"min_us":%s,"mean_us":%s,"median_us":%s,"p95_us":%s,"max_us":%s,"coefficient_of_variation":0.01,"tokens_per_second":%s}}\n' \
  "${prompt_tokens}" "${kv_cache_dtype}" "${kv_cache_layout}" \
  "${prefill_median}" "${prefill_median}" "${prefill_median}" \
  "${prefill_median}" "${prefill_median}" "${prefill_tps}" \
  "${generation_median}" "${generation_median}" "${generation_median}" \
  "${generation_median}" "${generation_median}" "${generation_tps}"
EOF

cat >"${fake_bin}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'NVIDIA GeForce RTX 5090, 580.65, 2100, 14001\n'
EOF

chmod +x "${fake_bin}"/*

run_benchmark() {
  PATH="${fake_bin}:${PATH}" \
    RAFTINFER_MODEL="${fixture_root}/raftinfer.gguf" \
    LLAMA_MODEL="${fixture_root}/llama.gguf" \
    RAFTINFER_CLI="${fake_bin}/raftinfer" \
    LLAMA_SERVER_BIN="${fake_bin}/llama-server" \
    GPU_PREFLIGHT="${fake_bin}/gpu-preflight" \
    RAFTINFER_GPU_LOCK="${fixture_root}/gpu.lock" \
    RAFTINFER_TEST_PREFLIGHT_LOG="${fixture_root}/preflight.log" \
    RAFTINFER_TEST_PREFLIGHT_FAIL_CALL="${RAFTINFER_TEST_PREFLIGHT_FAIL_CALL:-0}" \
    RAFTINFER_TEST_COMPLETION_LOG="${fixture_root}/completion.log" \
    RAFTINFER_TEST_RAFTINFER_ARG_LOG="${fixture_root}/raftinfer-args.log" \
    RAFTINFER_TEST_SLOW="${RAFTINFER_TEST_SLOW:-0}" \
    RAFTINFER_TEST_LLAMA_VERSION="${RAFTINFER_TEST_LLAMA_VERSION:-ok}" \
    RAFTINFER_KV_CACHE_DTYPE="bf16" \
    RAFTINFER_KV_CACHE_LAYOUT="head-major" \
    PARITY_REPORT="${fixture_root}/parity.jsonl" \
    PROVENANCE_JSON="${fixture_root}/provenance.json" \
    BENCHMARK_OUTPUT="${fixture_root}/benchmark.jsonl" \
    "${repo_root}/scripts/qwen35-benchmark.sh"
}

: >"${fixture_root}/preflight.log"
: >"${fixture_root}/completion.log"
: >"${fixture_root}/raftinfer-args.log"
RAFTINFER_TEST_PREFLIGHT_FAIL_CALL=2 \
  RAFTINFER_PREFLIGHT_RETRY_SECONDS=0 \
  run_benchmark

[[ "$(wc -l <"${fixture_root}/benchmark.jsonl" | tr -d ' ')" -eq 3 ]]
jq -e -s --arg model_sha "${model_sha}" '
  length == 3 and
  ([.[].arm] == ["pp128","pp512","tg128_pp512"]) and
  ([.[].prompt_tokens] == [128,512,512]) and
  all(.[];
    .warmup_iterations == 5 and
    .measured_iterations == 20 and
    .provenance.weight_format == "bf16" and
    .provenance.llama_cpp_revision == "1234567890abcdef1234567890abcdef12345678" and
    .provenance.artifact_sha256 == $model_sha and
    .schema_version == 2 and
    .provenance.raftinfer_model_sha256 == $model_sha and
    .provenance.llama_model_sha256 == $model_sha and
    (.provenance.llama_server_sha256 | type == "string" and length == 64) and
    .provenance.llama_server_version == "version: 1 (1234567)" and
    .provenance.llama_cpp_revision_verified == true and
    .parity.records == 4 and
    .parity.generated_tokens_per_record == 32 and
    .parity.exact_matches == 128 and
    .parity.passed == true and
    .parity.execution.attention == "online_tiled" and
    .parity.execution.kv_cache_dtype == "bf16" and
    .parity.execution.kv_cache_layout == "head-major" and
    .execution.attention == "online_tiled" and
    .execution.kv_cache_dtype == "bf16" and
    .execution.kv_cache_layout == "head-major" and
    (.raftinfer.prefill.coefficient_of_variation | type == "number") and
    (.raftinfer.generation.coefficient_of_variation | type == "number") and
    (.llama_cpp.prefill.coefficient_of_variation | type == "number") and
    (.llama_cpp.generation.coefficient_of_variation | type == "number") and
    .performance_floor_passed == true and
    .peak_allocated_gpu_bytes == 18000000000 and
    .peak_memory_status == "measured_by_raftinfer_rmm")
' "${fixture_root}/benchmark.jsonl" >/dev/null
"${repo_root}/scripts/qwen35-bf16-gate.sh" "${fixture_root}/benchmark.jsonl"
[[ "$(wc -l <"${fixture_root}/preflight.log" | tr -d ' ')" -eq 5 ]]
[[ "$(wc -l <"${fixture_root}/completion.log" | tr -d ' ')" -eq 75 ]]
[[ "$(wc -l <"${fixture_root}/raftinfer-args.log" | tr -d ' ')" -eq 3 ]]
grep -F 'dtype=bf16 layout=head-major' "${fixture_root}/raftinfer-args.log"

# A v1 provenance record must not be accepted as an automation input.
write_provenance "${model_sha}" "${llama_revision}" 1
set +e
run_benchmark >"${fixture_root}/legacy-schema-stdout" \
  2>"${fixture_root}/legacy-schema-stderr"
legacy_schema_status=$?
set -e
[[ "${legacy_schema_status}" -eq 43 ]]
grep -F 'provenance JSON is not a pinned BF16 artifact' \
  "${fixture_root}/legacy-schema-stderr"
write_provenance

# A renamed field is required: the v1 provenance spelling must be rejected.
jq -c 'del(.provenance.raftinfer_model_sha256) |
  .provenance.brt_model_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' \
  "${fixture_root}/benchmark.jsonl" >"${fixture_root}/legacy-field.jsonl"
set +e
"${repo_root}/scripts/qwen35-bf16-gate.sh" "${fixture_root}/legacy-field.jsonl" \
  >"${fixture_root}/legacy-field-stdout" \
  2>"${fixture_root}/legacy-field-stderr"
legacy_field_status=$?
set -e
[[ "${legacy_field_status}" -ne 0 ]]
grep -F 'measured model SHA256' "${fixture_root}/legacy-field-stderr"

# A legacy top-level benchmark namespace is rejected even with valid v2
# provenance and all other RAFTInfer fields intact.
jq -c '.brt = .raftinfer | del(.raftinfer)' \
  "${fixture_root}/benchmark.jsonl" >"${fixture_root}/legacy-namespace.jsonl"
set +e
"${repo_root}/scripts/qwen35-bf16-gate.sh" "${fixture_root}/legacy-namespace.jsonl" \
  >"${fixture_root}/legacy-namespace-stdout" \
  2>"${fixture_root}/legacy-namespace-stderr"
legacy_namespace_status=$?
set -e
[[ "${legacy_namespace_status}" -ne 0 ]]
grep -F 'resolved attention must be online_tiled' \
  "${fixture_root}/legacy-namespace-stderr"

printf 'different bytes\n' >"${fixture_root}/llama.gguf"
set +e
run_benchmark >"${fixture_root}/model-hash-stdout" \
  2>"${fixture_root}/model-hash-stderr"
model_hash_status=$?
set -e
[[ "${model_hash_status}" -eq 43 ]]
grep -F 'model SHA256 does not match provenance artifact' \
  "${fixture_root}/model-hash-stderr"
cp "${fixture_root}/raftinfer.gguf" "${fixture_root}/llama.gguf"

write_parity online_tiled bf16 token-major
set +e
run_benchmark >"${fixture_root}/parity-policy-stdout" \
  2>"${fixture_root}/parity-policy-stderr"
parity_policy_status=$?
set -e
[[ "${parity_policy_status}" -eq 41 ]]
grep -F 'parity execution does not match benchmark policy' \
  "${fixture_root}/parity-policy-stderr"
write_parity online_tiled bf16 head-major 0
set +e
run_benchmark >"${fixture_root}/parity-missing-stdout" \
  2>"${fixture_root}/parity-missing-stderr"
parity_missing_status=$?
set -e
[[ "${parity_missing_status}" -eq 41 ]]
grep -F 'parity execution diagnostics are missing' \
  "${fixture_root}/parity-missing-stderr"
write_parity

set +e
RAFTINFER_TEST_LLAMA_VERSION=unknown run_benchmark \
  >"${fixture_root}/llama-version-unknown-stdout" \
  2>"${fixture_root}/llama-version-unknown-stderr"
llama_unknown_status=$?
set -e
[[ "${llama_unknown_status}" -eq 43 ]]
grep -F 'llama-server --version did not verify pinned revision' \
  "${fixture_root}/llama-version-unknown-stderr"

set +e
RAFTINFER_TEST_LLAMA_VERSION=wrong run_benchmark \
  >"${fixture_root}/llama-version-wrong-stdout" \
  2>"${fixture_root}/llama-version-wrong-stderr"
llama_wrong_status=$?
set -e
[[ "${llama_wrong_status}" -eq 43 ]]
grep -F 'llama-server --version did not verify pinned revision' \
  "${fixture_root}/llama-version-wrong-stderr"

set +e
RAFTINFER_TEST_SLOW=1 run_benchmark \
  >"${fixture_root}/slow-stdout" \
  2>"${fixture_root}/slow-stderr"
slow_status=$?
set -e

[[ "${slow_status}" -eq 42 ]]
grep -F 'performance floor failed' "${fixture_root}/slow-stderr"
jq -e -s 'length == 3 and any(.[]; .performance_floor_passed == false)' \
  "${fixture_root}/benchmark.jsonl" >/dev/null

cat >"${fixture_root}/parity.jsonl" <<'EOF'
{"schema_version":2,"name":"bad","parity_passed":false}
EOF
: >"${fixture_root}/preflight.log"
set +e
run_benchmark >"${fixture_root}/blocked-stdout" 2>"${fixture_root}/blocked-stderr"
blocked_status=$?
set -e

[[ "${blocked_status}" -eq 41 ]]
grep -F 'parity report is not fully passing' "${fixture_root}/blocked-stderr"
[[ ! -s "${fixture_root}/preflight.log" ]]
