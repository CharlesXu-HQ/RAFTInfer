#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/brt-parity-script.XXXXXX")"
trap 'rm -rf "${fixture_root}"' EXIT

fake_bin="${fixture_root}/bin"
mkdir -p "${fake_bin}"
touch "${fixture_root}/brt.gguf" "${fixture_root}/llama.gguf"

cat >"${fake_bin}/gpu-preflight" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'preflight\n' >>"${BRT_TEST_PREFLIGHT_LOG}"
call_count="$(wc -l <"${BRT_TEST_PREFLIGHT_LOG}" | tr -d ' ')"
if [[ "${call_count}" -eq "${BRT_TEST_PREFLIGHT_FAIL_CALL:-0}" ]]; then
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
trap 'exit 0' TERM INT
while :; do
  sleep 1
done
EOF

cat >"${fake_bin}/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
url="${*: -1}"
case "${url}" in
  */health)
    printf '{"status":"ok"}\n'
    ;;
  */apply-template)
    printf '{"prompt":"rendered prompt"}\n'
    ;;
  */tokenize)
    printf '{"tokens":[10,11]}\n'
    ;;
  */completion)
    printf '{"tokens":[20,21]}\n'
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
kv_cache_dtype=''
kv_cache_layout=''
while [[ "$#" -gt 0 ]]; do
  case "$1" in
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
printf 'dtype=%s layout=%s\n' "${kv_cache_dtype}" "${kv_cache_layout}" \
  >>"${BRT_TEST_BRT_ARG_LOG}"
resolved_dtype="${BRT_TEST_RESOLVED_DTYPE:-${kv_cache_dtype:-f32}}"
resolved_layout="${BRT_TEST_RESOLVED_LAYOUT:-${kv_cache_layout:-token-major}}"
resolved_attention="${BRT_TEST_RESOLVED_ATTENTION:-online_tiled}"
execution_json="$(jq -nc \
  --arg attention "${resolved_attention}" \
  --arg dtype "${resolved_dtype}" \
  --arg layout "${resolved_layout}" \
  '{attention:$attention,kv_cache_dtype:$dtype,kv_cache_layout:$layout,
    decode_graph_enabled:true,decode_graph_captured:false,
    decode_graph_replayed:false,attention_workspace_bytes:0}')"
case "${BRT_TEST_MISMATCH:-0}" in
  missing_execution)
    printf '{"schema_version":1,"prompt_token_ids":[10,11],"generated_token_ids":[20,21],"text":"x"}\n'
    ;;
  1)
    jq -nc --argjson execution "${execution_json}" \
      '{schema_version:1,prompt_token_ids:[10,11],generated_token_ids:[20,22],
        text:"x",execution:$execution}'
    ;;
  short)
    jq -nc --argjson execution "${execution_json}" \
      '{schema_version:1,prompt_token_ids:[10,11],generated_token_ids:[20],
        text:"x",execution:$execution}'
    ;;
  *)
    jq -nc --argjson execution "${execution_json}" \
      '{schema_version:1,prompt_token_ids:[10,11],generated_token_ids:[20,21],
        text:"x",execution:$execution}'
    ;;
esac
EOF

chmod +x "${fake_bin}"/*

run_parity() {
  PATH="${fake_bin}:${PATH}" \
    BRT_MODEL="${fixture_root}/brt.gguf" \
    LLAMA_MODEL="${fixture_root}/llama.gguf" \
    BRT_CLI="${fake_bin}/brt-cli" \
    LLAMA_SERVER_BIN="${fake_bin}/llama-server" \
    GPU_PREFLIGHT="${fake_bin}/gpu-preflight" \
    BRT_GPU_LOCK="${fixture_root}/gpu.lock" \
    BRT_TEST_PREFLIGHT_LOG="${fixture_root}/preflight.log" \
    BRT_TEST_PREFLIGHT_FAIL_CALL="${BRT_TEST_PREFLIGHT_FAIL_CALL:-0}" \
    BRT_TEST_BRT_ARG_LOG="${fixture_root}/brt-args.log" \
    BRT_TEST_MISMATCH="${BRT_TEST_MISMATCH:-0}" \
    BRT_TEST_RESOLVED_DTYPE="${BRT_TEST_RESOLVED_DTYPE:-}" \
    BRT_TEST_RESOLVED_LAYOUT="${BRT_TEST_RESOLVED_LAYOUT:-}" \
    BRT_TEST_RESOLVED_ATTENTION="${BRT_TEST_RESOLVED_ATTENTION:-}" \
    PARITY_OUTPUT="${fixture_root}/parity.jsonl" \
    "${repo_root}/scripts/qwen35-parity.sh"
}

: >"${fixture_root}/preflight.log"
: >"${fixture_root}/brt-args.log"
run_parity

[[ "$(wc -l <"${fixture_root}/parity.jsonl" | tr -d ' ')" -eq 4 ]]
jq -e -s 'length == 4 and all(.[]; .parity_passed == true)' \
  "${fixture_root}/parity.jsonl" >/dev/null
jq -e -s 'all(.[]; .execution.attention == "online_tiled" and
  .execution.kv_cache_dtype == "f32" and
  .execution.kv_cache_layout == "token-major")' \
  "${fixture_root}/parity.jsonl" >/dev/null
[[ "$(wc -l <"${fixture_root}/preflight.log" | tr -d ' ')" -eq 5 ]]
[[ "$(wc -l <"${fixture_root}/brt-args.log" | tr -d ' ')" -eq 4 ]]
grep -Fx 'dtype= layout=' "${fixture_root}/brt-args.log"

: >"${fixture_root}/preflight.log"
: >"${fixture_root}/brt-args.log"
BRT_KV_CACHE_DTYPE=bf16 \
  BRT_KV_CACHE_LAYOUT=head-major \
  run_parity
[[ "$(wc -l <"${fixture_root}/brt-args.log" | tr -d ' ')" -eq 4 ]]
grep -Fx 'dtype=bf16 layout=head-major' "${fixture_root}/brt-args.log"
jq -e -s 'all(.[]; .execution.attention == "online_tiled" and
  .execution.kv_cache_dtype == "bf16" and
  .execution.kv_cache_layout == "head-major")' \
  "${fixture_root}/parity.jsonl" >/dev/null

set +e
BRT_KV_CACHE_DTYPE=bf16 \
  BRT_KV_CACHE_LAYOUT=head-major \
  BRT_TEST_RESOLVED_LAYOUT=token-major \
  run_parity >"${fixture_root}/resolved-mismatch-stdout" \
  2>"${fixture_root}/resolved-mismatch-stderr"
resolved_mismatch_status=$?
set -e
[[ "${resolved_mismatch_status}" -eq 44 ]]
grep -F 'resolved execution policy does not match requested policy' \
  "${fixture_root}/resolved-mismatch-stderr"

set +e
BRT_KV_CACHE_DTYPE=bf16 \
  BRT_KV_CACHE_LAYOUT=head-major \
  BRT_TEST_MISMATCH=missing_execution \
  run_parity >"${fixture_root}/missing-execution-stdout" \
  2>"${fixture_root}/missing-execution-stderr"
missing_execution_status=$?
set -e
[[ "${missing_execution_status}" -eq 44 ]]
grep -F 'resolved execution diagnostics are missing' \
  "${fixture_root}/missing-execution-stderr"

set +e
BRT_KV_CACHE_DTYPE=bad run_parity \
  >"${fixture_root}/bad-kv-stdout" \
  2>"${fixture_root}/bad-kv-stderr"
bad_kv_status=$?
set -e
[[ "${bad_kv_status}" -eq 2 ]]
grep -F 'BRT_KV_CACHE_DTYPE must be f32 or bf16' \
  "${fixture_root}/bad-kv-stderr"

: >"${fixture_root}/preflight.log"
BRT_TEST_PREFLIGHT_FAIL_CALL=1 \
  BRT_PREFLIGHT_RETRY_SECONDS=0 \
  run_parity
[[ "$(wc -l <"${fixture_root}/preflight.log" | tr -d ' ')" -eq 6 ]]

: >"${fixture_root}/preflight.log"
BRT_TEST_PREFLIGHT_FAIL_CALL=2 \
  BRT_PREFLIGHT_RETRY_SECONDS=0 \
  run_parity
[[ "$(wc -l <"${fixture_root}/preflight.log" | tr -d ' ')" -eq 6 ]]

: >"${fixture_root}/preflight.log"
set +e
BRT_TEST_MISMATCH=1 run_parity \
  >"${fixture_root}/mismatch-stdout" \
  2>"${fixture_root}/mismatch-stderr"
mismatch_status=$?
set -e

[[ "${mismatch_status}" -eq 40 ]]
grep -F 'generated token mismatch at index 1: expected=21 actual=22' \
  "${fixture_root}/mismatch-stderr"
jq -e -s 'length == 1 and .[0].parity_passed == false and
  .[0].mismatch.kind == "generated" and .[0].mismatch.index == 1 and
  .[0].mismatch.expected == 21 and .[0].mismatch.actual == 22 and
  .[0].execution.attention == "online_tiled"' \
  "${fixture_root}/parity.jsonl" >/dev/null

set +e
BRT_TEST_MISMATCH=short run_parity \
  >"${fixture_root}/short-stdout" \
  2>"${fixture_root}/short-stderr"
short_status=$?
set -e

[[ "${short_status}" -eq 40 ]]
grep -F 'generated token mismatch at index 1: expected=21 actual=missing' \
  "${fixture_root}/short-stderr"
jq -e -s 'length == 1 and .[0].mismatch.expected == 21 and
  .[0].mismatch.actual == null' "${fixture_root}/parity.jsonl" >/dev/null
