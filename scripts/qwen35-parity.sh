#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
corpus="${QWEN35_GENERATION_CORPUS:-${repo_root}/tests/parity/qwen35-generation-corpus.jsonl}"
output="${PARITY_OUTPUT:-${repo_root}/build/evidence/qwen35-parity.jsonl}"
raftinfer_model="${RAFTINFER_MODEL:-}"
llama_model="${LLAMA_MODEL:-${raftinfer_model}}"
raftinfer_cli="${RAFTINFER_CLI:-${repo_root}/target/release/raftinfer}"
llama_server="${LLAMA_SERVER_BIN:-}"
gpu_preflight="${GPU_PREFLIGHT:-${repo_root}/scripts/gpu-preflight.sh}"
gpu_lock="${RAFTINFER_GPU_LOCK:-/tmp/raftinfer-qwen35-gpu.lock}"
curl_bin="${CURL_BIN:-curl}"
jq_bin="${JQ_BIN:-jq}"
context_tokens="${RAFTINFER_CONTEXT_TOKENS:-4096}"
max_new_tokens="${RAFTINFER_MAX_NEW_TOKENS:-32}"
kv_cache_dtype="${RAFTINFER_KV_CACHE_DTYPE:-}"
kv_cache_layout="${RAFTINFER_KV_CACHE_LAYOUT:-}"
llama_port="${LLAMA_SERVER_PORT:-18080}"
llama_url="http://127.0.0.1:${llama_port}"
preflight_retries="${RAFTINFER_PREFLIGHT_RETRIES:-30}"
preflight_retry_seconds="${RAFTINFER_PREFLIGHT_RETRY_SECONDS:-1}"

fail_usage() {
  printf 'qwen35-parity: %s\n' "$1" >&2
  exit 2
}

[[ -n "${raftinfer_model}" ]] || fail_usage "RAFTINFER_MODEL is required"
[[ -f "${raftinfer_model}" ]] || fail_usage "RAFTINFER_MODEL does not exist: ${raftinfer_model}"
[[ -f "${llama_model}" ]] || fail_usage "LLAMA_MODEL does not exist: ${llama_model}"
[[ -x "${raftinfer_cli}" ]] || fail_usage "RAFTINFER_CLI is not executable: ${raftinfer_cli}"
[[ -n "${llama_server}" && -x "${llama_server}" ]] ||
  fail_usage "LLAMA_SERVER_BIN must name an executable"
[[ -x "${gpu_preflight}" ]] ||
  fail_usage "GPU_PREFLIGHT is not executable: ${gpu_preflight}"
[[ -f "${corpus}" ]] || fail_usage "generation corpus does not exist: ${corpus}"
command -v "${curl_bin}" >/dev/null ||
  fail_usage "curl command not found: ${curl_bin}"
command -v "${jq_bin}" >/dev/null ||
  fail_usage "jq command not found: ${jq_bin}"
[[ "${context_tokens}" =~ ^[1-9][0-9]*$ ]] ||
  fail_usage "RAFTINFER_CONTEXT_TOKENS must be a positive integer"
[[ "${max_new_tokens}" =~ ^[1-9][0-9]*$ ]] ||
  fail_usage "RAFTINFER_MAX_NEW_TOKENS must be a positive integer"
[[ "${llama_port}" =~ ^[1-9][0-9]*$ ]] ||
  fail_usage "LLAMA_SERVER_PORT must be a positive integer"
[[ "${preflight_retries}" =~ ^[1-9][0-9]*$ ]] ||
  fail_usage "RAFTINFER_PREFLIGHT_RETRIES must be a positive integer"
[[ "${preflight_retry_seconds}" =~ ^[0-9]+$ ]] ||
  fail_usage "RAFTINFER_PREFLIGHT_RETRY_SECONDS must be a non-negative integer"
if [[ -n "${kv_cache_dtype}" &&
      "${kv_cache_dtype}" != "f32" &&
      "${kv_cache_dtype}" != "bf16" ]]; then
  fail_usage "RAFTINFER_KV_CACHE_DTYPE must be f32 or bf16"
fi
if [[ -n "${kv_cache_layout}" &&
      "${kv_cache_layout}" != "token-major" &&
      "${kv_cache_layout}" != "head-major" ]]; then
  fail_usage "RAFTINFER_KV_CACHE_LAYOUT must be token-major or head-major"
fi

raftinfer_policy_args=()
if [[ -n "${kv_cache_dtype}" ]]; then
  raftinfer_policy_args+=(--kv-cache-dtype "${kv_cache_dtype}")
fi
if [[ -n "${kv_cache_layout}" ]]; then
  raftinfer_policy_args+=(--kv-cache-layout "${kv_cache_layout}")
fi

wait_for_gpu_preflight() {
  local attempt=1
  local preflight_output
  local preflight_status
  while :; do
    if preflight_output="$("${gpu_preflight}" 2>&1)"; then
      return 0
    fi
    preflight_status=$?
    if [[ "${attempt}" -ge "${preflight_retries}" ]]; then
      printf '%s\n' "${preflight_output}" >&2
      return "${preflight_status}"
    fi
    sleep "${preflight_retry_seconds}"
    attempt=$((attempt + 1))
  done
}

mkdir -p "$(dirname "${output}")"
reference_file="$(mktemp "${TMPDIR:-/tmp}/raftinfer-qwen35-reference.XXXXXX")"
server_log="$(mktemp "${TMPDIR:-/tmp}/raftinfer-qwen35-llama-server.XXXXXX")"
server_pid=''

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -f "${reference_file}" "${server_log}"
}
trap cleanup EXIT

exec 9>"${gpu_lock}"
if ! flock -n 9; then
  printf 'qwen35-parity: GPU lock is held: %s\n' "${gpu_lock}" >&2
  exit 30
fi

wait_for_gpu_preflight
"${llama_server}" \
  --model "${llama_model}" \
  --host 127.0.0.1 \
  --port "${llama_port}" \
  --ctx-size "${context_tokens}" \
  --parallel 1 \
  --gpu-layers 999 \
  --no-warmup \
  >"${server_log}" 2>&1 &
server_pid=$!

server_ready=0
for _ in $(seq 1 60); do
  if "${curl_bin}" -fsS "${llama_url}/health" >/dev/null 2>&1; then
    server_ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "${server_ready}" -ne 1 ]]; then
  printf 'qwen35-parity: llama-server did not become ready\n' >&2
  tail -100 "${server_log}" >&2 || true
  exit 31
fi

while IFS= read -r corpus_line || [[ -n "${corpus_line}" ]]; do
  [[ -n "${corpus_line}" ]] || continue
  if ! "${jq_bin}" -e '
    type == "object" and
    (.name | type == "string" and length > 0) and
    (.prompt | type == "string" and length > 0)
  ' <<<"${corpus_line}" >/dev/null; then
    printf 'qwen35-parity: malformed corpus record: %s\n' "${corpus_line}" >&2
    exit 32
  fi
  name="$("${jq_bin}" -r '.name' <<<"${corpus_line}")"
  prompt="$("${jq_bin}" -r '.prompt' <<<"${corpus_line}")"

  template_request="$("${jq_bin}" -nc --arg prompt "${prompt}" \
    '{messages:[{role:"user",content:$prompt}]}')"
  template_response="$("${curl_bin}" -fsS \
    -H 'Content-Type: application/json' \
    -d "${template_request}" \
    "${llama_url}/apply-template")"
  rendered_prompt="$("${jq_bin}" -er '.prompt | select(type == "string")' \
    <<<"${template_response}")"

  tokenize_request="$("${jq_bin}" -nc --arg content "${rendered_prompt}" \
    '{content:$content,add_special:false,parse_special:true}')"
  tokenize_response="$("${curl_bin}" -fsS \
    -H 'Content-Type: application/json' \
    -d "${tokenize_request}" \
    "${llama_url}/tokenize")"
  prompt_tokens="$("${jq_bin}" -ec '
    .tokens | select(type == "array" and all(.[]; type == "number"))
  ' <<<"${tokenize_response}")"

  completion_request="$("${jq_bin}" -nc \
    --arg prompt "${rendered_prompt}" \
    --argjson max_new_tokens "${max_new_tokens}" \
    '{
      prompt:$prompt,
      n_predict:$max_new_tokens,
      temperature:0,
      samplers:["temperature"],
      repeat_penalty:1,
      cache_prompt:false,
      return_tokens:true,
      reasoning_format:"none"
    }')"
  completion_response="$("${curl_bin}" -fsS \
    -H 'Content-Type: application/json' \
    -d "${completion_request}" \
    "${llama_url}/completion")"
  generated_tokens="$("${jq_bin}" -ec '
    .tokens | select(type == "array" and all(.[]; type == "number"))
  ' <<<"${completion_response}")"

  "${jq_bin}" -nc \
    --arg name "${name}" \
    --arg prompt "${prompt}" \
    --argjson prompt_tokens "${prompt_tokens}" \
    --argjson generated_tokens "${generated_tokens}" \
    '{
      name:$name,
      prompt:$prompt,
      reference_prompt_token_ids:$prompt_tokens,
      reference_generated_token_ids:$generated_tokens
    }' >>"${reference_file}"
done <"${corpus}"

kill "${server_pid}" 2>/dev/null || true
wait "${server_pid}" 2>/dev/null || true
server_pid=''

: >"${output}"

first_mismatch() {
  "${jq_bin}" -nr \
    --argjson expected "$1" \
    --argjson actual "$2" \
    '([range(0; ([($expected | length), ($actual | length)] | max)) |
      select($expected[.] != $actual[.])] | first) // -1'
}

token_at() {
  "${jq_bin}" -nc \
    --argjson tokens "$1" \
    --argjson index "$2" \
    'if $index < ($tokens | length) then $tokens[$index]
     else null
     end'
}

display_token() {
  if [[ "$1" == "null" ]]; then
    printf 'missing\n'
  else
    "${jq_bin}" -nr --argjson token "$1" '$token | tostring'
  fi
}

validate_resolved_execution() {
  local execution="$1"
  if [[ -n "${kv_cache_dtype}" ]] &&
    ! "${jq_bin}" -e --arg dtype "${kv_cache_dtype}" \
      '.kv_cache_dtype == $dtype' <<<"${execution}" >/dev/null; then
    printf 'qwen35-parity: resolved execution policy does not match requested policy\n' >&2
    exit 44
  fi
  if [[ -n "${kv_cache_layout}" ]] &&
    ! "${jq_bin}" -e --arg layout "${kv_cache_layout}" \
      '.kv_cache_layout == $layout' <<<"${execution}" >/dev/null; then
    printf 'qwen35-parity: resolved execution policy does not match requested policy\n' >&2
    exit 44
  fi
}

while IFS= read -r reference_line || [[ -n "${reference_line}" ]]; do
  [[ -n "${reference_line}" ]] || continue
  name="$("${jq_bin}" -r '.name' <<<"${reference_line}")"
  prompt="$("${jq_bin}" -r '.prompt' <<<"${reference_line}")"
  expected_prompt="$("${jq_bin}" -c '.reference_prompt_token_ids' \
    <<<"${reference_line}")"
  expected_generated="$("${jq_bin}" -c '.reference_generated_token_ids' \
    <<<"${reference_line}")"

  wait_for_gpu_preflight
  raftinfer_response="$("${raftinfer_cli}" generate \
    --model "${raftinfer_model}" \
    --prompt "${prompt}" \
    --max-new-tokens "${max_new_tokens}" \
    --context "${context_tokens}" \
    ${raftinfer_policy_args[@]+"${raftinfer_policy_args[@]}"} \
    --output-format json)"
  if ! raftinfer_execution="$("${jq_bin}" -ec '
    .execution |
    select(type == "object" and
      .attention == "online_tiled" and
      (.kv_cache_dtype | type == "string" and length > 0) and
      (.kv_cache_layout | type == "string" and length > 0))
  ' <<<"${raftinfer_response}")"; then
    printf 'qwen35-parity: resolved execution diagnostics are missing\n' >&2
    exit 44
  fi
  validate_resolved_execution "${raftinfer_execution}"
  actual_prompt="$("${jq_bin}" -ec '
    .prompt_token_ids | select(type == "array" and all(.[]; type == "number"))
  ' <<<"${raftinfer_response}")"
  actual_generated="$("${jq_bin}" -ec '
    .generated_token_ids | select(type == "array" and all(.[]; type == "number"))
  ' <<<"${raftinfer_response}")"

  mismatch_kind=''
  mismatch_index="$(first_mismatch "${expected_prompt}" "${actual_prompt}")"
  if [[ "${mismatch_index}" -ge 0 ]]; then
    mismatch_kind='prompt'
    expected_tokens="${expected_prompt}"
    actual_tokens="${actual_prompt}"
  else
    mismatch_index="$(first_mismatch "${expected_generated}" "${actual_generated}")"
    if [[ "${mismatch_index}" -ge 0 ]]; then
      mismatch_kind='generated'
      expected_tokens="${expected_generated}"
      actual_tokens="${actual_generated}"
    fi
  fi

  if [[ -n "${mismatch_kind}" ]]; then
    expected_token_json="$(token_at "${expected_tokens}" "${mismatch_index}")"
    actual_token_json="$(token_at "${actual_tokens}" "${mismatch_index}")"
    expected_token="$(display_token "${expected_token_json}")"
    actual_token="$(display_token "${actual_token_json}")"
    "${jq_bin}" -nc \
      --arg name "${name}" \
      --arg prompt "${prompt}" \
      --argjson reference_prompt_token_ids "${expected_prompt}" \
      --argjson raftinfer_prompt_token_ids "${actual_prompt}" \
      --argjson reference_generated_token_ids "${expected_generated}" \
      --argjson raftinfer_generated_token_ids "${actual_generated}" \
      --arg kind "${mismatch_kind}" \
      --argjson index "${mismatch_index}" \
      --argjson expected "${expected_token_json}" \
      --argjson actual "${actual_token_json}" \
      --argjson execution "${raftinfer_execution}" \
      '{
        schema_version:2,
        name:$name,
        prompt:$prompt,
        parity_passed:false,
        reference_prompt_token_ids:$reference_prompt_token_ids,
        raftinfer_prompt_token_ids:$raftinfer_prompt_token_ids,
        reference_generated_token_ids:$reference_generated_token_ids,
        raftinfer_generated_token_ids:$raftinfer_generated_token_ids,
        execution:$execution,
        mismatch:{kind:$kind,index:$index,expected:$expected,actual:$actual},
        diagnostic:"token-id mismatch; block/logit diagnostics unavailable through the stable C ABI"
      }' >>"${output}"
    printf '%s token mismatch at index %s: expected=%s actual=%s\n' \
      "${mismatch_kind}" "${mismatch_index}" "${expected_token}" \
      "${actual_token}" >&2
    exit 40
  fi

  "${jq_bin}" -nc \
    --arg name "${name}" \
    --arg prompt "${prompt}" \
    --argjson prompt_token_ids "${actual_prompt}" \
    --argjson generated_token_ids "${actual_generated}" \
    --argjson execution "${raftinfer_execution}" \
    '{
      schema_version:2,
      name:$name,
      prompt:$prompt,
      parity_passed:true,
      prompt_token_ids:$prompt_token_ids,
      generated_token_ids:$generated_token_ids,
      execution:$execution,
      mismatch:null
    }' >>"${output}"
done <"${reference_file}"

printf 'qwen35-parity: pass records=%s output=%s\n' \
  "$(wc -l <"${output}" | tr -d ' ')" "${output}"
