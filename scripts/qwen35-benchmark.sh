#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parity_report="${PARITY_REPORT:-${repo_root}/build/evidence/qwen35-parity.jsonl}"
provenance_json="${PROVENANCE_JSON:-${repo_root}/build/evidence/qwen35-bf16.provenance.json}"
output="${BENCHMARK_OUTPUT:-${repo_root}/build/evidence/qwen35-benchmark.jsonl}"
brt_model="${BRT_MODEL:-}"
llama_model="${LLAMA_MODEL:-${brt_model}}"
brt_cli="${BRT_CLI:-${repo_root}/target/release/brt-cli}"
llama_server="${LLAMA_SERVER_BIN:-}"
gpu_preflight="${GPU_PREFLIGHT:-${repo_root}/scripts/gpu-preflight.sh}"
gpu_lock="${BRT_GPU_LOCK:-/tmp/brt-qwen35-gpu.lock}"
curl_bin="${CURL_BIN:-curl}"
jq_bin="${JQ_BIN:-jq}"
nvidia_smi="${NVIDIA_SMI_BIN:-nvidia-smi}"
context_tokens="${BRT_CONTEXT_TOKENS:-4096}"
warmup_iterations=5
measured_iterations=20
generated_tokens=128
kv_cache_dtype="${BRT_KV_CACHE_DTYPE:-bf16}"
kv_cache_layout="${BRT_KV_CACHE_LAYOUT:-head-major}"
llama_port="${LLAMA_SERVER_PORT:-18081}"
llama_url="http://127.0.0.1:${llama_port}"
preflight_retries="${BRT_PREFLIGHT_RETRIES:-30}"
preflight_retry_seconds="${BRT_PREFLIGHT_RETRY_SECONDS:-1}"

fail_usage() {
  printf 'qwen35-benchmark: %s\n' "$1" >&2
  exit 2
}

sha256_file() {
  if command -v shasum >/dev/null; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

[[ -n "${brt_model}" ]] || fail_usage "BRT_MODEL is required"
[[ -f "${brt_model}" ]] || fail_usage "BRT_MODEL does not exist: ${brt_model}"
[[ -f "${llama_model}" ]] || fail_usage "LLAMA_MODEL does not exist: ${llama_model}"
[[ -x "${brt_cli}" ]] || fail_usage "BRT_CLI is not executable: ${brt_cli}"
[[ -n "${llama_server}" && -x "${llama_server}" ]] ||
  fail_usage "LLAMA_SERVER_BIN must name an executable"
[[ -x "${gpu_preflight}" ]] ||
  fail_usage "GPU_PREFLIGHT is not executable: ${gpu_preflight}"
[[ -f "${parity_report}" ]] ||
  fail_usage "PARITY_REPORT does not exist: ${parity_report}"
[[ -f "${provenance_json}" ]] ||
  fail_usage "PROVENANCE_JSON does not exist: ${provenance_json}"
command -v "${curl_bin}" >/dev/null ||
  fail_usage "curl command not found: ${curl_bin}"
command -v "${jq_bin}" >/dev/null ||
  fail_usage "jq command not found: ${jq_bin}"
command -v "${nvidia_smi}" >/dev/null ||
  fail_usage "nvidia-smi command not found: ${nvidia_smi}"
[[ "${preflight_retries}" =~ ^[1-9][0-9]*$ ]] ||
  fail_usage "BRT_PREFLIGHT_RETRIES must be a positive integer"
[[ "${preflight_retry_seconds}" =~ ^[0-9]+$ ]] ||
  fail_usage "BRT_PREFLIGHT_RETRY_SECONDS must be a non-negative integer"
[[ "${kv_cache_dtype}" == "f32" || "${kv_cache_dtype}" == "bf16" ]] ||
  fail_usage "BRT_KV_CACHE_DTYPE must be f32 or bf16"
[[ "${kv_cache_layout}" == "token-major" || "${kv_cache_layout}" == "head-major" ]] ||
  fail_usage "BRT_KV_CACHE_LAYOUT must be token-major or head-major"

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

if ! "${jq_bin}" -e -s '
  length == 4 and
  all(.[]; .parity_passed == true and
           (.generated_token_ids | type == "array" and length == 32)) and
  ([.[].prompt_token_ids[]] | length > 0)
' "${parity_report}" >/dev/null; then
  printf 'qwen35-benchmark: parity report is not fully passing\n' >&2
  exit 41
fi
if ! "${jq_bin}" -e -s '
  all(.[]; .execution | type == "object" and
    (.attention | type == "string" and length > 0) and
    (.kv_cache_dtype | type == "string" and length > 0) and
    (.kv_cache_layout | type == "string" and length > 0))
' "${parity_report}" >/dev/null; then
  printf 'qwen35-benchmark: parity execution diagnostics are missing\n' >&2
  exit 41
fi
if ! parity_execution="$("${jq_bin}" -ecs '
  [.[].execution] as $executions |
  if (($executions | unique | length) == 1) then
    $executions[0]
  else
    empty
  end
' "${parity_report}")"; then
  printf 'qwen35-benchmark: parity execution diagnostics are inconsistent\n' >&2
  exit 41
fi
if ! "${jq_bin}" -e \
  --arg dtype "${kv_cache_dtype}" \
  --arg layout "${kv_cache_layout}" '
    .attention == "online_tiled" and
    .kv_cache_dtype == $dtype and
    .kv_cache_layout == $layout
  ' <<<"${parity_execution}" >/dev/null; then
  printf 'qwen35-benchmark: parity execution does not match benchmark policy\n' >&2
  exit 41
fi

if ! "${jq_bin}" -e '
  .schema_version == 1 and
  .conversion.outtype == "bf16" and
  (.llama_cpp.reference_revision | type == "string" and
    test("^[0-9a-fA-F]{40}$")) and
  (.artifact.sha256 | type == "string" and
    test("^[0-9a-fA-F]{64}$"))
' "${provenance_json}" >/dev/null; then
  printf 'qwen35-benchmark: provenance JSON is not a pinned BF16 artifact\n' >&2
  exit 43
fi

artifact_sha="$("${jq_bin}" -er '.artifact.sha256' "${provenance_json}")"
brt_model_sha="$(sha256_file "${brt_model}")"
llama_model_sha="$(sha256_file "${llama_model}")"
llama_server_sha="$(sha256_file "${llama_server}")"
if [[ "${brt_model_sha}" != "${artifact_sha}" ||
      "${llama_model_sha}" != "${artifact_sha}" ]]; then
  printf 'qwen35-benchmark: model SHA256 does not match provenance artifact\n' >&2
  exit 43
fi
llama_revision="$("${jq_bin}" -er '.llama_cpp.reference_revision' \
  "${provenance_json}")"
llama_revision_short="${llama_revision:0:7}"
llama_server_dir="$(dirname "${llama_server}")"
if ! llama_server_version="$(
  LD_LIBRARY_PATH="${llama_server_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${llama_server}" --version 2>&1
)"; then
  printf 'qwen35-benchmark: llama-server --version failed\n' >&2
  exit 43
fi
llama_server_version="${llama_server_version//$'\r'/}"
llama_server_version="${llama_server_version%$'\n'}"
if [[ -z "${llama_server_version}" ||
      "${llama_server_version}" == *unknown* ||
      "${llama_server_version}" != *"${llama_revision_short}"* ]]; then
  printf 'qwen35-benchmark: llama-server --version did not verify pinned revision\n' >&2
  exit 43
fi
provenance_sha="$(sha256_file "${provenance_json}")"
provenance_payload="$("${jq_bin}" -c \
  --arg path "${provenance_json}" \
  --arg sha "${provenance_sha}" \
  --arg brt_model_sha "${brt_model_sha}" \
  --arg llama_model_sha "${llama_model_sha}" \
  --arg llama_server_sha "${llama_server_sha}" \
  --arg llama_server_version "${llama_server_version}" '
    {
      path:$path,
      sha256:$sha,
      weight_format:.conversion.outtype,
      llama_cpp_revision:.llama_cpp.reference_revision,
      model_revision:.model.revision,
      artifact_sha256:.artifact.sha256,
      brt_model_sha256:$brt_model_sha,
      llama_model_sha256:$llama_model_sha,
      llama_server_sha256:$llama_server_sha,
      llama_server_version:$llama_server_version,
      llama_cpp_revision_verified:true
    }
  ' "${provenance_json}")"
parity_payload="$("${jq_bin}" -cs '
  {
    records:length,
    generated_tokens_per_record:32,
    exact_matches:([.[].generated_token_ids | length] | add),
    passed:(length == 4 and all(.[]; .parity_passed == true and
      (.generated_token_ids | length == 32))),
    execution:.[0].execution
  }
' "${parity_report}")"
base_tokens="$("${jq_bin}" -cs '[.[].prompt_token_ids[]]' "${parity_report}")"
mkdir -p "$(dirname "${output}")"
brt_results="$(mktemp "${TMPDIR:-/tmp}/brt-qwen35-benchmark-brt.XXXXXX")"
llama_results="$(mktemp "${TMPDIR:-/tmp}/brt-qwen35-benchmark-llama.XXXXXX")"
timing_samples="$(mktemp "${TMPDIR:-/tmp}/brt-qwen35-benchmark-timings.XXXXXX")"
server_log="$(mktemp "${TMPDIR:-/tmp}/brt-qwen35-benchmark-server.XXXXXX")"
server_pid=''

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  rm -f "${brt_results}" "${llama_results}" "${timing_samples}" \
    "${server_log}"
}
trap cleanup EXIT

exec 9>"${gpu_lock}"
if ! flock -n 9; then
  printf 'qwen35-benchmark: GPU lock is held: %s\n' "${gpu_lock}" >&2
  exit 30
fi

make_prompt_tokens() {
  "${jq_bin}" -nc \
    --argjson base "${base_tokens}" \
    --argjson count "$1" \
    '[range(0; $count) | $base[. % ($base | length)]]'
}

: >"${brt_results}"
for arm_spec in pp128:128 pp512:512 tg128_pp512:512; do
  arm="${arm_spec%%:*}"
  prompt_count="${arm_spec##*:}"
  prompt_tokens="$(make_prompt_tokens "${prompt_count}")"
  prompt_csv="$("${jq_bin}" -jr 'join(",")' <<<"${prompt_tokens}")"
  wait_for_gpu_preflight
  brt_result="$("${brt_cli}" benchmark \
    --model "${brt_model}" \
    --prompt-token-ids "${prompt_csv}" \
    --prompt-tokens "${prompt_count}" \
    --decode-tokens "${generated_tokens}" \
    --context "${context_tokens}" \
    --kv-cache-dtype "${kv_cache_dtype}" \
    --kv-cache-layout "${kv_cache_layout}" \
    --warmups "${warmup_iterations}" \
    --iterations "${measured_iterations}")"
  if ! "${jq_bin}" -e \
    --arg arm "${arm}" \
    --arg dtype "${kv_cache_dtype}" \
    --arg layout "${kv_cache_layout}" \
    --argjson prompt_count "${prompt_count}" \
    --argjson generated_tokens "${generated_tokens}" \
    --argjson warmups "${warmup_iterations}" \
    --argjson iterations "${measured_iterations}" '
      .schema_version == 1 and
      .prompt_tokens == $prompt_count and
      .generated_tokens == $generated_tokens and
      .warmup_iterations == $warmups and
      .measured_iterations == $iterations and
      (.peak_allocated_gpu_bytes |
        type == "number" and . > 0 and floor == .) and
      (.prefill.median_us | type == "number" and . > 0) and
      (.prefill.mean_us | type == "number" and isfinite and . > 0) and
      (.prefill.coefficient_of_variation | type == "number" and isfinite and . >= 0) and
      (.generation.median_us | type == "number" and . > 0) and
      (.generation.mean_us | type == "number" and isfinite and . > 0) and
      (.generation.coefficient_of_variation | type == "number" and isfinite and . >= 0) and
      .execution.attention == "online_tiled" and
      .execution.kv_cache_dtype == $dtype and
      .execution.kv_cache_layout == $layout and
      (if $arm == "tg128_pp512" then
         .execution.decode_graph_replayed == true
       else true end)
    ' <<<"${brt_result}" >/dev/null; then
    printf 'qwen35-benchmark: malformed BRT benchmark output for %s\n' \
      "${arm}" >&2
    exit 32
  fi
  "${jq_bin}" -c --arg arm "${arm}" '. + {arm:$arm}' <<<"${brt_result}" \
    >>"${brt_results}"
done

gpu_row="$("${nvidia_smi}" \
  --query-gpu=name,driver_version,clocks.sm,clocks.mem \
  --format=csv,noheader,nounits \
  --id="${BRT_GPU_ID:-0}")"
if [[ "$(wc -l <<<"${gpu_row}" | tr -d ' ')" -ne 1 ]]; then
  printf 'qwen35-benchmark: expected one nvidia-smi GPU row\n' >&2
  exit 33
fi
IFS=',' read -r gpu_name driver_version sm_clock_mhz memory_clock_mhz \
  <<<"${gpu_row}"
trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}
gpu_name="$(trim "${gpu_name}")"
driver_version="$(trim "${driver_version}")"
sm_clock_mhz="$(trim "${sm_clock_mhz}")"
memory_clock_mhz="$(trim "${memory_clock_mhz}")"

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
  printf 'qwen35-benchmark: llama-server did not become ready\n' >&2
  tail -100 "${server_log}" >&2 || true
  exit 34
fi

: >"${llama_results}"
for arm_spec in pp128:128 pp512:512 tg128_pp512:512; do
  arm="${arm_spec%%:*}"
  prompt_count="${arm_spec##*:}"
  prompt_tokens="$(make_prompt_tokens "${prompt_count}")"
  completion_request="$("${jq_bin}" -nc \
    --argjson prompt "${prompt_tokens}" \
    --argjson generated_tokens "${generated_tokens}" \
    '{
      prompt:$prompt,
      n_predict:$generated_tokens,
      temperature:0,
      samplers:["temperature"],
      repeat_penalty:1,
      cache_prompt:false,
      ignore_eos:true,
      return_tokens:false,
      reasoning_format:"none"
    }')"
  : >"${timing_samples}"
  total_iterations=$((warmup_iterations + measured_iterations))
  for iteration in $(seq 1 "${total_iterations}"); do
    completion_response="$("${curl_bin}" -fsS \
      -H 'Content-Type: application/json' \
      -d "${completion_request}" \
      "${llama_url}/completion")"
    if ! "${jq_bin}" -e \
      --argjson prompt_count "${prompt_count}" \
      --argjson generated_tokens "${generated_tokens}" '
        .timings.prompt_n == $prompt_count and
        .timings.predicted_n == $generated_tokens and
        (.timings.prompt_ms | type == "number" and . > 0) and
        (.timings.predicted_ms | type == "number" and . > 0)
      ' <<<"${completion_response}" >/dev/null; then
      printf 'qwen35-benchmark: malformed llama.cpp timing for PP%s\n' \
        "${prompt_count}" >&2
      exit 35
    fi
    if [[ "${iteration}" -gt "${warmup_iterations}" ]]; then
      "${jq_bin}" -c '.timings' <<<"${completion_response}" \
        >>"${timing_samples}"
    fi
  done

  "${jq_bin}" -cs \
    --argjson prompt_tokens "${prompt_count}" \
    --argjson generated_tokens "${generated_tokens}" \
    --argjson warmups "${warmup_iterations}" \
    --argjson iterations "${measured_iterations}" \
    --arg arm "${arm}" \
    --argjson provenance "${provenance_payload}" '
      def stats:
        sort as $values |
        ($values | length) as $count |
        (($values | add) / $count) as $mean |
        {
          min_us:$values[0],
          median_us:(
            if ($count % 2) == 0 then
              (($values[($count / 2) - 1] + $values[$count / 2]) / 2)
            else
              $values[($count / 2) | floor]
            end
          ),
          mean_us:$mean,
          p95_us:$values[((($count * 95 + 99) / 100) | floor) - 1],
          max_us:$values[-1],
          coefficient_of_variation:(
            ((($values | map((. - $mean) * (. - $mean)) | add) / $count)
              | sqrt) / $mean
          )
        };
      {
        schema_version:1,
        arm:$arm,
        prompt_tokens:$prompt_tokens,
        generated_tokens:$generated_tokens,
        warmup_iterations:$warmups,
        measured_iterations:$iterations,
        revision:$provenance.llama_cpp_revision,
        prefill:(map(.prompt_ms * 1000) | stats),
        generation:(map(.predicted_ms * 1000) | stats)
      } |
      .prefill.tokens_per_second =
        ($prompt_tokens * 1000000 / .prefill.median_us) |
      .generation.tokens_per_second =
        ($generated_tokens * 1000000 / .generation.median_us)
    ' "${timing_samples}" >>"${llama_results}"
done

kill "${server_pid}" 2>/dev/null || true
wait "${server_pid}" 2>/dev/null || true
server_pid=''

: >"${output}"
floor_failed=0
for arm in pp128 pp512 tg128_pp512; do
  brt_result="$("${jq_bin}" -ec \
    --arg arm "${arm}" \
    'select(.arm == $arm)' "${brt_results}")"
  if ! "${jq_bin}" -e \
    --argjson parity_execution "${parity_execution}" \
    '.execution.attention == $parity_execution.attention and
     .execution.kv_cache_dtype == $parity_execution.kv_cache_dtype and
     .execution.kv_cache_layout == $parity_execution.kv_cache_layout' \
    <<<"${brt_result}" >/dev/null; then
    printf 'qwen35-benchmark: BRT benchmark execution does not match parity execution\n' >&2
    exit 32
  fi
  llama_result="$("${jq_bin}" -ec \
    --arg arm "${arm}" \
    'select(.arm == $arm)' "${llama_results}")"
  record="$("${jq_bin}" -nc \
    --argjson generated_tokens "${generated_tokens}" \
    --argjson warmups "${warmup_iterations}" \
    --argjson iterations "${measured_iterations}" \
    --argjson brt "${brt_result}" \
    --argjson llama "${llama_result}" \
    --arg arm "${arm}" \
    --argjson provenance "${provenance_payload}" \
    --argjson parity "${parity_payload}" \
    --arg gpu_name "${gpu_name}" \
    --arg driver_version "${driver_version}" \
    --arg cuda_version "${CUDA_VERSION:-13.2}" \
    --arg raft_version "${RAFT_VERSION:-26.06}" \
    --arg rmm_version "${RMM_VERSION:-26.06}" \
    --arg sm_clock_mhz "${sm_clock_mhz}" \
    --arg memory_clock_mhz "${memory_clock_mhz}" '
      ($brt.prefill.tokens_per_second /
        $llama.prefill.tokens_per_second) as $prefill_ratio |
      ($brt.generation.tokens_per_second /
        $llama.generation.tokens_per_second) as $generation_ratio |
      {
        schema_version:1,
        arm:$arm,
        prompt_tokens:$brt.prompt_tokens,
        generated_tokens:$generated_tokens,
        warmup_iterations:$warmups,
        measured_iterations:$iterations,
        provenance:$provenance,
        parity:$parity,
        execution:$brt.execution,
        gpu:{
          name:$gpu_name,
          driver_version:$driver_version,
          sm_clock_mhz:($sm_clock_mhz | tonumber),
          memory_clock_mhz:($memory_clock_mhz | tonumber)
        },
        software:{
          cuda_version:$cuda_version,
          raft_version:$raft_version,
          rmm_version:$rmm_version
        },
        brt:$brt,
        llama_cpp:$llama,
        throughput_ratio:{
          prefill:$prefill_ratio,
          generation:$generation_ratio
        },
        performance_floor:1.0,
        performance_floor_passed:
          (if $arm == "tg128_pp512" then
             $generation_ratio >= 1.0
           else
             $prefill_ratio >= 1.0
           end),
        peak_allocated_gpu_bytes:$brt.peak_allocated_gpu_bytes,
        peak_memory_status:"measured_by_brt_rmm"
      }
    ')"
  "${jq_bin}" -c . <<<"${record}" >>"${output}"
  if ! "${jq_bin}" -e '.performance_floor_passed == true' \
    <<<"${record}" >/dev/null; then
    floor_failed=1
  fi
done

if [[ "${floor_failed}" -ne 0 ]]; then
  printf 'qwen35-benchmark: performance floor failed; see %s\n' \
    "${output}" >&2
  exit 42
fi

printf 'qwen35-benchmark: pass records=3 output=%s\n' "${output}"
