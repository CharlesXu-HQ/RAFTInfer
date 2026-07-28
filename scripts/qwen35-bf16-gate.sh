#!/usr/bin/env bash
set -euo pipefail

jq_bin="${JQ_BIN:-jq}"

fail_usage() {
  printf 'qwen35-bf16-gate: %s\n' "$1" >&2
  exit 2
}

fail_gate() {
  printf 'qwen35-bf16-gate: %s\n' "$1" >&2
  exit 1
}

[[ "$#" -eq 1 ]] || fail_usage "usage: qwen35-bf16-gate.sh BENCHMARK_JSONL"
benchmark_jsonl="$1"
[[ -f "${benchmark_jsonl}" ]] ||
  fail_usage "benchmark JSONL does not exist: ${benchmark_jsonl}"
command -v "${jq_bin}" >/dev/null ||
  fail_usage "jq command not found: ${jq_bin}"

check() {
  local expression="$1"
  local message="$2"
  if ! "${jq_bin}" -e -s "${expression}" "${benchmark_jsonl}" >/dev/null; then
    fail_gate "${message}"
  fi
}

check '
  def valid_record:
    type == "object" and
    (.schema_version == 1);
  length > 0 and all(.[]; valid_record)
' "malformed benchmark JSONL"

check '
  def required_arm: . == "pp128" or . == "pp512" or . == "tg128_pp512";
  ([.[].arm] | all(.[]; type == "string" and required_arm))
' "unknown arm"

check '
  ([.[].arm] | sort) == ["pp128", "pp512", "tg128_pp512"]
' "exactly one record is required for each gated arm"

check '
  def finite_positive: type == "number" and isfinite and . > 0;
  all(.[]; .prompt_tokens | finite_positive) and
  all(.[]; .generated_tokens == 128) and
  all(.[]; .peak_allocated_gpu_bytes | finite_positive) and
  all(.[]; .throughput_ratio.prefill | finite_positive) and
  all(.[]; .throughput_ratio.generation | finite_positive)
' "finite positive number is required for every gated metric"

check '
  all(.[]; if .arm == "pp128" then .prompt_tokens == 128
           elif .arm == "pp512" then .prompt_tokens == 512
           elif .arm == "tg128_pp512" then .prompt_tokens == 512
           else false end) and
  all(.[]; .generated_tokens == 128)
' "arm shape mapping must be pp128=128, pp512=512, tg128_pp512=512 with 128 generated tokens"

check 'all(.[]; .warmup_iterations | type == "number" and floor == . and . >= 3)' \
  "warmups must be at least 3"

check 'all(.[]; .measured_iterations | type == "number" and floor == . and . >= 7)' \
  "measurements must be at least 7"

check '
  all(.[]; .provenance.weight_format == "bf16")
' "weight format must be bf16"

check '
  all(.[]; (.provenance.path | type == "string" and length > 0) and
           (.provenance.sha256 | type == "string" and
             test("^[0-9a-fA-F]{64}$")))
' "provenance path and sha256 are required"

check '
  all(.[]; .provenance.llama_cpp_revision | type == "string" and
           test("^[0-9a-fA-F]{40}$")) and
  all(.[]; .llama_cpp.revision == .provenance.llama_cpp_revision)
' "pinned llama.cpp revision is required"

check '
  all(.[]; (.provenance.artifact_sha256 | type == "string" and
             test("^[0-9a-fA-F]{64}$")) and
           .provenance.brt_model_sha256 == .provenance.artifact_sha256 and
           .provenance.llama_model_sha256 == .provenance.artifact_sha256)
' "measured model SHA256 must match provenance artifact SHA256"

check '
  all(.[]; (.provenance.llama_server_sha256 | type == "string" and
             test("^[0-9a-fA-F]{64}$")) and
           (.provenance.llama_server_version | type == "string" and
             length > 0) and
           (.provenance.llama_cpp_revision[0:7]) as $revision_short |
           (.provenance.llama_server_version | contains($revision_short)) and
           .provenance.llama_cpp_revision_verified == true)
' "llama-server version must verify pinned revision"

check '
  all(.[]; .parity.records == 4 and
           .parity.generated_tokens_per_record == 32 and
           .parity.exact_matches == 128 and
           .parity.passed == true)
' "exact parity report must contain 4 records x 32 generated tokens, all passed"

check '
  all(.[]; .execution.attention == "online_tiled" and
           .brt.execution.attention == .execution.attention)
' "resolved attention must be online_tiled"

check '
  all(.[]; (.execution.kv_cache_dtype | type == "string" and
             . == "bf16") and
           .brt.execution.kv_cache_dtype == .execution.kv_cache_dtype)
' "resolved kv_cache_dtype must be disclosed"

check '
  all(.[]; (.execution.kv_cache_layout | type == "string" and
             (. == "head-major")) and
           .brt.execution.kv_cache_layout == .execution.kv_cache_layout)
' "resolved kv_cache_layout must be disclosed"

check '
  all(.[]; .parity.execution.attention == .execution.attention and
           .parity.execution.kv_cache_dtype == .execution.kv_cache_dtype and
           .parity.execution.kv_cache_layout == .execution.kv_cache_layout)
' "parity execution must match benchmark execution"

check '
  all(.[]; if .arm == "tg128_pp512" then
             .execution.decode_graph_replayed == true and
             .brt.execution.decode_graph_replayed == true
           else true end)
' "TG128@PP512 must report decode graph replay"

check '
  def summary_ok:
    (.min_us | type == "number" and isfinite and . > 0) and
    (.median_us | type == "number" and isfinite and . > 0) and
    (.mean_us | type == "number" and isfinite and . > 0) and
    (.p95_us | type == "number" and isfinite and . > 0) and
    (.max_us | type == "number" and isfinite and . > 0) and
    (.tokens_per_second | type == "number" and isfinite and . > 0) and
    (.coefficient_of_variation | type == "number" and isfinite and . >= 0);
  all(.[]; .brt.prefill | summary_ok) and
  all(.[]; .brt.generation | summary_ok) and
  all(.[]; .llama_cpp.prefill | summary_ok) and
  all(.[]; .llama_cpp.generation | summary_ok)
' "latency summaries must contain finite positive numbers"

check '
  all(.[]; .brt.prefill.coefficient_of_variation <= 0.03 and
           .brt.generation.coefficient_of_variation <= 0.03)
' "BRT coefficient of variation must be at most 0.03"

check '
  all(.[]; .llama_cpp.prefill.coefficient_of_variation <= 0.03 and
           .llama_cpp.generation.coefficient_of_variation <= 0.03)
' "llama coefficient of variation must be at most 0.03"

check '
  any(.[]; .arm == "pp128" and .throughput_ratio.prefill >= 1.0)
' "pp128 prefill ratio must be at least 1.0"

check '
  any(.[]; .arm == "pp512" and .throughput_ratio.prefill >= 1.0)
' "pp512 prefill ratio must be at least 1.0"

check '
  any(.[]; .arm == "tg128_pp512" and .throughput_ratio.generation >= 1.0)
' "tg128_pp512 generation ratio must be at least 1.0"

check '
  [
    .[] | if .arm == "pp128" then .throughput_ratio.prefill
          elif .arm == "pp512" then .throughput_ratio.prefill
          elif .arm == "tg128_pp512" then .throughput_ratio.generation
          else empty end
  ] | max >= 1.1
' "at least one gated ratio must be at least 1.1"

printf 'qwen35-bf16-gate: pass records=3 input=%s\n' "${benchmark_jsonl}"
