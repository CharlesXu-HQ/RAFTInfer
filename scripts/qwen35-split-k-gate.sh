#!/usr/bin/env bash
set -euo pipefail

jq_bin="${JQ_BIN:-jq}"

fail_usage() {
  printf 'qwen35-split-k-gate: %s\n' "$1" >&2
  exit 2
}

fail_gate() {
  printf 'qwen35-split-k-gate: %s\n' "$1" >&2
  exit 1
}

[[ "$#" -eq 3 ]] ||
  fail_usage \
    "usage: qwen35-split-k-gate.sh BASELINE_JSONL CANDIDATE_A_JSONL CANDIDATE_B_JSONL"
baseline_jsonl="$1"
candidate_a_jsonl="$2"
candidate_b_jsonl="$3"

for input in "${baseline_jsonl}" "${candidate_a_jsonl}" "${candidate_b_jsonl}"; do
  [[ -f "${input}" ]] || fail_usage "benchmark JSONL does not exist: ${input}"
done
[[ "${baseline_jsonl}" != "${candidate_a_jsonl}" &&
   "${baseline_jsonl}" != "${candidate_b_jsonl}" &&
   "${candidate_a_jsonl}" != "${candidate_b_jsonl}" ]] ||
  fail_usage "baseline and candidate inputs must be three distinct files"
command -v "${jq_bin}" >/dev/null ||
  fail_usage "jq command not found: ${jq_bin}"

check() {
  local expression="$1"
  local message="$2"
  if ! "${jq_bin}" -e -n \
    --slurpfile baseline "${baseline_jsonl}" \
    --slurpfile candidate_a "${candidate_a_jsonl}" \
    --slurpfile candidate_b "${candidate_b_jsonl}" \
    "${expression}" >/dev/null; then
    fail_gate "${message}"
  fi
}

check '
  def valid_file($records):
    ($records | length) == 3 and
    ([$records[].arm] | sort) == ["pp128", "pp512", "tg128_pp512"] and
    all($records[]; type == "object" and .schema_version == 2);
  valid_file($baseline) and
  valid_file($candidate_a) and
  valid_file($candidate_b)
' "each input must contain exactly one schema-v2 record for every gated arm"

check '
  def valid_shapes($records):
    all($records[];
      .generated_tokens == 128 and
      (if .arm == "pp128" then .prompt_tokens == 128
       elif .arm == "pp512" or .arm == "tg128_pp512" then
         .prompt_tokens == 512
       else false end));
  valid_shapes($baseline) and
  valid_shapes($candidate_a) and
  valid_shapes($candidate_b)
' "arm shapes must be PP128, PP512, and TG128@PP512"

check '
  [$baseline[], $candidate_a[], $candidate_b[]] as $records |
  all($records[];
    (.provenance.artifact_sha256 | type == "string" and
      test("^[0-9a-fA-F]{64}$")) and
    .provenance.raftinfer_model_sha256 == .provenance.artifact_sha256 and
    (.gpu.name | type == "string" and length > 0) and
    (.gpu.driver_version | type == "string" and length > 0) and
    (.software.cuda_version | type == "string" and length > 0) and
    (.software.raft_version | type == "string" and length > 0) and
    (.software.rmm_version | type == "string" and length > 0) and
    (.warmup_iterations | type == "number" and isfinite and
      floor == . and . > 0) and
    (.measured_iterations | type == "number" and isfinite and
      floor == . and . > 0)) and
  ([$records[] | {
    model_sha:.provenance.raftinfer_model_sha256,
    gpu_name:.gpu.name,
    driver:.gpu.driver_version,
    cuda:.software.cuda_version,
    raft:.software.raft_version,
    rmm:.software.rmm_version,
    warmups:.warmup_iterations,
    measurements:.measured_iterations
  }] | unique | length) == 1
' "artifact, GPU, software, warmup, and measurement identity must match"

check '
  def finite_positive:
    type == "number" and isfinite and . > 0;
  def valid_metrics($records):
    all($records[];
      (.raftinfer.prefill.tokens_per_second | finite_positive) and
      (.raftinfer.generation.tokens_per_second | finite_positive) and
      (.raftinfer.prefill.coefficient_of_variation |
        type == "number" and isfinite and . >= 0) and
      (.raftinfer.generation.coefficient_of_variation |
        type == "number" and isfinite and . >= 0));
  valid_metrics($baseline) and
  valid_metrics($candidate_a) and
  valid_metrics($candidate_b)
' "all gated metrics must be finite positive numbers with finite variability"

check '
  def disclosed:
    (.decode_attention == "single_block" or .decode_attention == "split_k") and
    (.decode_attention_partition_tokens | type == "number" and
      isfinite and floor == . and . >= 0) and
    (.decode_attention_threshold_tokens | type == "number" and
      isfinite and floor == . and . >= 0) and
    (.decode_attention_context_bucket_tokens | type == "number" and
      isfinite and floor == . and . >= 0) and
    (.decode_attention_split_k_graph_captured | type == "boolean");
  all([$baseline[], $candidate_a[], $candidate_b[]][];
    (.execution | disclosed) and
    (.raftinfer.execution | disclosed) and
    .execution == .raftinfer.execution)
' "decode attention diagnostics must be fully disclosed"

check '
  def candidate_ok($records):
    all($records[];
      .parity.records == 4 and
      .parity.generated_tokens_per_record == 32 and
      .parity.exact_matches == 128 and
      .parity.passed == true and
      .raftinfer.prefill.coefficient_of_variation <= 0.03 and
      .raftinfer.generation.coefficient_of_variation <= 0.03 and
      (if .arm == "pp512" or .arm == "tg128_pp512" then
         .execution.decode_attention == "split_k" and
         (.execution.decode_attention_partition_tokens == 256 or
           .execution.decode_attention_partition_tokens == 512) and
         .execution.decode_attention_threshold_tokens ==
           .execution.decode_attention_partition_tokens and
         .execution.decode_attention_context_bucket_tokens > 0 and
         .execution.decode_attention_split_k_graph_captured == true
       else true end));
  candidate_ok($candidate_a) and candidate_ok($candidate_b)
' "both candidates must have exact parity, stable RAFTInfer timings, and split-K long-context execution"

check '
  def arm($records; $name): first($records[] | select(.arm == $name));
  def promoted($candidate):
    (arm($baseline; "pp128")) as $baseline_pp128 |
    (arm($baseline; "pp512")) as $baseline_pp512 |
    (arm($baseline; "tg128_pp512")) as $baseline_tg128 |
    (arm($candidate; "pp128")) as $candidate_pp128 |
    (arm($candidate; "pp512")) as $candidate_pp512 |
    (arm($candidate; "tg128_pp512")) as $candidate_tg128 |
    $candidate_pp512.raftinfer.generation.tokens_per_second >=
      ($baseline_pp512.raftinfer.generation.tokens_per_second * 1.01) and
    $candidate_tg128.raftinfer.generation.tokens_per_second >=
      ($baseline_tg128.raftinfer.generation.tokens_per_second * 1.01) and
    $candidate_pp128.raftinfer.prefill.tokens_per_second >=
      ($baseline_pp128.raftinfer.prefill.tokens_per_second * 0.99) and
    $candidate_pp512.raftinfer.prefill.tokens_per_second >=
      ($baseline_pp512.raftinfer.prefill.tokens_per_second * 0.99) and
    $candidate_pp128.raftinfer.generation.tokens_per_second >=
      ($baseline_pp128.raftinfer.generation.tokens_per_second * 0.99);
  promoted($candidate_a) and promoted($candidate_b)
' "both candidates must independently satisfy every promotion threshold"

printf 'qwen35-split-k-gate: pass baseline=%s candidate_a=%s candidate_b=%s\n' \
  "${baseline_jsonl}" "${candidate_a_jsonl}" "${candidate_b_jsonl}"
