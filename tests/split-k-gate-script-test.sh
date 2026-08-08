#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/raftinfer-split-k-gate.XXXXXX")"
trap 'rm -rf "${fixture_root}"' EXIT

gate="${repo_root}/scripts/qwen35-split-k-gate.sh"
baseline="${fixture_root}/baseline.jsonl"
candidate_a="${fixture_root}/candidate-a.jsonl"
candidate_b="${fixture_root}/candidate-b.jsonl"

write_fixture() {
  local output="$1"
  local kind="$2"
  shift 2
  : >"${output}"
  for arm in "$@"; do
    case "${arm}" in
      pp128)
        prompt_tokens=128
        ;;
      pp512 | tg128_pp512)
        prompt_tokens=512
        ;;
    esac
    if [[ "${kind}" == baseline ]]; then
      prefill_tps=100000
      generation_tps=100000
      decode_attention=single_block
      partition_tokens=0
      threshold_tokens=0
      context_bucket_tokens=0
      split_k_graph_captured=false
    else
      prefill_tps=100000
      generation_tps=100000
      decode_attention=split_k
      partition_tokens=256
      threshold_tokens=256
      context_bucket_tokens=1024
      split_k_graph_captured=true
      case "${arm}" in
        pp128)
          prefill_tps=99000
          generation_tps=99000
          decode_attention=single_block
          partition_tokens=0
          threshold_tokens=0
          context_bucket_tokens=0
          split_k_graph_captured=false
          ;;
        pp512)
          [[ "${kind}" == candidate_a ]] && generation_tps=101000 || generation_tps=102000
          [[ "${kind}" == candidate_a ]] && prefill_tps=99000 || prefill_tps=100000
          ;;
        tg128_pp512)
          [[ "${kind}" == candidate_a ]] && generation_tps=101000 || generation_tps=103000
          ;;
      esac
    fi

    jq -nc \
      --arg arm "${arm}" \
      --argjson prompt_tokens "${prompt_tokens}" \
      --argjson prefill_tps "${prefill_tps}" \
      --argjson generation_tps "${generation_tps}" \
      --arg decode_attention "${decode_attention}" \
      --argjson partition_tokens "${partition_tokens}" \
      --argjson threshold_tokens "${threshold_tokens}" \
      --argjson context_bucket_tokens "${context_bucket_tokens}" \
      --argjson split_k_graph_captured "${split_k_graph_captured}" '
        {
          schema_version:2,
          arm:$arm,
          prompt_tokens:$prompt_tokens,
          generated_tokens:128,
          warmup_iterations:5,
          measured_iterations:20,
          provenance:{
            artifact_sha256:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            raftinfer_model_sha256:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          },
          parity:{
            records:4,
            generated_tokens_per_record:32,
            exact_matches:128,
            passed:true
          },
          gpu:{name:"NVIDIA GeForce RTX 5090",driver_version:"580.65"},
          software:{cuda_version:"13.2",raft_version:"26.06",rmm_version:"26.06"},
          execution:{
            decode_attention:$decode_attention,
            decode_attention_partition_tokens:$partition_tokens,
            decode_attention_threshold_tokens:$threshold_tokens,
            decode_attention_context_bucket_tokens:$context_bucket_tokens,
            decode_attention_split_k_graph_captured:$split_k_graph_captured
          },
          raftinfer:{
            prefill:{tokens_per_second:$prefill_tps,coefficient_of_variation:0.01},
            generation:{tokens_per_second:$generation_tps,coefficient_of_variation:0.01},
            execution:{
              decode_attention:$decode_attention,
              decode_attention_partition_tokens:$partition_tokens,
              decode_attention_threshold_tokens:$threshold_tokens,
              decode_attention_context_bucket_tokens:$context_bucket_tokens,
              decode_attention_split_k_graph_captured:$split_k_graph_captured
            }
          }
        }
      ' >>"${output}"
  done
}

write_pass_fixtures() {
  write_fixture "${baseline}" baseline pp128 pp512 tg128_pp512
  # Deliberately reorder both candidates: the evaluator must join by arm.
  write_fixture "${candidate_a}" candidate_a tg128_pp512 pp128 pp512
  write_fixture "${candidate_b}" candidate_b pp512 tg128_pp512 pp128
}

expect_fail() {
  local name="$1"
  local candidate="$2"
  local filter="$3"
  local expected="${4:-qwen35-split-k-gate:}"
  write_pass_fixtures
  local target
  case "${candidate}" in
    baseline) target="${baseline}" ;;
    candidate_a) target="${candidate_a}" ;;
    candidate_b) target="${candidate_b}" ;;
  esac
  jq -c "${filter}" "${target}" >"${fixture_root}/${name}.jsonl"
  cp "${fixture_root}/${name}.jsonl" "${target}"
  set +e
  "${gate}" "${baseline}" "${candidate_a}" "${candidate_b}" \
    >"${fixture_root}/${name}.out" 2>"${fixture_root}/${name}.err"
  local status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    printf 'expected split-K gate failure for %s\n' "${name}" >&2
    exit 1
  fi
  grep -F "${expected}" "${fixture_root}/${name}.err"
}

write_pass_fixtures
"${gate}" "${baseline}" "${candidate_a}" "${candidate_b}" \
  >"${fixture_root}/pass.out"
grep -F "qwen35-split-k-gate: pass baseline=${baseline} candidate_a=${candidate_a} candidate_b=${candidate_b}" \
  "${fixture_root}/pass.out"

expect_fail pp512_generation candidate_a \
  'if .arm == "pp512" then .raftinfer.generation.tokens_per_second = 100999 else . end'
expect_fail tg128_generation candidate_b \
  'if .arm == "tg128_pp512" then .raftinfer.generation.tokens_per_second = 100999 else . end'
expect_fail pp128_prefill candidate_a \
  'if .arm == "pp128" then .raftinfer.prefill.tokens_per_second = 98999 else . end'
expect_fail pp512_prefill candidate_b \
  'if .arm == "pp512" then .raftinfer.prefill.tokens_per_second = 98999 else . end'
expect_fail pp128_generation candidate_b \
  'if .arm == "pp128" then .raftinfer.generation.tokens_per_second = 98999 else . end'
expect_fail bad_parity candidate_a \
  'if .arm == "pp512" then .parity.exact_matches = 127 else . end'
expect_fail incomplete_parity candidate_b \
  'if .arm == "pp512" then .parity.generated_tokens_per_record = 31 else . end'
expect_fail unstable_cv candidate_b \
  'if .arm == "pp512" then .raftinfer.generation.coefficient_of_variation = 0.030001 else . end'
expect_fail missing_arm candidate_a 'select(.arm != "pp512")'
expect_fail artifact_mismatch candidate_b \
  '.provenance.artifact_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"'
expect_fail model_sha_mismatch candidate_a \
  '.provenance.raftinfer_model_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"'
expect_fail gpu_mismatch candidate_b '.gpu.name = "Different GPU"'
expect_fail driver_mismatch candidate_a '.gpu.driver_version = "different"'
expect_fail cuda_mismatch candidate_b '.software.cuda_version = "different"'
expect_fail raft_mismatch candidate_a '.software.raft_version = "different"'
expect_fail rmm_mismatch candidate_b '.software.rmm_version = "different"'
expect_fail warmup_mismatch candidate_a '.warmup_iterations = 4'
expect_fail measured_mismatch candidate_b '.measured_iterations = 19'
expect_fail nonfinite_value candidate_a \
  'if .arm == "pp512" then .raftinfer.generation.tokens_per_second = 1e999 else . end'
expect_fail undisclosed_path candidate_b \
  'if .arm == "pp512" then del(.execution.decode_attention_partition_tokens) else . end'
expect_fail fallback_path candidate_a \
  'if .arm == "tg128_pp512" then
     .execution.decode_attention = "single_block" |
     .execution.decode_attention_partition_tokens = 0 |
     .execution.decode_attention_threshold_tokens = 0 |
     .execution.decode_attention_context_bucket_tokens = 0 |
     .execution.decode_attention_split_k_graph_captured = false |
     .raftinfer.execution = .execution
   else . end' \
  'both candidates must have exact parity, stable RAFTInfer timings, and split-K long-context execution'

write_pass_fixtures
set +e
"${gate}" "${baseline}" "${candidate_a}" \
  >"${fixture_root}/single-candidate.out" 2>"${fixture_root}/single-candidate.err"
single_candidate_status=$?
set -e
[[ "${single_candidate_status}" -eq 2 ]]
grep -F 'usage: qwen35-split-k-gate.sh BASELINE_JSONL CANDIDATE_A_JSONL CANDIDATE_B_JSONL' \
  "${fixture_root}/single-candidate.err"

printf 'split-k-gate-script-test: pass\n'
