#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/raftinfer-bf16-gate.XXXXXX")"
trap 'rm -rf "${fixture_root}"' EXIT

gate="${repo_root}/scripts/qwen35-bf16-gate.sh"
jsonl="${fixture_root}/benchmark.jsonl"

write_pass_fixture() {
  : >"${jsonl}"
  for arm in pp128 pp512 tg128_pp512; do
    case "${arm}" in
      pp128)
        prompt_tokens=128
        prefill_ratio=1.12
        generation_ratio=1.02
        graph_replayed=false
        ;;
      pp512)
        prompt_tokens=512
        prefill_ratio=1.01
        generation_ratio=1.03
        graph_replayed=false
        ;;
      tg128_pp512)
        prompt_tokens=512
        prefill_ratio=1.01
        generation_ratio=1.04
        graph_replayed=true
        ;;
    esac
    jq -nc \
      --arg arm "${arm}" \
      --argjson prompt_tokens "${prompt_tokens}" \
      --argjson prefill_ratio "${prefill_ratio}" \
      --argjson generation_ratio "${generation_ratio}" \
      --argjson graph_replayed "${graph_replayed}" '
        {
          schema_version:2,
          arm:$arm,
          prompt_tokens:$prompt_tokens,
          generated_tokens:128,
          warmup_iterations:5,
          measured_iterations:20,
          provenance:{
            path:"/evidence/qwen35-bf16.provenance.json",
            sha256:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            weight_format:"bf16",
            llama_cpp_revision:"1234567890abcdef1234567890abcdef12345678",
            artifact_sha256:"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            raftinfer_model_sha256:"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            llama_model_sha256:"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            llama_server_sha256:"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            llama_server_version:"llama.cpp build 1234567890ab",
            llama_cpp_revision_verified:true
          },
          parity:{
            records:4,
            generated_tokens_per_record:32,
            exact_matches:128,
            passed:true,
            execution:{
              attention:"online_tiled",
              kv_cache_dtype:"bf16",
              kv_cache_layout:"head-major",
              decode_graph_enabled:true,
              decode_graph_captured:false,
              decode_graph_replayed:false,
              attention_workspace_bytes:0
            }
          },
          execution:{
            attention:"online_tiled",
            kv_cache_dtype:"bf16",
            kv_cache_layout:"head-major",
            decode_graph_captured:$graph_replayed,
            decode_graph_replayed:$graph_replayed
          },
          raftinfer:{
            schema_version:2,
            prompt_tokens:$prompt_tokens,
            generated_tokens:128,
            warmup_iterations:5,
            measured_iterations:20,
            peak_allocated_gpu_bytes:18000000000,
            prefill:{
              min_us:1000,
              median_us:1000,
              mean_us:1000,
              p95_us:1000,
              max_us:1000,
              coefficient_of_variation:0.01,
              tokens_per_second:128000
            },
            generation:{
              min_us:2000,
              median_us:2000,
              mean_us:2000,
              p95_us:2000,
              max_us:2000,
              coefficient_of_variation:0.01,
              tokens_per_second:64000
            },
            execution:{
              attention:"online_tiled",
              kv_cache_dtype:"bf16",
              kv_cache_layout:"head-major",
              decode_graph_captured:$graph_replayed,
              decode_graph_replayed:$graph_replayed
            }
          },
          llama_cpp:{
            schema_version:2,
            prompt_tokens:$prompt_tokens,
            generated_tokens:128,
            warmup_iterations:5,
            measured_iterations:20,
            revision:"1234567890abcdef1234567890abcdef12345678",
            prefill:{
              min_us:1100,
              median_us:1100,
              mean_us:1100,
              p95_us:1100,
              max_us:1100,
              coefficient_of_variation:0.01,
              tokens_per_second:(128000 / $prefill_ratio)
            },
            generation:{
              min_us:2100,
              median_us:2100,
              mean_us:2100,
              p95_us:2100,
              max_us:2100,
              coefficient_of_variation:0.01,
              tokens_per_second:(64000 / $generation_ratio)
            }
          },
          throughput_ratio:{
            prefill:$prefill_ratio,
            generation:$generation_ratio
          },
          performance_floor:1.0,
          performance_floor_passed:true,
          peak_allocated_gpu_bytes:18000000000,
          peak_memory_status:"measured_by_raftinfer_rmm"
        }
      ' >>"${jsonl}"
  done
}

expect_pass() {
  write_pass_fixture
  "${gate}" "${jsonl}" >"${fixture_root}/pass.out"
  grep -F 'qwen35-bf16-gate: pass' "${fixture_root}/pass.out"
}

expect_fail() {
  local name="$1"
  local filter="$2"
  local expected="$3"
  write_pass_fixture
  if [[ "${filter}" == "__NONFINITE_PREFILL__" ]]; then
    awk '
      changed == 0 && /"prefill":1.12/ {
        sub(/"prefill":1.12/, "\"prefill\":NaN")
        changed = 1
      }
      { print }
    ' "${jsonl}" >"${fixture_root}/${name}.jsonl"
  else
    jq -c "${filter}" "${jsonl}" >"${fixture_root}/${name}.jsonl"
  fi
  set +e
  "${gate}" "${fixture_root}/${name}.jsonl" \
    >"${fixture_root}/${name}.out" \
    2>"${fixture_root}/${name}.err"
  local status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    printf 'expected failure for %s\n' "${name}" >&2
    exit 1
  fi
  grep -F "${expected}" "${fixture_root}/${name}.err"
}

expect_pass
expect_fail pp128_slow \
  'if .arm == "pp128" then .throughput_ratio.prefill = 0.99 else . end' \
  'pp128 prefill ratio'
expect_fail pp512_slow \
  'if .arm == "pp512" then .throughput_ratio.prefill = 0.99 else . end' \
  'pp512 prefill ratio'
expect_fail tg128_slow \
  'if .arm == "tg128_pp512" then .throughput_ratio.generation = 0.99 else . end' \
  'tg128_pp512 generation ratio'
expect_fail no_110 \
  '.throughput_ratio.prefill = 1.01 | .throughput_ratio.generation = 1.01' \
  'at least one gated ratio'
expect_fail wrong_weight \
  'if .arm == "pp128" then .provenance.weight_format = "f16" else . end' \
  'weight format'
expect_fail missing_weight \
  'if .arm == "pp128" then del(.provenance.weight_format) else . end' \
  'weight format'
expect_fail non_online_attention \
  'if .arm == "pp512" then .execution.attention = "materialized_reference" else . end' \
  'online_tiled'
expect_fail missing_kv_dtype \
  'if .arm == "pp512" then del(.execution.kv_cache_dtype) else . end' \
  'kv_cache_dtype'
expect_fail missing_kv_layout \
  'if .arm == "pp512" then del(.execution.kv_cache_layout) else . end' \
  'kv_cache_layout'
expect_fail missing_graph_replay \
  'if .arm == "tg128_pp512" then .execution.decode_graph_replayed = false else . end' \
  'decode graph replay'
expect_fail bad_parity \
  'if .arm == "pp128" then .parity.exact_matches = 127 else . end' \
  'exact parity'
expect_fail missing_parity_execution \
  'if .arm == "pp128" then del(.parity.execution) else . end' \
  'parity execution'
expect_fail parity_execution_mismatch \
  'if .arm == "pp128" then .parity.execution.kv_cache_layout = "token-major" else . end' \
  'parity execution'
expect_fail missing_llama_revision \
  'if .arm == "pp128" then .provenance.llama_cpp_revision = "" else . end' \
  'llama.cpp revision'
expect_fail model_hash_mismatch \
  'if .arm == "pp128" then .provenance.llama_model_sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" else . end' \
  'model SHA256'
expect_fail llama_version_unverified \
  'if .arm == "pp128" then .provenance.llama_cpp_revision_verified = false else . end' \
  'llama-server version'
expect_fail few_warmups \
  'if .arm == "pp128" then .warmup_iterations = 2 else . end' \
  'warmups'
expect_fail few_measurements \
  'if .arm == "pp128" then .measured_iterations = 6 else . end' \
  'measurements'
expect_fail raftinfer_unstable \
  'if .arm == "pp512" then .raftinfer.prefill.coefficient_of_variation = 0.031 else . end' \
  'RAFTINFER coefficient of variation'
expect_fail llama_unstable \
  'if .arm == "pp512" then .llama_cpp.prefill.coefficient_of_variation = 0.031 else . end' \
  'llama coefficient of variation'
expect_fail missing_arm \
  'select(.arm != "pp512")' \
  'exactly one record'
expect_fail duplicate_arm \
  '. as $record | ., (select(.arm == "pp128") | $record)' \
  'exactly one record'
expect_fail unknown_arm \
  'if .arm == "pp128" then .arm = "pp256" else . end' \
  'unknown arm'
expect_fail wrong_prompt_shape \
  'if .arm == "pp128" then .prompt_tokens = 512 else . end' \
  'arm shape mapping'
expect_fail nonfinite_metric \
  '__NONFINITE_PREFILL__' \
  'finite positive number'
expect_fail zero_metric \
  'if .arm == "pp128" then .throughput_ratio.prefill = 0 else . end' \
  'finite positive number'
expect_fail wrong_type_metric \
  'if .arm == "pp128" then .throughput_ratio.prefill = "1.2" else . end' \
  'finite positive number'

printf 'bf16-gate-script-test: pass\n'
