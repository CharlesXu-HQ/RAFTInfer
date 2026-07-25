#!/usr/bin/env bash
set -euo pipefail

hf_model_dir="${HF_MODEL_DIR:-}"
hf_model_revision="${HF_MODEL_REVISION:-}"
transformers_revision="${TRANSFORMERS_REVISION:-}"
converter_dir="${LLAMA_CONVERTER_DIR:-}"
converter_revision="${LLAMA_CONVERTER_REVISION:-}"
reference_dir="${LLAMA_REFERENCE_DIR:-}"
reference_revision="${LLAMA_REFERENCE_REVISION:-}"
python_bin="${PYTHON_BIN:-python3}"
output_gguf="${OUTPUT_GGUF:-}"
provenance_output="${PROVENANCE_OUTPUT:-}"
model_revision_file="${HF_MODEL_REVISION_FILE:-${hf_model_dir}/.brt-source-revision}"
jq_bin="${JQ_BIN:-jq}"

fail_usage() {
  printf 'prepare-qwen35-gguf: %s\n' "$1" >&2
  exit 2
}

require_revision() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9a-fA-F]{40}$ ]] ||
    fail_usage "${name} must be a full 40-character Git revision"
}

[[ -n "${hf_model_dir}" && -d "${hf_model_dir}" ]] ||
  fail_usage "HF_MODEL_DIR must name the local Qwen3.5-9B checkpoint directory"
[[ -n "${converter_dir}" && -d "${converter_dir}" ]] ||
  fail_usage "LLAMA_CONVERTER_DIR must name a llama.cpp checkout"
[[ -n "${reference_dir}" && -d "${reference_dir}" ]] ||
  fail_usage "LLAMA_REFERENCE_DIR must name a llama.cpp checkout"
[[ -n "${output_gguf}" ]] || fail_usage "OUTPUT_GGUF is required"
[[ -n "${provenance_output}" ]] || fail_usage "PROVENANCE_OUTPUT is required"
[[ -x "${python_bin}" ]] || fail_usage "PYTHON_BIN is not executable: ${python_bin}"
[[ -f "${converter_dir}/convert_hf_to_gguf.py" ]] ||
  fail_usage "llama.cpp converter is missing: ${converter_dir}/convert_hf_to_gguf.py"
[[ -f "${model_revision_file}" ]] ||
  fail_usage "model revision evidence is missing: ${model_revision_file}"
command -v git >/dev/null || fail_usage "git is required"
command -v "${jq_bin}" >/dev/null || fail_usage "jq is required"

require_revision "HF_MODEL_REVISION" "${hf_model_revision}"
require_revision "TRANSFORMERS_REVISION" "${transformers_revision}"
require_revision "LLAMA_CONVERTER_REVISION" "${converter_revision}"
require_revision "LLAMA_REFERENCE_REVISION" "${reference_revision}"

recorded_model_revision="$(tr -d '[:space:]' <"${model_revision_file}")"
if [[ "${recorded_model_revision}" != "${hf_model_revision}" ]]; then
  printf 'prepare-qwen35-gguf: model revision evidence mismatch: expected=%s actual=%s\n' \
    "${hf_model_revision}" "${recorded_model_revision}" >&2
  exit 3
fi

actual_converter_revision="$(git -C "${converter_dir}" rev-parse HEAD)"
if [[ "${actual_converter_revision}" != "${converter_revision}" ]]; then
  printf 'prepare-qwen35-gguf: converter revision mismatch: expected=%s actual=%s\n' \
    "${converter_revision}" "${actual_converter_revision}" >&2
  exit 3
fi

actual_reference_revision="$(git -C "${reference_dir}" rev-parse HEAD)"
if [[ "${actual_reference_revision}" != "${reference_revision}" ]]; then
  printf 'prepare-qwen35-gguf: reference revision mismatch: expected=%s actual=%s\n' \
    "${reference_revision}" "${actual_reference_revision}" >&2
  exit 3
fi

if [[ -e "${output_gguf}" ]]; then
  fail_usage "OUTPUT_GGUF already exists: ${output_gguf}"
fi
if [[ -e "${provenance_output}" ]]; then
  fail_usage "PROVENANCE_OUTPUT already exists: ${provenance_output}"
fi

mkdir -p "$(dirname "${output_gguf}")" "$(dirname "${provenance_output}")"
conversion_succeeded=0
cleanup_partial() {
  if [[ "${conversion_succeeded}" -ne 1 ]]; then
    rm -f "${output_gguf}"
  fi
}
trap cleanup_partial EXIT

conversion_command=(
  "${python_bin}"
  "${converter_dir}/convert_hf_to_gguf.py"
  "${hf_model_dir}"
  --outfile
  "${output_gguf}"
  --outtype
  bf16
)
"${conversion_command[@]}"
[[ -s "${output_gguf}" ]] ||
  fail_usage "converter did not produce a non-empty GGUF: ${output_gguf}"

artifact_size="$(wc -c <"${output_gguf}" | tr -d ' ')"
if command -v sha256sum >/dev/null; then
  artifact_sha256="$(sha256sum "${output_gguf}" | awk '{print $1}')"
elif command -v shasum >/dev/null; then
  artifact_sha256="$(shasum -a 256 "${output_gguf}" | awk '{print $1}')"
else
  fail_usage "sha256sum or shasum is required"
fi
created_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
conversion_command_json="$(printf '%s\n' "${conversion_command[@]}" |
  "${jq_bin}" -Rsc 'split("\n")[:-1]')"

"${jq_bin}" -n \
  --arg created_utc "${created_utc}" \
  --arg model_path "${hf_model_dir}" \
  --arg model_revision "${hf_model_revision}" \
  --arg model_revision_evidence "${model_revision_file}" \
  --arg transformers_revision "${transformers_revision}" \
  --arg converter_path "${converter_dir}" \
  --arg converter_revision "${converter_revision}" \
  --arg reference_path "${reference_dir}" \
  --arg reference_revision "${reference_revision}" \
  --arg output_path "${output_gguf}" \
  --arg sha256 "${artifact_sha256}" \
  --argjson size_bytes "${artifact_size}" \
  --argjson command "${conversion_command_json}" '
    {
      schema_version:1,
      created_utc:$created_utc,
      model:{
        repository:"Qwen/Qwen3.5-9B",
        local_path:$model_path,
        revision:$model_revision,
        revision_evidence:$model_revision_evidence
      },
      transformers_revision:$transformers_revision,
      llama_cpp:{
        converter_path:$converter_path,
        converter_revision:$converter_revision,
        reference_path:$reference_path,
        reference_revision:$reference_revision
      },
      conversion:{
        outtype:"bf16",
        command:$command
      },
      artifact:{
        path:$output_path,
        size_bytes:$size_bytes,
        sha256:$sha256
      }
    }
  ' >"${provenance_output}"

conversion_succeeded=1
printf 'prepare-qwen35-gguf: wrote %s bytes=%s sha256=%s provenance=%s\n' \
  "${output_gguf}" "${artifact_size}" "${artifact_sha256}" \
  "${provenance_output}"
