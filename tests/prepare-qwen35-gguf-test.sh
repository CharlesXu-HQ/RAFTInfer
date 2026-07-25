#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/brt-prepare-qwen35.XXXXXX")"
trap 'rm -rf "${fixture_root}"' EXIT

model_revision='1111111111111111111111111111111111111111'
transformers_revision='2222222222222222222222222222222222222222'
converter_revision='3333333333333333333333333333333333333333'
reference_revision='4444444444444444444444444444444444444444'

model_dir="${fixture_root}/model"
converter_dir="${fixture_root}/converter"
reference_dir="${fixture_root}/reference"
fake_bin="${fixture_root}/bin"
mkdir -p "${model_dir}" "${converter_dir}" "${reference_dir}" "${fake_bin}"
printf '%s\n' "${model_revision}" >"${model_dir}/.brt-source-revision"
touch "${converter_dir}/convert_hf_to_gguf.py"

cat >"${fake_bin}/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
directory=''
if [[ "$1" == "-C" ]]; then
  directory="$2"
  shift 2
fi
[[ "$1" == "rev-parse" && "$2" == "HEAD" ]]
case "${directory}" in
  *'/converter') printf '%s\n' "${BRT_TEST_CONVERTER_REVISION}" ;;
  *'/reference') printf '%s\n' "${BRT_TEST_REFERENCE_REVISION}" ;;
  *) exit 1 ;;
esac
EOF

cat >"${fake_bin}/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
outfile=''
while [[ "$#" -gt 0 ]]; do
  if [[ "$1" == "--outfile" ]]; then
    outfile="$2"
    shift 2
  else
    shift
  fi
done
printf 'deterministic bf16 gguf\n' >"${outfile}"
EOF

chmod +x "${fake_bin}"/*

output_gguf="${fixture_root}/Qwen3.5-9B-bf16.gguf"
provenance_json="${fixture_root}/provenance.json"
PATH="${fake_bin}:${PATH}" \
  HF_MODEL_DIR="${model_dir}" \
  HF_MODEL_REVISION="${model_revision}" \
  TRANSFORMERS_REVISION="${transformers_revision}" \
  LLAMA_CONVERTER_DIR="${converter_dir}" \
  LLAMA_CONVERTER_REVISION="${converter_revision}" \
  LLAMA_REFERENCE_DIR="${reference_dir}" \
  LLAMA_REFERENCE_REVISION="${reference_revision}" \
  PYTHON_BIN="${fake_bin}/python" \
  OUTPUT_GGUF="${output_gguf}" \
  PROVENANCE_OUTPUT="${provenance_json}" \
  BRT_TEST_CONVERTER_REVISION="${converter_revision}" \
  BRT_TEST_REFERENCE_REVISION="${reference_revision}" \
  "${repo_root}/scripts/prepare-qwen35-gguf.sh"

[[ -f "${output_gguf}" ]]
expected_size="$(wc -c <"${output_gguf}" | tr -d ' ')"
expected_sha="$(shasum -a 256 "${output_gguf}" | awk '{print $1}')"
jq -e \
  --arg model_revision "${model_revision}" \
  --arg transformers_revision "${transformers_revision}" \
  --arg converter_revision "${converter_revision}" \
  --arg reference_revision "${reference_revision}" \
  --arg output_gguf "${output_gguf}" \
  --arg sha "${expected_sha}" \
  --argjson size "${expected_size}" '
    .schema_version == 1 and
    .model.revision == $model_revision and
    .transformers_revision == $transformers_revision and
    .llama_cpp.converter_revision == $converter_revision and
    .llama_cpp.reference_revision == $reference_revision and
    .artifact.path == $output_gguf and
    .artifact.size_bytes == $size and
    .artifact.sha256 == $sha and
    .conversion.outtype == "bf16" and
    (.conversion.command | type == "array" and length == 7)
  ' "${provenance_json}" >/dev/null

printf '%s\n' 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  >"${model_dir}/.brt-source-revision"
set +e
PATH="${fake_bin}:${PATH}" \
  HF_MODEL_DIR="${model_dir}" \
  HF_MODEL_REVISION="${model_revision}" \
  TRANSFORMERS_REVISION="${transformers_revision}" \
  LLAMA_CONVERTER_DIR="${converter_dir}" \
  LLAMA_CONVERTER_REVISION="${converter_revision}" \
  LLAMA_REFERENCE_DIR="${reference_dir}" \
  LLAMA_REFERENCE_REVISION="${reference_revision}" \
  PYTHON_BIN="${fake_bin}/python" \
  OUTPUT_GGUF="${output_gguf}" \
  PROVENANCE_OUTPUT="${provenance_json}" \
  BRT_TEST_CONVERTER_REVISION="${converter_revision}" \
  BRT_TEST_REFERENCE_REVISION="${reference_revision}" \
  "${repo_root}/scripts/prepare-qwen35-gguf.sh" \
  >"${fixture_root}/mismatch-stdout" \
  2>"${fixture_root}/mismatch-stderr"
mismatch_status=$?
set -e

[[ "${mismatch_status}" -eq 3 ]]
grep -F 'model revision evidence mismatch' "${fixture_root}/mismatch-stderr"
