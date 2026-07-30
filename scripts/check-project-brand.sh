#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

required=(
  README.md
  README.zh-CN.md
  CONTRIBUTING.md
  CODE_OF_CONDUCT.md
  SECURITY.md
  CHANGELOG.md
  CITATION.cff
  .github/CODEOWNERS
  .github/workflows/ci.yml
  benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl
  docs/assets/qwen35-bf16-rtx5090.svg
)
for path in "${required[@]}"; do
  test -f "${path}" || {
    printf 'project-brand: missing %s\n' "${path}" >&2
    exit 1
  }
done

legacy_pattern='Blackwell RAFT Runtime|namespace[[:space:]]+brt|include/brt|Brt[A-Z]|brt_[A-Za-z]|(^|[^[:alnum:]_])BRT(_[A-Za-z0-9]+)?([^[:alnum:]_]|$)|brt-(sys|runtime|cli|dev|smoke|qwen35-logits)|(^|[^[:alnum:]_])brt_cpp([^[:alnum:]_]|$)|(^|[^[:alnum:]_])project[[:space:]]*\([[:space:]]*brt[[:space:]]*\)|\.brt([."]|$)|brt_model_sha256'
if git grep -I -nE "${legacy_pattern}" -- \
  . \
  ':!docs/superpowers/specs/2026-07-30-raftinfer-open-source-renaming-design.md' \
  ':!docs/superpowers/plans/2026-07-30-raftinfer-open-source-renaming.md' \
  ':!CHANGELOG.md' \
  ':!scripts/check-project-brand.sh' \
  ':!.superpowers/sdd'; then
  printf 'project-brand: active legacy identifier found\n' >&2
  exit 1
fi

git grep -q 'README.zh-CN.md' -- README.md
git grep -q 'README.md' -- README.zh-CN.md
git grep -q 'schema_version.*2' -- \
  tests benchmarks scripts cpp rust
