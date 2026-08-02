#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workflow="$repo_root/.github/workflows/ci.yml"

fail() {
  printf 'github workflow security test: %s\n' "$1" >&2
  exit 1
}

test -f "$workflow" || fail "missing workflow: $workflow"

grep -Fqx '    runs-on: ubuntu-24.04' "$workflow" || fail "CI must use ubuntu-24.04"
if grep -Fq 'ubuntu-latest' "$workflow"; then
  fail "CI must not use ubuntu-latest"
fi

grep -Fqx '      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0' "$workflow" || fail "checkout must be pinned to the approved commit"
grep -Fqx '      - uses: dtolnay/rust-toolchain@4cda84d5c5c54efe2404f9d843567869ab1699d4' "$workflow" || fail "Rust toolchain action must be pinned to the approved commit"
grep -Fqx '          toolchain: 1.97.1' "$workflow" || fail "Rust toolchain must be exact"
grep -Fqx '          components: rustfmt, clippy' "$workflow" || fail "Rust lint components must be explicit"

uses_count=$(grep -Ec '^[[:space:]]*-[[:space:]]+uses:' "$workflow" || true)
[[ "$uses_count" -eq 2 ]] || fail "workflow must contain only the approved action uses"

while IFS= read -r action; do
  [[ "$action" =~ ^[^@]+@[0-9a-f]{40}$ ]] || fail "action ref is not an immutable commit: $action"
done < <(sed -nE 's/^[[:space:]]*-[[:space:]]+uses:[[:space:]]*([^[:space:]#]+).*/\1/p' "$workflow")

grep -Fqx 'permissions:' "$workflow" || fail "workflow must declare permissions"
grep -Fqx '  contents: read' "$workflow" || fail "workflow must retain read-only contents permission"
if grep -Eq '^[[:space:]]*[^#[:space:]]+:[[:space:]]*write' "$workflow"; then
  fail "workflow permissions must remain read-only"
fi
