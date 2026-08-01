#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docs="${repo_root}/docs/environment.md"
test -f "${docs}" || {
  printf 'script-env-docs: missing docs/environment.md\n' >&2
  exit 1
}

allowlisted_system_variables=(
  BASH_SOURCE HOME LD_LIBRARY_PATH PATH TMPDIR XDG_CACHE_HOME
)
is_allowlisted_system_variable() {
  local variable
  for variable in "${allowlisted_system_variables[@]}"; do
    [[ "$1" == "${variable}" ]] && return 0
  done
  return 1
}

missing=0
while IFS= read -r variable; do
  is_allowlisted_system_variable "${variable}" && continue
  if ! grep -Fq -- "${variable}" "${docs}"; then
    printf 'script-env-docs: undocumented public variable %s\n' "${variable}" >&2
    missing=1
  fi
done < <(
  {
    rg -o --no-filename '\$\{?[A-Z][A-Z0-9_]*' "${repo_root}/scripts" --glob '*.sh' |
      sed -E 's/^\$\{?//'
    rg -o --no-filename 'RAFTINFER_[A-Z0-9_]+' "${repo_root}/scripts" --glob '*.sh'
  } | sort -u
)
(( missing == 0 ))
