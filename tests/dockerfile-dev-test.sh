#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="${repo_root}/containers/Dockerfile.dev"

grep -Eq 'apt-get install -y --no-install-recommends.*cmake|cmake.*apt-get install -y --no-install-recommends' "${dockerfile}" || {
  echo "Dockerfile.dev must install cmake" >&2
  exit 1
}
grep -Fq 'dpkg --compare-versions' "${dockerfile}" || {
  echo "Dockerfile.dev must enforce the required CMake version" >&2
  exit 1
}
grep -Fq '3.26.4' "${dockerfile}" || {
  echo "Dockerfile.dev must enforce CMake >= 3.26.4" >&2
  exit 1
}
