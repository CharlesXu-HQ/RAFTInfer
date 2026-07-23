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
grep -Fxq 'ARG RUSTUP_DIST_SERVER=https://static.rust-lang.org' "${dockerfile}" || {
  echo "Dockerfile.dev must default RUSTUP_DIST_SERVER to the official endpoint" >&2
  exit 1
}
grep -Fxq 'ARG RUSTUP_UPDATE_ROOT=https://static.rust-lang.org/rustup' "${dockerfile}" || {
  echo "Dockerfile.dev must default RUSTUP_UPDATE_ROOT to the official endpoint" >&2
  exit 1
}
grep -Fxq 'RUN export RUSTUP_DIST_SERVER RUSTUP_UPDATE_ROOT \' "${dockerfile}" || {
  echo "Dockerfile.dev must export Rustup endpoints before the curl pipeline" >&2
  exit 1
}
grep -Fq '&& curl --proto' "${dockerfile}" || {
  echo "Dockerfile.dev must keep curl in the exported Rustup pipeline" >&2
  exit 1
}
if grep -Eq '^ENV .*RUSTUP_(DIST_SERVER|UPDATE_ROOT)=' "${dockerfile}"; then
  echo "Dockerfile.dev must not persist Rustup mirror endpoints at runtime" >&2
  exit 1
fi
if grep -Fq 'rsproxy.cn' "${dockerfile}"; then
  echo "Dockerfile.dev must not hardcode a regional Rustup mirror" >&2
  exit 1
fi
