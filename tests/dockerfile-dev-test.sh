#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="${repo_root}/containers/Dockerfile.dev"

grep -Fxq 'ARG CMAKE_PIP_INDEX_URL=https://pypi.org/simple' "${dockerfile}" || {
  echo "Dockerfile.dev must default CMAKE_PIP_INDEX_URL to the official PyPI index" >&2
  exit 1
}
grep -Fq 'RUN /opt/conda/bin/python -m pip install --no-cache-dir --index-url "${CMAKE_PIP_INDEX_URL}" cmake==3.30.4 \' "${dockerfile}" || {
  echo "Dockerfile.dev must install pinned CMake 3.30.4 with the existing conda Python" >&2
  exit 1
}
grep -Eq 'apt-get install -y --no-install-recommends.*g\+\+|g\+\+.*apt-get install -y --no-install-recommends' "${dockerfile}" || {
  echo "Dockerfile.dev must install the minimal g++ host compiler" >&2
  exit 1
}
grep -Fq 'c++ -x c++ - -o /tmp/cxx-smoke' "${dockerfile}" || {
  echo "Dockerfile.dev must compile a C++ host-tool smoke program" >&2
  exit 1
}
grep -Fq '&& /tmp/cxx-smoke' "${dockerfile}" || {
  echo "Dockerfile.dev must execute the C++ host-tool smoke program" >&2
  exit 1
}
grep -Fq 'dpkg --compare-versions "$(cmake --version | sed -n '\''1s/.* //p'\'')" ge 3.30.4' "${dockerfile}" || {
  echo "Dockerfile.dev must fully enforce the RAFT CMake >= 3.30.4 floor" >&2
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
  echo "Dockerfile.dev must export Rustup endpoints before installer download" >&2
  exit 1
}
grep -Fq '&& curl --proto '\''=https'\'' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init' "${dockerfile}" || {
  echo "Dockerfile.dev must download the Rustup installer before execution" >&2
  exit 1
}
grep -Fq '&& sh /tmp/rustup-init -y --profile minimal --default-toolchain 1.96.0' "${dockerfile}" || {
  echo "Dockerfile.dev must execute the downloaded Rustup installer" >&2
  exit 1
}
if grep -Fq '| sh -s' "${dockerfile}"; then
  echo "Dockerfile.dev must not mask curl failure with a shell pipeline" >&2
  exit 1
fi
grep -Fq 'rustc --version' "${dockerfile}" || {
  echo "Dockerfile.dev must verify rustc after installation" >&2
  exit 1
}
grep -Fq '= 1.96.0' "${dockerfile}" || {
  echo "Dockerfile.dev must require rustc version 1.96.0 exactly" >&2
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
