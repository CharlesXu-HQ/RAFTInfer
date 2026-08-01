#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$(mktemp -d "${TMPDIR:-/tmp}/raftinfer-native-library-type.XXXXXX")"
trap 'rm -rf "${fixture_root}"' EXIT

shared_build="${fixture_root}/shared-build"
shared_install="${fixture_root}/shared-install"

cmake \
  -S "${repo_root}" \
  -B "${shared_build}" \
  -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=OFF \
  -DRAFTINFER_BUILD_TESTS=OFF \
  -DRAFTINFER_NATIVE_LIBRARY_TYPE=SHARED \
  -DCMAKE_INSTALL_PREFIX="${shared_install}"
cmake --build "${shared_build}"
cmake --install "${shared_build}"

case "$(uname -s)" in
  Darwin)
    shared_library="${shared_install}/lib/libraftinfer_cpp.dylib"
    ;;
  Linux)
    shared_library="${shared_install}/lib/libraftinfer_cpp.so"
    ;;
  *)
    printf 'unsupported test host: %s\n' "$(uname -s)" >&2
    exit 1
    ;;
esac

[[ -f "${shared_library}" ]] || {
  printf 'expected shared RAFTINFER library at %s\n' "${shared_library}" >&2
  find "${shared_install}" -maxdepth 3 -type f -print >&2
  exit 1
}

if cmake \
  -S "${repo_root}" \
  -B "${fixture_root}/invalid-build" \
  -G Ninja \
  -DRAFTINFER_ENABLE_CUDA=OFF \
  -DRAFTINFER_BUILD_TESTS=OFF \
  -DRAFTINFER_NATIVE_LIBRARY_TYPE=INVALID; then
  printf 'invalid RAFTINFER_NATIVE_LIBRARY_TYPE was accepted\n' >&2
  exit 1
fi
