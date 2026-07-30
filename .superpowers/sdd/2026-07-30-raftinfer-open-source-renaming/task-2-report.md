# Task 2 — Native RAFTInfer Rename Report

## RED

Before production changes, the updated source-contract tests failed for the
intended legacy production spellings:

- `c_api_source_test.cmake`: legacy `<brt/...>` include in `c_api.cpp`.
- `device_context_source_test.cmake`: legacy `namespace brt` in device code.
- `qwen35_executor_source_test.cmake`: legacy `namespace brt` in executor code.

## GREEN

- Moved the option module to `cmake/RAFTInferOptions.cmake`, public headers to
  `cpp/include/raftinfer`, and the native smoke tool to
  `cpp/tools/raftinfer_smoke.cpp`.
- Renamed project options, targets, tests, C++ namespaces/includes, C ABI
  types/functions/macros, and native artifacts to the RAFTInfer contract.
- Kept C ABI enum values, fields, `struct_size` checks, ownership, and existing
  error behavior unchanged.
- Added install rules verified by a real out-of-tree C consumer using the
  installed `<raftinfer/c_api.h>` and `libraftinfer_cpp`.

## Verification

- Host configure/build with `RAFTINFER_ENABLE_CUDA=OFF` and
  `RAFTINFER_BUILD_TESTS=ON`: pass.
- `ctest --test-dir build/host --output-on-failure`: 18/18 pass.
- Shared host build: produced `libraftinfer_cpp.dylib`.
- Installed static host library, then configured, built, linked, and ran an
  external C consumer successfully with a valid nonzero pool size.
- Scoped legacy production-surface scan: no matches outside the intentional
  negative source-contract assertions.
- `git diff --check`: pass.

## Concerns / handoff

`tests/public-surface-test.sh` currently cannot advance to its intended Rust
package/CLI RED for two Task 1 test-infrastructure defects, neither changed in
this task:

1. `ctest -N | grep -Fq ...` under `pipefail` returns status 141 because
   `grep -q` exits early and CTest receives SIGPIPE, although
   `cpp/CTestTestfile.cmake` contains `raftinfer_c_api_test` and `ctest -N`
   lists 18 tests.
2. Its C consumer sets `initial_pool_bytes = 0` but the preserved C ABI error
   contract rejects zero (`initial_pool_bytes must be non-zero`).  The same
   installed consumer passes with `64 * 1024 * 1024`.

Task 1's owner should correct those test inputs/pipeline mechanics before
rerunning the full public-surface contract.
