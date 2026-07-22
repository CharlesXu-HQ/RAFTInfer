# Task 3 evidence report

## Scope delivered

- Added the Cargo workspace with `brt-sys`, `brt-runtime`, and `brt-cli`.
- Built the native static library through CMake in host mode, including the static-library install rule required by Cargo's CMake build helper.
- Added raw opaque C ABI declarations, including `brt_last_error_message` to cover the full exported ABI.
- Added the safe RAII `Engine`, `EngineConfig`, and displayable native `Error` wrapper.  The wrapper accepts only coarse configuration and an opaque engine handle; it neither accepts GPU pointers nor exposes native resource ownership.
- Added the documented `info` CLI command.

## TDD evidence

1. RED: Added `rust/brt-runtime/tests/engine.rs` and crate manifests before implementation. `cargo test -p brt-runtime` failed because `brt-sys` had no source target, as expected.
2. GREEN: Added the raw binding, CMake build script, wrapper, and CLI. The two engine lifecycle/error tests passed.
3. RED: The review-fix test required exact `info` output and rejection of undocumented `host-info`; it failed while the alias still succeeded.
4. GREEN: Removed the alias; the focused CLI tests passed.

## Final verification

All commands were run from the worktree:

```text
cargo fmt --check                                      PASS
cargo test --workspace                                 PASS (4 integration tests)
cargo run -p brt-cli -- info                           PASS (backend=host)
cmake --build build/host                               PASS
ctest --test-dir build/host --output-on-failure        PASS (1/1)
```

## Self-review

- `git diff --check` passed.
- Reviewed the complete Task 3 diff: ABI structs match the authoritative C header layouts, failure messages are copied before the native thread-local buffer can change, and `Drop` is the sole native-engine destruction path.
- Cargo initially could not resolve the configured registry inside the sandbox. A scoped approval retry downloaded only the required `cmake` build dependency and its transitive build helpers; no tests were weakened.

## Remaining concerns

The wrapper intentionally mirrors the current coarse C ABI only. It should gain explicit thread-safety guarantees only when the C++ engine's cross-thread lifecycle contract is defined.

## Review Fixes

- Removed the undocumented `host-info` alias; only explicit `info` and the existing default-to-`info` behavior can succeed.
- Replaced alias coverage with an exact-byte assertion for `info` output (`backend=host\n`) and a focused rejection assertion for `host-info`.

### Review-fix verification

```text
cargo fmt --check
  exit 0
cargo test -p brt-cli
  2 passed; 0 failed
cargo test --workspace
  4 integration tests passed; 0 failed
cargo run -p brt-cli -- info
  backend=host
git diff --check
  exit 0
```

### Build-script invalidation fix

- Added `cargo:rerun-if-env-changed=BRT_ENABLE_CUDA` before reading the backend-selection environment variable, so changing host/CUDA selection invalidates Cargo's native build output.
- No standalone source-contract test was added: Cargo evaluates build-script invalidation before compiling or running package tests, so proving it with a Rust test would require a nested Cargo invocation and an isolated target directory. That would test Cargo's cache orchestration rather than this project's runtime behavior and would add disproportionate test machinery. The workspace build/test command recompiled `brt-sys` after the script changed and validated the directive's build-script output path.

```text
cargo fmt --check
  exit 0
cargo test --workspace
  brt-sys rebuilt; 4 integration tests passed; 0 failed
git diff --check
  exit 0
```
