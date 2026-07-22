# Task 3 evidence report

## Scope delivered

- Added the Cargo workspace with `brt-sys`, `brt-runtime`, and `brt-cli`.
- Built the native static library through CMake in host mode, including the static-library install rule required by Cargo's CMake build helper.
- Added raw opaque C ABI declarations, including `brt_last_error_message` to cover the full exported ABI.
- Added the safe RAII `Engine`, `EngineConfig`, and displayable native `Error` wrapper.  The wrapper accepts only coarse configuration and an opaque engine handle; it neither accepts GPU pointers nor exposes native resource ownership.
- Added the `info` CLI command and a tested `host-info` compatibility alias required by the host validation command.

## TDD evidence

1. RED: Added `rust/brt-runtime/tests/engine.rs` and crate manifests before implementation. `cargo test -p brt-runtime` failed because `brt-sys` had no source target, as expected.
2. GREEN: Added the raw binding, CMake build script, wrapper, and CLI. The two engine lifecycle/error tests passed.
3. RED: Added the host CLI integration test. `cargo test -p brt-cli --test cli` failed because `host-info` was unknown.
4. GREEN: Accepted `host-info` as an alias for `info`; the CLI integration test passed.

## Final verification

All commands were run from the worktree:

```text
cargo fmt --check                                      PASS
cargo test --workspace                                 PASS (3 integration tests)
cargo run -p brt-cli -- info                           PASS (backend=host)
cargo run -p brt-cli -- host-info                      PASS (backend=host)
cmake --build build/host                               PASS
ctest --test-dir build/host --output-on-failure        PASS (1/1)
```

## Self-review

- `git diff --check` passed.
- Reviewed the complete Task 3 diff: ABI structs match the authoritative C header layouts, failure messages are copied before the native thread-local buffer can change, and `Drop` is the sole native-engine destruction path.
- Cargo initially could not resolve the configured registry inside the sandbox. A scoped approval retry downloaded only the required `cmake` build dependency and its transitive build helpers; no tests were weakened.

## Remaining concerns

The wrapper intentionally mirrors the current coarse C ABI only. It should gain explicit thread-safety guarantees only when the C++ engine's cross-thread lifecycle contract is defined.
