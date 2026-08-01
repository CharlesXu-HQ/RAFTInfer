# Task 3 Report: Rename the Rust workspace, FFI, and CLI

## Scope

- Renamed the Rust workspace directories to `rust/raftinfer-sys`,
  `rust/raftinfer-runtime`, and `rust/raftinfer-cli`.
- Renamed Cargo packages/dependencies to `raftinfer-sys`,
  `raftinfer-runtime`, and `raftinfer-cli`.
- Made `raftinfer` the sole CLI binary target.
- Renamed Rust imports, raw C FFI types/functions/constants, build-script CMake
  options, environment variables, native link name, tests, and user-facing CLI
  diagnostics to RAFTInfer-only names.
- Preserved runtime behavior and the Task 2 native ABI.

## RED

At base commit `57948c662c729ca0faf1fa1757197f8aec32ab07`, before production
changes:

```text
$ tests/public-surface-test.sh
unexpected workspace packages: ['brt-cli', 'brt-runtime', 'brt-sys']
```

The test first configured, built, installed, and consumed the Task 2 RAFTInfer
CMake/C ABI successfully. It then failed for the intended missing Rust public
surface, proving that the existing regression contract detected the package
rename.

## GREEN

After the Rust rename:

```text
$ tests/public-surface-test.sh
exit 0
```

The test observed exactly the three RAFTInfer packages, built the `raftinfer`
binary in an isolated target directory, and verified `raftinfer info` reported
`backend=host`.

`cargo metadata --format-version 1 --no-deps` reported:

```text
raftinfer-sys     library target: raftinfer_sys
raftinfer-runtime library target: raftinfer_runtime
raftinfer-cli     binary target: raftinfer
```

## ABI verification

An independent C/Rust compiler cross-check compared Task 2's native headers
with `raftinfer-sys`. The output matched exactly for size, alignment, every
field offset, and relevant numeric constants for:

- `RaftInferEngineConfig`
- `RaftInferStatus`
- `RaftInferSmokeResult`
- `RaftInferOwnedBuffer`
- `RaftInferQwen35ExecutionPolicy`
- `RaftInferSessionConfig`
- `RaftInferSessionDiagnostics`
- `RaftInferTokenResult`
- status, attention, KV dtype, and KV layout constants

All runtime-created extensible ABI structs initialize `struct_size` with
`size_of` for the matching `raftinfer_sys::RaftInfer*` type.

## Verification

The final gate uses the isolated Cargo target directory
`/private/tmp/raftinfer-task3-target-20260801`.

```text
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
tests/public-surface-test.sh
```

Results:

- formatting check: pass
- workspace all-target tests: 57 passed, 0 failed
- clippy with warnings denied: pass
- public-surface contract: pass
- active legacy Rust/package/FFI/build/CLI identifier scan: no matches

## Concerns

The tokenizer-spec version-1 wire magic remains `BRTTOK\0` in the parser and
its corruption fixtures. This is serialized format compatibility, not an
active Rust package, API, environment, native-library, or CLI alias. Changing
the bytes would be a wire-format migration outside Task 3; Task 4 should decide
whether schema/fixture migration requires a new tokenizer-spec version.

No `brt`/`BRT` compatibility package, import, symbol, environment fallback,
library fallback, CLI binary, or wrapper was retained.
