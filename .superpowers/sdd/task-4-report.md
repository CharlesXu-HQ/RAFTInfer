# M2 Task 4 Report: Stable session ABI and host ownership

## Outcome

Implemented the coarse Qwen3.5 session ABI across C++ and Rust.

- Added opaque `BrtSessionHandle`, `BrtSessionConfig`, `BrtTokenResult`, and
  `brt_session_create/prefill/decode/reset/destroy`.
- Changed native model ownership from unique handle ownership to stable shared
  model ownership so a C caller may destroy `BrtModelHandle` after successful
  session creation.
- Added `brt::Session`, which constructs `Qwen35HostState` from the validated
  model `Qwen35Config` and requested max context.
- Host-only `prefill` and `decode` validate inputs and return
  `BRT_STATUS_UNAVAILABLE` without mutating token results or session state.
- Added Rust `Session<'model, 'engine>` with
  `PhantomData<&'model Model<'engine>>` plus `Rc<()>` marker for !Send/!Sync.
  `prefill`, `decode`, and `reset` are coarse one-FFI-call methods and require
  `&mut self`.

## RED evidence

- `cmake --build build/host --target brt_c_api_test -j4` failed before
  implementation because `BrtSessionHandle`, `BrtSessionConfig`,
  `BrtTokenResult`, and `brt_session_*` were undeclared.
- `cargo test -p brt-runtime` then failed on an imprecise lifetime in
  `Model::create_session`; fixed by binding
  `create_session<'model>(&'model self) -> Session<'model, 'engine>`.

## GREEN evidence

- `ctest --test-dir build/host -R brt_c_api_test --output-on-failure`
  passed: 1/1.
- `cargo test -p brt-runtime` passed: runtime integration tests 6/6.
- `scripts/local-check.sh` passed:
  - CTest: 15/15.
  - Rust CLI tests: 2/2.
  - Rust runtime tests: 6/6.
  - Rust sys/doc tests: pass.

## Notes / concerns

- No inference fallback was added. Host-only prefill/decode intentionally stop
  at `BRT_STATUS_UNAVAILABLE`; CUDA execution is reserved for the next task.
- Rust integration tests include a small GGUF fixture builder to exercise the
  real model-load/session-create path without depending on C++ test headers.
