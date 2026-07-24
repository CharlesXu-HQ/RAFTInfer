### Task 4 Report: Independent CPU reference operators

**Status:** complete

**Commit:** final Task 4 `HEAD`, `feat: add CPU operator references`

## Takeover

- Took over after a previous agent had already written and committed Task 4
  code.
- Re-read the task brief, inspected the target files and commit diff, and
  re-ran the required verification instead of assuming the existing commit was
  correct.

## RED

- Added `cpp/tests/reference_operators_test.cpp` and registered
  `brt_reference_operators_test`.
- Observed the required RED failure:
  `fatal error: '../reference/bf16.hpp' file not found`.

## GREEN

- Implemented `brt::reference` BF16 helpers and CPU reference operators:
  `rms_norm`, `rope`, `bf16_linear`, `softmax`, `argmax`, `embedding`,
  `add`, and `swiglu`.
- Kept references independent of CUDA, RAFT, and bw24 runtime dispatch code.

## Tests

- `cmake --build build/host --target brt_reference_operators_test` passed:
  `ninja: no work to do`.
- `ctest --test-dir build/host -R brt_reference_operators_test --output-on-failure`
  passed: 1/1, 0 failed.
- `ctest --test-dir build/host --output-on-failure` passed: 7/7, 0 failed.

## Changes

- `cpp/reference/bf16.hpp`: BF16 storage plus FP32/BF16 round-to-nearest-even
  conversion with `std::bit_cast`.
- `cpp/reference/operators.hpp`: span-based reference operator API with explicit
  dimension structs.
- `cpp/reference/operators.cpp`: CPU scalar reference implementations with
  shape/span validation before output writes.
- `cpp/tests/reference_operators_test.cpp`: exact fixtures, seeded randomized
  coverage, and invalid-input assertions.
- `cpp/CMakeLists.txt`: compiled reference operators into `brt_cpp` and
  registered the reference operator test.

## Semantics Covered

- RMSNorm uses `1 / sqrt(mean(x^2) + epsilon)` and rejects negative epsilon.
- RoPE supports `position`, `base`, `rotary_dim`, and adjacent interleaved pairs;
  odd rotary dimensions throw.
- BF16 conversion uses RNE; BF16 linear accumulates in FP32.
- Softmax subtracts row max and accumulates the denominator in FP64.
- Argmax returns the first index on ties.
- Embedding validates all token IDs before writing output.
- Add is elementwise.
- SwiGLU is `silu(gate) * up`.

## Concerns

- No blocker. The reference API is intentionally minimal and internal to the
  current C++ tree.
