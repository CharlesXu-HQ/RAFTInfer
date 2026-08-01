# Task 9 report

Status: local implementation complete; target CUDA validation delegated to
controller

Base: `adc06d8`

## Implementation

- Kept the default policy as online F32 token-major.
- Added explicit online KV cache indexing for both supported layouts:
  - token-major: `[token][kv_head][head_dim]`
  - head-major: `[kv_head][token][head_dim]`
- Routed the selected `Qwen35KvCacheLayout` through public attention dispatch,
  direct online prefill/decode entry points, and executor graph/direct decode.
- Preserved two contiguous K/V planes for both layouts.
- Preserved FP32 online max, denominator, score, and output accumulation.
- Kept materialized reference attention restricted to F32 token-major.
- Added BF16 cache append/load support with round-to-nearest conversion and an
  aligned `__nv_bfloat162` append path where adjacent elements are available.
- Rejected unknown dtype/layout values through policy and direct-entry
  signature validation instead of treating them as token-major.

## Tests

- Replaced head-major fallback/unsupported assertions with behavioral coverage
  requiring all four online cache policies to remain online:
  - F32 token-major
  - F32 head-major
  - BF16 token-major
  - BF16 head-major
- Extended attention coverage across activation dtype, cache dtype, and layout
  for prefill lengths `17` and `128`, continued prefill with nonzero
  `past_tokens`, decode contexts `32`, `128`, and `512`, and direct
  host/device-position decode append.
- Added independent test-side layout indexing for cache placement checks across
  both K/V planes, comparing BF16 placement against BF16 roundtrip values.
- Extended executor coverage so diagnostics must report requested dtype/layout
  without fallback, including prefill, decode, reset/replay of the same
  sequence, greedy argmax parity against the materialized reference fixture,
  graph capture/replay diagnostics for the release shape, and zero measured RMM
  allocation on repeated execution.

## TDD evidence

- RED was represented by tests first: attention tests call the direct online
  decode API with an explicit layout parameter and executor tests require
  head-major policies to stay online rather than falling back.
- Local CUDA RED/GREEN compile/run could not be collected on this Mac worktree
  because `cmake -S . -B build/cuda -G Ninja -DRAFTINFER_ENABLE_CUDA=ON` fails before
  compiling tests with `Failed to find nvcc`.

## Local verification

- `git diff --check` — passed.
- `scripts/local-check.sh` — passed:
  - host CTest: 18/18 passed
  - native library type/parity/benchmark/prepare script fixtures passed
  - Rust tests passed

## Remaining controller-owned validation

- CUDA 13.2 `sm_120a` compile.
- Focused attention, executor, and graph GPU tests.
- Four-prompt, 32-token exact Qwen3.5-9B greedy parity for all four online
  cache policies.
- PP512 and TG128@PP512 A/B for all passing cache modes.
- Candidate promotion decision. This commit intentionally does not promote
  BF16 or head-major defaults.
