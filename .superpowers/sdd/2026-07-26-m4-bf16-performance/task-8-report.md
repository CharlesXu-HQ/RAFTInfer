# Task 8 report

Status: local implementation complete; target CUDA validation delegated to
controller

Base: `575445c`

## Implementation

- Added `GatedDeltaSchedule`, `GatedDeltaLaunchPolicy`, and
  `GatedDeltaScheduleDiagnostic` to the gated-delta public kernel interface.
- Added a policy-aware `qwen35_gated_delta` overload. The existing overload
  remains source-compatible and selects the conservative current schedule.
- Added explicit `sm_120` register-resident recurrent update candidates for
  key/value dimensions `64` and `128`, with four warps per block and
  `__launch_bounds__(128, 2)`.
- Kept the existing project-native `register_recurrent_128_kernel` as the
  current baseline/fallback. No BW24 Gated Delta kernel was imported because
  the controller's reuse scan found no reusable/imported BW24 implementation in
  this repository.
- Added construction-time executor diagnostics for buckets `1`, `128`, and
  `512`. Until target RTX5090 benchmarking proves a candidate is faster and
  correct, these diagnostics intentionally report
  `register_resident_current` with `candidate_accepted=false`.
- Routed executor delta launches through the stored construction-time policy,
  so prefill/decode/graph replay do not perform timing or candidate promotion.

## Tests

- Added CUDA delta correctness coverage for candidate schedules:
  - key/value dimensions `64` and `128`
  - prefill lengths `17`, `128`, and `512`
  - continued prefill
  - eight one-token decode steps
  - hidden output, convolution state, and recurrent state against the CPU FP32
    reference threshold
- Added executor diagnostics assertions proving gated-delta schedule metadata is
  immutable across prefill, decode capture, and graph replay.

## Local verification

- `git diff --check` — passed.
- `scripts/local-check.sh` — passed:
  - host CTest: 18/18 passed
  - native library type/parity/benchmark/prepare script fixtures passed
  - Rust tests passed

## Remaining controller-owned validation

- CUDA 13.2 `sm_120a` compile on the RTX5090 target.
- `brt_qwen35_cuda_delta_test` and `brt_qwen35_executor_test` on target.
- zero decode allocation gate.
- four-prompt, 32-token exact Qwen3.5-9B BF16 greedy parity.
- focused microbenchmark evidence for current vs candidate schedules at buckets
  `1`, `128`, and `512`.

## Notes

- Candidate promotion is deliberately not enabled locally. The selected schedule
  remains `register_resident_current` until target-side microbenchmark evidence
  proves a candidate is meaningfully faster while passing output/state
  correctness.
