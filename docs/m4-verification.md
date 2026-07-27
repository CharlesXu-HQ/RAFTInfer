# M4 BF16 performance verification

## Task 8 base target evidence

- Commit: `b50b370 perf: tune Qwen3.5 Gated DeltaNet schedules`
- Target evidence reported before Fix Round1:
  - CUDA 13.2 `sm_120a` compile: passed.
  - Focused GPU tests: `brt_qwen35_cuda_delta_test`,
    `brt_qwen35_executor_test`, and `brt_qwen35_cuda_graph_test` passed 3/3.

## Task 8 Fix Round1

Fix Round1 addresses review feedback that schedule selection must be a real
construction-time contract, not static metadata:

- Executor construction benchmarks current and candidate gated-delta schedules
  for buckets `1`, `128`, and `512`.
- Candidate promotion requires correctness agreement against the current path
  for hidden output, convolution state, and recurrent state.
- Diagnostics now expose current median, candidate median, candidate schedule,
  correctness status, accepted/rejected status, and rejection reason.
- Runtime prefill/decode/graph replay use the immutable selected policy; tests
  assert diagnostics are stable after those paths.
- The focused executor test can emit controller-readable JSONL through
  `BRT_QWEN35_DELTA_MICROBENCHMARK_OUTPUT`.
- Unknown gated-delta schedule enum values are rejected explicitly.

Local validation on 2026-07-27:

```text
git diff --check
scripts/local-check.sh
```

Result:

- `git diff --check`: passed.
- Host C++ tests: 18/18 passed.
- Rust tests: passed.
- CUDA target testing: not run locally; target CUDA 13.2/sm_120a compile and
  focused GPU tests are still required for the Fix Round1 commit.
