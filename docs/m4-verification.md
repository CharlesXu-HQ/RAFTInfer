# M4 BF16 performance verification

## Task 8 base target evidence

- Commit: `b50b370 perf: tune Qwen3.5 Gated DeltaNet schedules`
- Target evidence reported before Fix Round1:
  - CUDA 13.2 `sm_120a` compile: passed.
  - Focused GPU tests: `brt_qwen35_cuda_delta_test`,
    `brt_qwen35_executor_test`, and `brt_qwen35_cuda_graph_test` passed 3/3.

## Task 8 final evidence

Final Task 8 code commit: `ffc99ed fix: respect tuned fixture graph support`.

Target validation on CUDA 13.2 `sm_120a`:

- Compile: PASS.
- Focused GPU CTest: 3/3 PASS.
  - `brt_qwen35_cuda_delta_test`: PASS, 0.51s.
  - `brt_qwen35_executor_test`: PASS, 0.54s.
  - `brt_qwen35_cuda_graph_test`: PASS, 1.71s.
- Independent review: APPROVE.
- Real Qwen3.5-9B parity: 4/4 prompts × 32 generated tokens exact.
  - Evidence:
    `/home/charles/brt-validation/task8-ffc99ed/qwen35-task8-parity.jsonl`
- Post-validation GPU preflight: idle.

Task 8 addresses review feedback that schedule selection must be a real
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
- The 64-dim tuned fixture microbenchmark is a focused kernel/executor
  selection diagnostic, not an end-to-end throughput claim. Real-model
  Qwen3.5-9B uses the production 128-dim path and passed exact greedy-token
  parity as recorded above.

Focused gated-delta microbenchmark evidence:

- Path:
  `/home/charles/brt-validation/task8-ffc99ed/task8-delta-microbench.jsonl`

| Bucket tokens | Key/value dim | Current median ms | Candidate median ms | Accepted | Correctness |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 64 | 0.014368 | 0.008192 | true | true |
| 128 | 64 | 0.607232 | 0.14336 | true | true |
| 512 | 64 | 2.40742 | 0.548864 | true | true |

Local validation on 2026-07-27:

```text
git diff --check
scripts/local-check.sh
```

Result:

- `git diff --check`: passed.
- Host C++ tests: 18/18 passed.
- Rust tests: passed.
- CUDA target testing: passed for the final Task 8 commit `ffc99ed` as recorded
  above.
