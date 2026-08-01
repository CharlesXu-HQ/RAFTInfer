STATUS: DONE_WITH_CONCERNS

## Root Cause

`CudaWeightPlan::upload` previously flattened every Qwen3.5 manifest tensor
through one `checked_type` path. That treated documented F32 auxiliary tensors
as primary CUDA weights and rejected valid Qwen3.5 GGUF layouts where main
matrices are F16/BF16 but norms and DeltaNet auxiliary tensors are F32.

## RED

Intended RED command on CUDA target after syncing only the test/fixture changes:

```bash
cd <repo>
scripts/gpu-preflight.sh
cmake --build build/cuda --target raftinfer_cuda_weights_test
./build/cuda/cpp/raftinfer_cuda_weights_test
```

Expected failure before the production fix: the new mixed F16/BF16 primary plus
F32 auxiliary fixture reaches immutable upload and throws `CudaWeightError`
from the old unsupported-primary-type path for an auxiliary tensor.

Actual RED status: not observed by this agent. SSH to `<target-host>`
is reachable only through password auth in this session, and the sandbox
review rejected non-interactive password SSH to the target. Local CUDA
configuration also fails because `nvcc` is unavailable.

## GREEN

Implemented behavior:

- Manifest-role enumeration marks token embedding, output projection,
  attention/MLP/DeltaNet main matrices as primary.
- Manifest-role enumeration marks final output norm, common input and
  post-attention norms, full-attention Q/K norms, and linear-attention
  convolution, time-step bias, recurrent A, and output norm as auxiliary.
- Primary tensors must be F16/BF16 and must all match the selected primary
  dtype.
- Auxiliary tensors may already match the selected primary dtype, or may be
  GGUF F32. F32 auxiliary payloads are converted once during immutable upload.
- CUDA execution views for converted auxiliary tensors are exposed as the
  selected primary dtype with converted byte size.

Commands run locally:

```bash
git diff --check -- cpp/model/cuda_weights.cu cpp/tests/qwen35_gguf_fixture.hpp cpp/tests/cuda_weights_test.cu
cmake --build build/host
cmake --build build/m2a-release
ctest --test-dir build/host --output-on-failure -R 'raftinfer_(gguf_reader|session|c_api|qwen35_executor_reference)_test'
ctest --test-dir build/m2a-release --output-on-failure -R 'raftinfer_(qwen35_manifest|qwen35_config|qwen35_fixture|qwen35_state|gguf_reader|session|c_api|qwen35_executor_reference)_test'
cmake -S cpp -B /private/tmp/raftinfer-cuda-probe -DRAFTINFER_ENABLE_CUDA=ON -DRAFTINFER_BUILD_TESTS=ON
```

Observed output summary:

- `git diff --check`: passed.
- `cmake --build build/host`: passed.
- `cmake --build build/m2a-release`: passed.
- host CTest subset: 4/4 passed.
- release CTest subset: 8/8 passed.
- CUDA probe: failed because `FindCUDAToolkit` could not find `nvcc`.

## Files

- `cpp/model/cuda_weights.cu`
- `cpp/tests/qwen35_gguf_fixture.hpp`
- `cpp/tests/cuda_weights_test.cu`
- `.superpowers/sdd/2026-07-25-m2-qwen35-end-to-end/task-10-f32-aux-report.md`

## Commit

This commit: `fix: accept Qwen3.5 F32 auxiliary CUDA weights`.

## Self-Review

- Production classification uses manifest role/pointer enumeration, not tensor
  name strings.
- Existing same-dtype exact byte-copy tests are still present and unchanged in
  semantics.
- New mixed fixture computes F32 auxiliary byte sizes and aligned offsets, so it
  is a valid GGUF fixture rather than a descriptor-only fake.
- New tests cover F16-primary/F32-aux and BF16-primary/F32-aux conversion, plus
  real F32 primary rejection.
- No per-token conversion path was added; conversion happens while building the
  immutable upload payload.

## Risks

- Required CUDA RED/GREEN evidence for `raftinfer_cuda_weights_test` is missing from
  this agent because remote password SSH automation was rejected by sandbox
  review and local CUDA tooling is unavailable.
- F16 conversion is implemented in host code for upload-time conversion; CUDA
  target compilation still needs to confirm toolchain compatibility.
