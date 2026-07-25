# M2 Qwen3.5-9B Verification

Date: 2026-07-25

## Status

M2 source implementation and host verification are present, but M2 is **not
accepted**. The target host still has an older source tree, so the new CUDA
executor and Rust/CUDA link path have not been compiled or executed there.
Exact real-model parity, performance, and peak-memory evidence are also
outstanding.

| Gate | Current evidence | Status |
|---|---|---|
| GGUF/config/manifest validation | Host CTest and Rust tests | Pass |
| CPU FP32 hybrid executor | Deterministic nonzero full-executor reference test | Pass |
| Rust tokenizer/generation/benchmark behavior | Rust unit and integration tests | Pass |
| Script control flow and failure handling | Controlled fake-dependency tests | Pass |
| CUDA 13 / `sm_120a` compilation | Current source not synchronized to target | Pending |
| BF16/F16 CUDA operators vs CPU FP32 | Test source exists; target execution pending | Pending |
| CUDA full executor vs CPU reference | Test source exists; target execution pending | Pending |
| Exact prompt and greedy token parity | Harness exists; pinned artifact/reference run pending | Pending |
| PP128/PP512 + TG128 performance | Harness exists; real run pending | Pending |
| Peak allocated GPU bytes | RMM-backed C ABI/CLI metric and staged CUDA assertions implemented; real run pending | Pending |
| Reproducible model/converter/reference provenance | Existing GGUF revisions incomplete | Pending |

## Fresh host evidence

The following checks passed on 2026-07-25 with AppleClang
`21.0.0.21000101`, CUDA disabled:

```bash
scripts/local-check.sh
```

Result:

- 17/17 CTest tests passed;
- native library mode test built and installed a real shared library and
  rejected an invalid library type;
- parity harness test covered four passing records, a value mismatch, and a
  shortened token array;
- benchmark harness test covered a passing report, the 0.8x performance-floor
  failure, and parity-before-benchmark refusal;
- GGUF preparation test covered a complete provenance record and revision
  mismatch refusal;
- 3/3 CLI unit tests, 14/14 CLI integration tests, 14/14 Rust runtime
  integration tests, and 18/18 tokenizer tests passed;
- Rust formatting and documentation tests passed.

Additional checks:

```bash
cargo clippy --workspace --all-targets -- -D warnings
bash -n scripts/*.sh tests/*-test.sh
git diff --check
```

All passed.

Host evidence proves parsing, validation, reference semantics, API ownership,
argument validation, output schemas, and orchestration failure behavior. It
does not prove CUDA compilation, GPU numerical correctness, real-model token
parity, or performance.

## Correctness contract

M2 uses layered correctness:

1. GGUF structure, tensor manifest, hybrid 3-linear/1-full block ordering, and
   tokenizer metadata must validate before publishing a model handle.
2. BF16/F16 CUDA operator and executor results must satisfy
   `abs <= 2e-2 || rel <= 2e-2` against independent CPU FP32 references.
3. Prefill, decode, continued state, reset, tail dimensions, and invalid shapes
   must pass on the target.
4. The four-prompt corpus must match the pinned llama.cpp reference exactly for
   every prompt token ID and every greedy generated token ID.

The CUDA executor test also checks that repeated prefill/decode operations do
not add RMM allocations after session construction.

## Parity contract

`scripts/qwen35-parity.sh` runs llama-server and BRT sequentially under one
cooperative GPU lock. The corpus covers English factual text, English Rust
code, Simplified Chinese, and mixed Chinese/English punctuation.

The reference path obtains the rendered chat prompt, raw prompt token IDs, and
raw generated token IDs from llama-server. BRT emits the same fields through
`brt-cli generate --output-format json`. The first mismatch records its kind,
index, expected token, and actual token; a missing token is represented as
JSON `null`. Block/logit diagnostics remain unavailable through the current
stable C ABI and are reported explicitly rather than fabricated.

## Performance contract

`scripts/qwen35-benchmark.sh` refuses to run unless the parity report is fully
passing. It then measures BRT and llama.cpp with identical token arrays:

- prompt processing at 128 and 512 tokens;
- generation of 128 tokens;
- 5 warmup iterations and 20 measured iterations;
- min, median, nearest-rank p95, max, and tokens/second;
- GPU model, driver, SM/memory clocks, CUDA, RAFT, and RMM versions;
- peak BRT logical GPU bytes allocated through the RMM pool;
- BRT/llama.cpp throughput ratios with a required floor of 0.8 for both phases.

The BRT CLI holds one model and session per benchmark arm so loading is excluded
from measured latency. The harness runs both implementations sequentially and
re-runs `scripts/gpu-preflight.sh` before GPU work.

The CUDA DeviceContext wraps its RMM pool with an allocation statistics
resource. All BRT model weights, persistent state, and workspaces use that
resource. The stable C ABI exposes the observed high-water mark, the Rust CLI
emits it as `peak_allocated_gpu_bytes`, and the comparison report labels it
`peak_memory_status:"measured_by_brt_rmm"`. This is BRT logical allocation
through the pool, not the pool's upstream reserved capacity. Counter work
occurs only on allocation/deallocation, outside the kernel execution loop.
The CUDA-gated C ABI test records the engine baseline, then requires the peak
to increase after model-weight upload and again after session construction; it
also requires prefill/decode to leave that peak unchanged. This assertion has
not yet run on the target.

## Target validation still required

After legitimate source synchronization to `/home/charles/brt-workspace`, run
inside the CUDA 13 / RAFT-RMM 26.06 development environment:

```bash
cmake -S . -B build/cuda -G Ninja \
  -DBRT_ENABLE_CUDA=ON \
  -DBRT_BUILD_TESTS=ON
cmake --build build/cuda
scripts/gpu-preflight.sh
ctest --test-dir build/cuda --output-on-failure
BRT_ENABLE_CUDA=ON cargo build --release -p brt-cli
```

Before each GPU run, `scripts/gpu-preflight.sh` must pass. If another compute
process is present, stop this project's validation; do not terminate or
interfere with the other process.

Final M2 acceptance additionally requires:

- a newly prepared BF16 GGUF with complete provenance from
  `scripts/prepare-qwen35-gguf.sh`;
- a pinned llama.cpp reference checkout;
- a fully passing `scripts/qwen35-parity.sh` report;
- a fully passing `scripts/qwen35-benchmark.sh` report;
- a nonzero real `peak_allocated_gpu_bytes` value from each benchmark arm.
