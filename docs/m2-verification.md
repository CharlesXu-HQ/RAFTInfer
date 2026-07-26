# M2 Qwen3.5-9B Verification

Date: 2026-07-26

## Status

M2 is **accepted** for the Qwen3.5-9B dense text path on RTX 50. The complete
CUDA executor and Rust CLI were built on the target, all 23 CUDA-enabled CTest
tests passed, four real-model prompts matched pinned llama.cpp greedy token IDs
exactly, and both PP128/TG128 and PP512/TG128 exceeded the required 0.8
throughput ratio.

| Gate | Current evidence | Status |
|---|---|---|
| GGUF/config/manifest validation | Host CTest and Rust tests | Pass |
| CPU FP32 hybrid executor | Deterministic nonzero full-executor reference test | Pass |
| Rust tokenizer/generation/benchmark behavior | Rust unit and integration tests | Pass |
| Script control flow and failure handling | Controlled fake-dependency tests | Pass |
| CUDA 13 / `sm_120a` compilation | Target CUDA build completed | Pass |
| BF16/F16/F32 CUDA operators vs CPU FP32 | CUDA-enabled operator tests on RTX 5090 | Pass |
| CUDA full executor vs CPU reference | `brt_qwen35_executor_test` on RTX 5090 | Pass |
| Exact prompt and greedy token parity | 4/4 prompts, 32 generated tokens each | Pass |
| PP128/PP512 + TG128 performance | Formal 5-warmup/20-measurement report | Pass |
| Peak allocated GPU bytes | `18583400704`, measured through RMM | Pass |
| Reproducible model/converter/reference provenance | Pinned model, converter, runtime, command, size, SHA-256 | Pass |

## Fresh host evidence

The following checks passed on 2026-07-26 with AppleClang
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
argument validation, output schemas, and orchestration failure behavior. The
target evidence below supplies the CUDA, numerical, parity, and performance
proofs.

## Target CUDA evidence

The final source was built and tested in `brt-dev:26.06-cuda13` on an NVIDIA
GeForce RTX 5090 with driver `580.159.03`, CUDA `13.2`, RAFT `26.06`, and RMM
`26.06`. A fresh `scripts/gpu-preflight.sh` passed immediately before the GPU
run.

```text
100% tests passed, 0 tests failed out of 23
Total Test time (real) = 2.86 sec
```

The gated-delta suite includes the Qwen3.5 model shape (`key_dim=128`,
`value_dim=128`) so the register-resident optimized path is checked against the
independent CPU FP32 reference, not only exercised by the benchmark.

## Correctness contract

M2 uses layered correctness:

1. GGUF structure, tensor manifest, hybrid 3-linear/1-full block ordering, and
   tokenizer metadata must validate before publishing a model handle.
2. CUDA operators with FP32 activations and BF16/F16 primary weights must satisfy
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
JSON `null`. The executor also provides opt-in trace records and a standalone
logits probe for mismatch localization; neither path is enabled in normal
execution or benchmark timing.

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
from measured latency. TG128 means 128 timed `decode` calls after the prompt
prefill; the token selected by prefill seeds the first decode and is not counted
as a timed generation token. The harness runs both implementations sequentially
and re-runs `scripts/gpu-preflight.sh` before GPU work.

The accepted formal report contains:

| Arm | BRT prefill | llama.cpp prefill | Ratio | BRT generation | llama.cpp generation | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| PP128 + TG128 | 6065.34 tok/s | 3435.46 tok/s | 1.7655 | 82.00 tok/s | 84.23 tok/s | 0.9735 |
| PP512 + TG128 | 5469.65 tok/s | 6612.42 tok/s | 0.8272 | 70.60 tok/s | 84.12 tok/s | 0.8393 |

Both arms report `performance_floor_passed:true`. The optimized
`key_dim=128` gated-delta kernel follows the proven llama.cpp strategy of
keeping each recurrent-state column in registers across the token loop. The
decode attention path parallelizes logits by key token and output by
head-dimension tile to avoid under-occupying RTX 50 SMs at `tokens=1`.

The CUDA DeviceContext wraps its RMM pool with an allocation statistics
resource. All BRT model weights, persistent state, and workspaces use that
resource. The stable C ABI exposes the observed high-water mark, the Rust CLI
emits it as `peak_allocated_gpu_bytes`, and the comparison report labels it
`peak_memory_status:"measured_by_brt_rmm"`. This is BRT logical allocation
through the pool, not the pool's upstream reserved capacity. Counter work
occurs only on allocation/deallocation, outside the kernel execution loop.
The CUDA-gated C ABI test records the engine baseline, then requires the peak
to increase after model-weight upload and again after session construction; it
also requires prefill/decode to leave that peak unchanged. The assertion passed
on the target. The measured high-water mark for both benchmark arms is
`18583400704` bytes.

## Reproduction paths

- CUDA build: `/home/charles/brt-builds/m2-352fc42/cuda`
- Rust release build: `/home/charles/brt-builds/m2-352fc42/cargo-final`
- parity evidence:
  `/home/charles/brt-workspace/build/evidence/qwen35-parity.jsonl`
- benchmark evidence:
  `/home/charles/brt-workspace/build/evidence/qwen35-benchmark.jsonl`

Every future GPU rerun still requires a fresh preflight. If another compute
process is present, validation must stop without terminating or interfering
with that process.
