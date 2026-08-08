# RAFTInfer release-candidate verification

Status: **accepted; split-k-256 is the measured and verified runtime `auto`
default**. Split-k-256 passed the measured promotion gate. An initial
post-commit verification correctly rejected the first runtime integration
because the executor still resolved `auto` to single-block; the executor now
consumes the measured kernel planner default, and a fresh committed-snapshot
default-path run passed every required gate.

## Candidate and target build

- Measured candidate commit: `9a4fd772d89264f6206a2c125b4999db852f50c4`
- Deterministic `git archive` SHA-256:
  `fa743f9abca72dedd9d8de8903f7fa18d267eec7308c062f14a7a2f75bfd2488`
- Local/target synced-content manifest SHA-256:
  `a25662e71862398bfe5c383d878df3613062a8669a6ab672421b62dfb1768f79`
- Target source:
  `/home/charles/raftinfer-split-k-validation/source-9a4fd772d892`
- Target: NVIDIA GeForce RTX 5090 (driver 580.159.03)
- CUDA compiler: 13.2.86; CUDA host compiler: GNU 13.3.0
- RAFT and RMM: 26.06.0
- Native library SHA-256:
  `226bb7a1749a871696546acd7dc6ecf9d36802f620832e5f58aed2ffff31f681`
- Candidate CLI SHA-256:
  `edc661a8f6a373d9371954570ac89b3944fd01f8916102d9b6dcdb4ea1792253`

The admissible clean build is
`/home/charles/raftinfer-split-k-validation/build-9a4fd772d892-gcc13`.
The earlier unsuffixed build is excluded because nvcc selected the known-bad
conda GNU 15.2 host compiler before the host compiler was pinned explicitly.

## Required tests and exact parity

- Complete CUDA/C++ CTest: **25/25 passed**, no skipped or not-run test.
- Retained CTest-log SHA-256:
  `5ebac255dc2dea65ff6ac1ce63835a5472d18b3eb53e818dcd499907f9d7ff20`
- Fresh single-block parity: **4/4 prompts, 128/128 generated token IDs exact**.
- Single-block parity SHA-256:
  `4dacd35aa011d07a470b0d4980c18532a2c2c047a0fa36443ab1f1ddac266350`
- Selected split-k-256 parity: **4/4 prompts, 128/128 generated token IDs exact**.
- Selected parity SHA-256:
  `7f841c6f03aee2be19805b387f9287dc45118dd5389b3c7f54975e69ccdd8fb5`

All accepted runs used online-tiled attention, BF16 head-major KV cache, and
CUDA Graph capture/replay. GPU preflight and compute-process inspection ran
before every measurement. No unrelated compute process appeared and no process
was stopped or signaled.

## Pinned BF16 artifact and llama.cpp

- ModelScope revision:
  `460979c3d11864dd16408d860ac930a360a2fac2`
- BF16 GGUF SHA-256:
  `5e2d54b1b54df02cf1797e6a5e1465255ed68a9547bfd0ab0bde1357347d65e8`
- Schema-v2 provenance SHA-256:
  `a7e1730316d25f8ad878afddbc76d6cb150c3ed191327423efd49b1df687109c`
- llama.cpp revision:
  `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`
- llama-server SHA-256:
  `748f442b1fad03fd50b1896163e6e519a34a850edcb236b3565580a50566199e`

The server reported `version: 1 (aedb2a5)` and the provenance record names the
same revision, model artifact hash, and BF16 output type.

## Deterministic measured selection

Retained measurement root:
`/home/charles/raftinfer-split-k-validation/measurements-9a4fd772d892-20260808-1954-attempt-02`.
The first fresh root is retained separately as a discard: its parity completed,
but a same-second 100% utilization sample caused the wrapper's redundant outer
preflight to refuse before benchmarking. It contains no accepted benchmark.

The accepted same-window baseline SHA-256 is
`ef0f5c1de6cc5f4362d776009f45c60ff7fbcdfa6d025e9b828258129d8b0736`.
Exploratory benchmark hashes are
`e800dc5bf1fe6a8906dd4c79745891c00846714947b348095e4d3e20810b5eb2`
for split-k-256 and
`a44dac0219b89c4b5afc61bb7d7777115865df19e46f3e7208594116ddbbc3a6`
for split-k-512.

The approved score is the minimum of the PP512 and TG128@PP512 generation TPS
ratios against that baseline:

| Candidate | PP512 ratio | TG128@PP512 ratio | Minimum score |
| --- | ---: | ---: | ---: |
| split-k-256 | 1.021936 | 1.022659 | **1.021936** |
| split-k-512 | 0.995458 | 1.003948 | 0.995458 |

The deterministic selection is **split-k-256**. The selection JSON SHA-256 is
`4bb4076ca19784e4b2a01661288d23ac227ae40f1143cfd9f3e9688f50703d6c`.

## Independent formal samples and promotion gate

Formal sample A SHA-256:
`148fa610feb222d2e5ddbab296201bb2ab970a88f9ff8d6bf813a8d5f2e3a548`.
Formal sample B SHA-256:
`9189cd02f0d76c32fbdffab35d8a07f3ddbd90d70eb072bcef0f61ecc817ca6a`.
Both passed exact parity, the BF16 gate, every CV limit, diagnostic disclosure,
the at-least-1% long-context generation gains, and the at-most-1%
non-regression limits. The combined split-K gate passed; its log SHA-256 is
`aaa762f570d4304e9a8dd2355ec5a1644434900706db0a4a192cf0906d7ae631`.

Baseline-relative ratios were:

| Sample | PP128 prefill | PP512 prefill | PP128 generation | PP512 generation | TG128@PP512 generation |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.994037 | 0.996756 | 0.998973 | 1.022100 | 1.015088 |
| B | 0.993584 | 0.995808 | 0.990992 | 1.022284 | 1.023029 |

Per-arm medians (microseconds) and CVs for the two formal samples follow. Each
cell is `RAFTInfer / llama.cpp`.

| Sample / arm | Prefill median | Generation median | Prefill CV | Generation CV |
| --- | ---: | ---: | ---: | ---: |
| A / PP128 | 19,706 / 37,222.5 | 1,463,367 / 1,510,280.5 | 0.001563 / 0.004000 | 0.000079 / 0.000356 |
| A / PP512 | 60,886 / 72,438 | 1,472,801 / 1,511,832 | 0.000906 / 0.002382 | 0.000152 / 0.000227 |
| A / TG128@PP512 | 60,933 / 72,735 | 1,484,026 / 1,512,088 | 0.000397 / 0.002103 | 0.000240 / 0.000337 |
| B / PP128 | 19,715 / 37,120 | 1,475,152.5 / 1,511,138 | 0.000768 / 0.003586 | 0.000082 / 0.000294 |
| B / PP512 | 60,944 / 72,437 | 1,472,536 / 1,512,675 | 0.001111 / 0.002449 | 0.000158 / 0.000405 |
| B / TG128@PP512 | 60,951.5 / 72,687 | 1,472,507 / 1,513,170.5 | 0.000517 / 0.003734 | 0.000067 / 0.000371 |

PP128 correctly resolves to single-block with a 512-token context bucket and
no split-K graph. PP512 and TG128@PP512 resolve to split-K with partition and
threshold 256, context bucket 1,024, fixed attention workspace 264,192 bytes,
and captured split-K graph replay.

## Published representative and decision

The checked-in benchmark is formal sample A. Its SHA-256 is
`148fa610feb222d2e5ddbab296201bb2ab970a88f9ff8d6bf813a8d5f2e3a548`
and its RMM logical allocation peak is 18,315,248,384 bytes. Sample A is the
conservative representative for both promotion-critical long-context arms:
its baseline-relative PP512 and TG128@PP512 gains are lower than sample B's.

Decision: **promote `split_k_256` as `kDefaultOnlineDecodeMode`**. No threshold
was weakened and neither formal sample was substituted. The checked-in chart
compares RAFTInfer with llama.cpp only.

## Runtime `auto` integration correction

Commit `af9b9369a5cd3aa474d874d0eef5738b4fdc3824` changed the kernel planner
default, but its first complete post-commit runtime sample disclosed
single-block for all three benchmark arms. That attempt is rejected and is not
promotion evidence. The root cause was an executor-level `auto` override that
ran before kernel plan construction.

An executor-level CUDA test was added for the supported Qwen3.5 shape at a
4,096-token maximum context. Against the unchanged executor it failed exactly
because constructed diagnostics reported single-block rather than split-K;
the RED CTest log SHA-256 is
`31862feb6b4361c82b54816aa44d25315571f4b772278fae7d9dc169394e5851`.
The minimal correction resolves `auto` through `qwen35_online_decode_plan`, so
the executor has no second hardcoded default. The existing short-context guard
still falls back to single-block when the maximum context is smaller than the
selected partition. A clean target GREEN run passed the focused executor test
1/1; its CTest log SHA-256 is
`d1abe3f957e796c0ee912418c12edc9e449230163716b65bc6200425c1e7d65b`.
Fresh post-commit parity, two independent `auto` samples, and both promotion
gates subsequently passed as recorded below.

## Final committed-snapshot default-path verification

The accepted runtime snapshot is corrective commit
`997ac413caf52bbf97797225974d89ef4280cc25`, built from the exact 193-file
manifest
`daf375022bc1c53c8c04c68e8fa4f8207de4440b1bee71c2a4804c4a5b714b30`
in `/home/charles/raftinfer-split-k-validation/source-997ac413caf5-final-attempt-02`.
Its clean Release build used GNU 13.3.0 and CUDA 13.2.86 and passed the complete
CUDA CTest suite 25/25. The CTest-log SHA-256 is
`18c7c94b1f4e8cf47c9e2c1e9739cf4c3d4b12ff2a0bccb2e82877e8b3f623a0`.
The fresh native-library and CLI SHA-256 identities are
`cf8dab5ec9574e00210f90419f316aacfb60a4f4fd259875397e6c107f5cd65e`
and `edc661a8f6a373d9371954570ac89b3944fd01f8916102d9b6dcdb4ea1792253`.

Fresh forced-single-block and `auto` parity both passed 4/4 prompts and 128/128
exact generated token IDs. Their SHA-256 identities are
`4dacd35aa011d07a470b0d4980c18532a2c2c047a0fa36443ab1f1ddac266350`
and `7f841c6f03aee2be19805b387f9287dc45118dd5389b3c7f54975e69ccdd8fb5`.
The same-window single-block benchmark SHA-256 is
`2dc0abd6717993649c169a415348c3febd93bb4347a5fa46ee8ee80bc08b2332`;
independent `auto` samples A and B are
`a13ea85f9eaa8009b2c8179f7b1a53f36edd855a68b700defb9196004fcc6228`
and `c96458c06ccaad857e5983d32be61680c626d4915faa10c261fa8ce83381e49f`.

Both samples passed the BF16 gate and disclosed the required runtime plan:
PP128 used single-block with partition/threshold zero, bucket 512, and no
split-K graph; both long-context arms used split-K with partition/threshold
256, bucket 1,024, and captured graph replay.
Sample A and B BF16-gate log SHA-256 identities are
`ec7f246e4d2d090f9552bc0a4b9c672aa13d7662e6c7815f1d8672d69d39cffb`
and `862a0ca463afaf5d7164113c45ea9f2c0be489ea3ce324a1dfa60695859e6021`.
The combined split-K gate passed; its log SHA-256 is
`8d2e3a6520cf6e2491f5ee958beca9b94698d6716ce6be1234a3de8f916cdb24`.
The final evidence-hash manifest SHA-256 is
`4031f8cbe379f2730fd59aa344b7325a9b2aa09bac6bc9f9a44b3e8b1ed375f5`.

RAFTInfer per-arm medians, CVs, and baseline-relative ratios were:

| Sample / arm | Prefill median (us) | Generation median (us) | Prefill CV | Generation CV | Prefill ratio | Generation ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A / PP128 | 19,627 | 1,463,567 | 0.000693 | 0.000154 | 0.999847 | 0.999959 |
| A / PP512 | 60,921.5 | 1,472,611 | 0.000452 | 0.000061 | 1.003505 | 1.022828 |
| A / TG128@PP512 | 61,008 | 1,472,559 | 0.000816 | 0.000065 | 0.998984 | 1.022946 |
| B / PP128 | 19,663 | 1,463,870.5 | 0.000904 | 0.000215 | 0.998017 | 0.999752 |
| B / PP512 | 60,926 | 1,472,748.5 | 0.000434 | 0.000065 | 1.003430 | 1.022732 |
| B / TG128@PP512 | 61,253.5 | 1,472,641 | 0.000641 | 0.000055 | 0.994980 | 1.022889 |

The paired llama.cpp medians and CVs from the same samples were:

| Sample / arm | Prefill median (us) | Generation median (us) | Prefill CV | Generation CV |
| --- | ---: | ---: | ---: | ---: |
| A / PP128 | 36,963.5 | 1,512,016 | 0.005262 | 0.000291 |
| A / PP512 | 72,002.5 | 1,513,394.5 | 0.002803 | 0.000394 |
| A / TG128@PP512 | 72,346 | 1,513,419.5 | 0.002062 | 0.000351 |
| B / PP128 | 36,905 | 1,511,462.5 | 0.006034 | 0.000737 |
| B / PP512 | 72,001 | 1,512,891.5 | 0.002136 | 0.000310 |
| B / TG128@PP512 | 72,382 | 1,512,899.5 | 0.002230 | 0.000312 |

Every gain/non-regression and CV threshold passed independently in both
samples. Final postflight showed no compute applications, 32,095 MiB free,
0% utilization, and 40 C. No process was stopped or signaled.

## Scope

This evidence applies only to Qwen3.5-9B BF16 on this one RTX 5090 protocol.
It does not establish performance for other hardware, models, quantizations,
batch sizes, service latency, power, cost, or end-to-end serving workloads.
