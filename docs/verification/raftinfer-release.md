# RAFTInfer release-candidate verification

Status: **accepted**. This record captures the verified CUDA release build,
artifact recovery, parity, and conservative RTX 5090 BF16 benchmark evidence.

## Candidate and target build

- Candidate commit: `57a24c615debb59cfc7d3733bd185d2b1e22a214`
- Source archive SHA-256:
  `e6707be4621e9af250ce8b234b998ab798f45f6057d639817e1cef2b9a21e53b`
- Target: NVIDIA GeForce RTX 5090 (driver 580.159.03)
- CUDA toolkit: 13.2.86
- RAFT and RMM: 26.06.0
- Native library SHA-256:
  `1ba774227981b0f201530e0c282b22b347e20c4fb03d0aba21e9660d25a97381`
- Release CLI SHA-256:
  `d0ca538152f2acde5a21dff901cc838688feb8355c3bb5ece4b6888e17776486`

The clean CUDA Release build produced a shared native library. Dynamic-loader
verification established that the release CLI resolved that exact library with
the intended CUDA and RMM runtime libraries.

## Required tests and exact parity

- CUDA/C++ CTest: **25/25 passed**, with no skip or not-run markers.
- Retained CTest-log SHA-256:
  `f3c17027246c924348b9afd2b36152170a95608a242db686a1abe6c76af0593a`
- Greedy parity: **4/4 prompts, 128/128 generated token IDs exact**.
- Parity-report SHA-256:
  `86ed0e19fd3f3aa24c03478aa2c658765304e2b34b53b6d22065f373f4d64729`
- Execution policy: online-tiled attention, BF16 head-major KV cache, and CUDA
  Graph decode capture/replay.

## Recovered, pinned BF16 artifact

The selected ModelScope transport revision is
`460979c3d11864dd16408d860ac930a360a2fac2`; its safetensors payloads were
verified against that revision's LFS pointers. Conversion used Transformers
`a08ace4bbd97e721c98751deec37d87b026acadc` and the llama.cpp converter and
reference revision `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`.

The converter's pinned metadata override regenerated a schema-v2 BF16 GGUF
whose SHA-256 exactly equals the historical formal artifact:
`5e2d54b1b54df02cf1797e6a5e1465255ed68a9547bfd0ab0bde1357347d65e8`.
The schema-v2 provenance SHA-256 is
`a7e1730316d25f8ad878afddbc76d6cb150c3ed191327423efd49b1df687109c`.
This proves content equality while retaining current, pinned provenance.

## Accepted performance evidence

Each run used a fresh shared-GPU preflight and cooperative lock. Two idle
snapshots preceded each run; no unrelated GPU compute application was present.
Both runs completed five warmups and 20 measured iterations for every arm,
passed the BF16 gate, and kept every reported implementation/phase CV at or
below 3%.

The checked-in conservative representative is SHA-256
`a605359f65829e071c6c11f50a821c1f96d685ab9bac1ea45e03091cad5cc83c`.
It was selected because all five ratios are no higher than the independent
confirmation sample. Its RMM logical allocation peak was 18,314,984,192 bytes.

| Arm | Prefill ratio | Generation ratio | RAFTInfer prefill / generation CV | llama.cpp prefill / generation CV |
| --- | ---: | ---: | ---: | ---: |
| PP128 | 1.895845 | 1.031446 | 0.000815 / 0.000169 | 0.005188 / 0.000310 |
| PP512 | 1.193362 | 1.003922 | 0.001357 / 0.000196 | 0.002789 / 0.000417 |
| TG128@PP512 | 1.197573 | 1.001113 | 0.000557 / 0.000100 | 0.002885 / 0.000355 |

The independent confirmation report also passed the BF16 gate (exit 0), with
SHA-256 `0a0330953609dde7b08d9f8c05c771010d39063362ac41d9ca91e2e0bd1a710e`.
Its ratios were 1.905741/1.031482 (PP128), 1.195484/1.004175 (PP512), and
1.198738/1.003902 (TG128@PP512), prefill/generation respectively. The matching
passing sample confirms that the committed result is stable rather than a
single favorable measurement.

## Scope

This evidence applies only to Qwen3.5-9B BF16 on this one RTX 5090 protocol.
It does not establish performance for other hardware, models, quantizations,
batch sizes, service latency, power, cost, or end-to-end serving workloads.
