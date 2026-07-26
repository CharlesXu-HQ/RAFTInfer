# Qwen3.5-9B Artifact Provenance

## Publication rule

Model weights and converted GGUF artifacts are not committed to this
repository. A locally converted artifact is acceptable as M2 evidence only
when its source model, Transformers implementation, llama.cpp converter, and
llama.cpp reference runtime are all pinned to full revisions and the conversion
record contains the exact command, byte size, and SHA-256.

`scripts/prepare-qwen35-gguf.sh` enforces this contract. It:

- requires a full 40-character model revision and matching local
  `.brt-source-revision` evidence;
- verifies the exact converter and reference checkout revisions;
- records the pinned Transformers revision;
- invokes `convert_hf_to_gguf.py` with `--outtype bf16`;
- refuses to overwrite an existing artifact or provenance record;
- emits a machine-readable provenance record containing all revisions, paths,
  the exact argument vector, size, and SHA-256.

The model repository's license and any redistribution conditions remain
separate from this project's Apache-2.0 source license. They must be reviewed
before publishing or redistributing weights.

## Current target artifact

The following artifact was observed on the RTX 50 validation host:

| Field | Observed value |
|---|---|
| Hugging Face repository | `Qwen/Qwen3.5-9B` |
| Hugging Face checkout path | `/home/charles/brt-artifacts/Qwen3.5-9B-c202236` |
| Hugging Face revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| BF16 GGUF path | `/home/charles/brt-artifacts/Qwen3.5-9B-GGUF/Qwen3.5-9B-c202236-bf16.gguf` |
| GGUF size | `18407321408` bytes |
| GGUF SHA-256 | `5e2d54b1b54df02cf1797e6a5e1465255ed68a9547bfd0ab0bde1357347d65e8` |
| Transformers environment | `/home/charles/brt-tools/qwen35-reference-env` |
| Transformers revision | `a08ace4bbd97e721c98751deec37d87b026acadc` |
| llama.cpp converter/reference path | `/home/charles/brt-tools/llama.cpp-aedb2a5-full` |
| llama.cpp converter/reference revision | `aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3` |
| Provenance record | `/home/charles/brt-artifacts/Qwen3.5-9B-GGUF/Qwen3.5-9B-c202236-bf16.provenance.json` |

The provenance record was generated on `2026-07-25T15:22:53Z`. It pins the
source revision through
`/home/charles/brt-artifacts/Qwen3.5-9B-c202236/.brt-source-revision`, records
the exact `convert_hf_to_gguf.py ... --outtype bf16` argument vector, and pins
the same llama.cpp revision for conversion and reference execution. This is
the reproducible M2 golden artifact used by the accepted parity and performance
reports.

## Kernel provenance

No `bw24` kernel was imported because the audited material contained no
reusable Qwen3.5 implementation with existing performance evidence. The
register-resident gated-delta strategy was independently adapted from the
pinned llama.cpp implementation after profiling showed the project-native
generic kernel was the dominant prefill cost. The BRT code remains native to
this repository and is validated independently against CPU FP32 semantics.
Future `bw24` imports remain allowed when the source path, upstream revision,
license, local modifications, correctness evidence, and RTX 50 performance
evidence are recorded before reuse.
