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
| Hugging Face checkout path | `/home/charles/brt-artifacts/Qwen3.5-9B` |
| BF16 GGUF path | `/home/charles/brt-artifacts/Qwen3.5-9B-GGUF/Qwen3.5-9B-bf16.gguf` |
| GGUF size | `17920697312` bytes |
| GGUF SHA-256 | `cf362b9cc9f928ff7603c0b254f7ae547fa8ce35833a475782f34820e0f95444` |
| Transformers environment | `/home/charles/brt-tools/qwen35-reference-env` |
| Transformers version | `5.14.1` |
| Transformers revision | `a08ace4bbd97e721c98751deec37d87b026acadc` |

These values establish the identity and integrity of the existing GGUF file,
but they do **not** establish reproducible conversion provenance:

- the target model directory has no authoritative Hugging Face cache/revision
  metadata or `.brt-source-revision` record;
- the source model revision is therefore unresolved;
- the llama.cpp converter checkout/revision and exact conversion command are
  unresolved;
- no pinned llama.cpp reference checkout was found on the target.

Consequently, this existing file may be used for diagnostic loading only after
the usual GPU safety checks. It must not be presented as the reproducible M2
golden artifact, and its unknown revisions must not be guessed or backfilled.
A newly prepared artifact satisfying the script contract is required for final
M2 parity and performance evidence.

## Kernel provenance

The M2 Qwen3.5 CUDA executor uses project-native source. No `bw24` kernel was
imported because the audited material contained no reusable implementation
with existing performance evidence. Future `bw24` imports remain allowed when
the source path, upstream revision, license, local modifications, correctness
evidence, and RTX 50 performance evidence are recorded before reuse.
