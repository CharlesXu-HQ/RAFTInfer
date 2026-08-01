# Script environment variables

These are the supported environment variables read by the checked-in scripts.
Unset values use the defaults stated below. Test-only harness variables under
tests/ are intentionally not public configuration and are not listed here.
The documentation test excludes the standard process variables BASH_SOURCE,
HOME, LD_LIBRARY_PATH, PATH, TMPDIR, and XDG_CACHE_HOME.

## GPU safety and target synchronization

- RAFTINFER_MIN_FREE_MIB — minimum free GPU memory for preflight; default 2048.
- RAFTINFER_MAX_UTILIZATION_PERCENT — maximum acceptable GPU utilization for
  preflight; default 5.
- NVIDIA_SMI_BIN — nvidia-smi executable used by benchmark metadata collection.
- RAFTINFER_TARGET — SSH destination for sync-target.sh; default
  charles@192.168.124.8.
- RAFTINFER_TARGET_DIR — validated remote staging directory for sync-target.sh;
  default /home/charles/raftinfer-workspace.

## Parity and benchmark inputs

- RAFTINFER_MODEL — required local RAFTInfer GGUF model.
- RAFTINFER_CLI — RAFTInfer CLI executable; defaults to target/release/raftinfer.
- RAFTINFER_CONTEXT_TOKENS — model context length; default 4096.
- RAFTINFER_MAX_NEW_TOKENS — exact-parity generation length; default 32.
- RAFTINFER_KV_CACHE_DTYPE — KV cache dtype (f32 or bf16); benchmark defaults
  to bf16, while parity requires it explicitly.
- RAFTINFER_KV_CACHE_LAYOUT — KV cache layout (token-major or head-major);
  benchmark defaults to head-major, while parity requires it explicitly.
- RAFTINFER_GPU_LOCK — lock-file path serializing target-GPU runs; default
  /tmp/raftinfer-qwen35-gpu.lock.
- RAFTINFER_GPU_ID — GPU index used by benchmark metadata; default 0.
- RAFTINFER_PREFLIGHT_RETRIES — positive preflight retry count; default 30.
- RAFTINFER_PREFLIGHT_RETRY_SECONDS — non-negative delay between preflight
  attempts; default 1.
- GPU_PREFLIGHT — preflight executable; defaults to scripts/gpu-preflight.sh.
- QWEN35_GENERATION_CORPUS — exact-generation corpus JSONL used by parity.
- PARITY_OUTPUT — parity JSONL output path.
- PARITY_REPORT — accepted parity JSONL input for benchmark aggregation.
- BENCHMARK_OUTPUT — benchmark JSONL output path.
- LLAMA_MODEL — required llama.cpp GGUF model, which must match the RAFTInfer
  artifact digest.
- LLAMA_SERVER_BIN — required pinned llama-server executable.
- LLAMA_SERVER_PORT — loopback port passed to llama-server.

## BF16 artifact preparation and provenance

- HF_MODEL_DIR — local Hugging Face model checkout used for conversion.
- HF_MODEL_REVISION — required 40-character model revision.
- HF_MODEL_REVISION_FILE — file containing model-revision evidence.
- LLAMA_CONVERTER_DIR — local pinned llama.cpp converter checkout.
- LLAMA_CONVERTER_REVISION — required converter revision.
- LLAMA_REFERENCE_DIR — local pinned llama.cpp reference checkout.
- LLAMA_REFERENCE_REVISION — required reference revision.
- TRANSFORMERS_REVISION — required Transformers revision used by conversion.
- CURL_BIN — curl executable for artifact retrieval.
- PYTHON_BIN — Python executable for the converter.
- JQ_BIN — jq executable used to validate and construct JSON evidence.
- OUTPUT_GGUF — generated BF16 GGUF output path.
- PROVENANCE_OUTPUT — generated provenance JSON output path.
- PROVENANCE_JSON — pinned provenance JSON input used by benchmark and BF16
  gate scripts.
- CUDA_VERSION — recorded CUDA version for benchmark provenance.
- RAFT_VERSION — recorded RAFT version for benchmark provenance.
- RMM_VERSION — recorded RMM version for benchmark provenance.

## Build and gate controls

- RAFTINFER_ENABLE_CUDA — CMake CUDA build toggle (ON or OFF).
- RAFTINFER_NATIVE_LIBRARY_TYPE — CMake/Rust native library type (STATIC or
  SHARED).
- RAFTINFER_NATIVE_LIBRARY_DIRS — additional native library search paths for
  CUDA/RMM linking.
- RAFTINFER_BUILD_TESTS — CMake test-build toggle.
- RAFTINFER_RUNTIME_LIB — explicit runtime library passed to target validation.
- RAFTINFER_RUN_GPU_TESTS — enables target-GPU CTest coverage.
- RAFTINFER_VALIDATION_ROOT — validated root directory for target-GPU checks.
- RAFTINFER_WEIGHT_FORMAT — requested model weight format for validation.
