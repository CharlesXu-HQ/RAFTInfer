# RAFTInfer

[English](README.md)

面向 RTX 50 系列 GPU、由正确性门控的 RAFT/RMM 推理运行时，并使用自有 CUDA 高性能算子。

![Qwen3.5-9B BF16 RTX 5090 吞吐](docs/assets/qwen35-bf16-rtx5090.svg)

## 基准摘要

已接受的 NVIDIA GeForce RTX 5090 上 Qwen3.5-9B BF16 证据仅比较 RAFTInfer 与
llama.cpp。五项展示的测量中 RAFTInfer 均更高：PP128、PP512 prefill，以及
PP128、PP512、TG128@PP512 generation。图中 ratio 是源数据精确 ratio 四舍五入至三位
小数；见[基准方法](docs/benchmarks.md)与已提交的
[schema-v2 evidence](benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl)。
经测量的 `auto` 策略在两个长上下文 arm 使用 split-k-256，同时 PP128 保留零
workspace 的 single-block 路径。

协议为每个 arm 5 次 warmup、20 次测量，BF16 Qwen3.5-9B ModelScope revision
`460979c3d11864dd16408d860ac930a360a2fac2`、固定 llama.cpp revision
`aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3`，并对 4 prompts × 32 tokens 完成
exact greedy parity。这是单一已接受 RTX 5090 环境的结果，不是普遍的硬件或工作负载结论。

## 范围

- 仅支持 RTX 50 系列 Blackwell (`sm_120a`)；不支持其他 GPU 系列。
- 已接受证据覆盖 Qwen3.5-9B text generation、BF16 weights 和 BF16 KV cache。
- RAFT 与 RMM 是 device resource 与 allocation 基础层；项目自有 C++/CUDA 负责
  高性能模型算子；Rust 负责 coarse ABI/runtime、tokenizer、CLI 与 orchestration。
- `bw24` material 只可经 license、provenance、correctness、numerical 和 RTX 50
  performance gate 后复用。RAFTInfer 从不发布 `bw24` 性能比较。

## 架构

```text
Rust CLI/runtime → coarse C ABI → C++ execution plans → RAFT/RMM → cuBLASLt + custom CUDA kernels
```

ABI 在 prefill、decode、reset、logits transfer 等 model/session 操作处跨语言；Rust
不调用单个 CUDA operator。详见 [docs/architecture.md](docs/architecture.md)。

## 快速开始

```bash
cmake -S . -B build/host -G Ninja -DRAFTINFER_ENABLE_CUDA=OFF
cmake --build build/host
cargo run -p raftinfer-cli -- info
scripts/local-check.sh
```

在 RTX 50 host 上用 CUDA 构建，并使用改名后的 CLI：

```bash
cargo run -p raftinfer-cli -- generate \
  --model /path/to/Qwen3.5-9B-c202236-bf16.gguf \
  --prompt "Hello" --max-new-tokens 32 --context 4096 \
  --kv-cache-dtype bf16 --kv-cache-layout head-major --output-format json
```

## Parity 与性能复现

准备固定的 BF16 GGUF 后，依次运行 shared-GPU preflight、exact greedy parity、benchmark
和 BF16 gate：

```bash
scripts/gpu-preflight.sh
scripts/qwen35-parity.sh
scripts/qwen35-benchmark.sh
scripts/qwen35-bf16-gate.sh build/evidence/qwen35-benchmark.jsonl
```

脚本需要的 model 与 llama.cpp environment variables 请参阅
[脚本环境变量参考](docs/environment.md)。已提交数据用以下命令确定性渲染：

```bash
python3 tools/render_benchmark_chart.py benchmarks/results/qwen35-9b-bf16-rtx5090.jsonl docs/assets/qwen35-bf16-rtx5090.svg
```

## 局限

- 该比较仅为单一 RTX 5090 configuration 上的 Qwen3.5-9B BF16；不代表其他 model、
  quantization、GPU、batch size 或 serving workload。
- 比较报告 throughput，不报告 end-to-end service latency、power 或 cost。peak memory
  是 RAFTInfer 的 RMM logical-allocation peak，不是 whole-GPU process memory。
- CUDA correctness、parity 与 performance gate 需要目标 GPU；host-only CI 不执行它们。

## 路线图

在保持 correctness-before-performance gate 的前提下，扩展 Qwen3.5 coverage、target
validation automation 与可复现性文档。

## 贡献与安全

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 中的 review 与 RTX 50 validation contract；
漏洞报告见 [SECURITY.md](SECURITY.md)。

## 引用

引用 RAFTInfer 时请使用 [CITATION.cff](CITATION.cff)。

## 许可证

RAFTInfer source 使用 [Apache-2.0](LICENSE)。
