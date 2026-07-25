#pragma once

#include "qwen35_primitives.cuh"

#include <brt/tensor.h>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace brt::kernels {

struct Qwen35AttentionShape {
  std::size_t tokens;
  std::size_t query_heads;
  std::size_t kv_heads;
  std::size_t head_dim;
  std::size_t max_context_tokens;
  std::size_t past_tokens;
};

std::size_t qwen35_attention_workspace_floats(Qwen35AttentionShape shape);
std::size_t qwen35_attention_workspace_bytes(Qwen35AttentionShape shape);

void qwen35_causal_attention(const void *query, const void *key,
                             const void *value, const void *gate, void *output,
                             float *kv_cache, float *logits_workspace,
                             std::size_t logits_workspace_floats,
                             Qwen35AttentionShape shape, BrtDataType dtype,
                             cudaStream_t stream);

} // namespace brt::kernels
