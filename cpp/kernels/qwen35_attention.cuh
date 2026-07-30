#pragma once

#include "../execution/qwen35_execution_policy.hpp"
#include "qwen35_primitives.cuh"

#include <raftinfer/tensor.h>

#include <cuda_runtime_api.h>

#include <cstddef>

namespace raftinfer::kernels {

struct Qwen35AttentionShape {
  std::size_t tokens;
  std::size_t query_heads;
  std::size_t kv_heads;
  std::size_t head_dim;
  std::size_t max_context_tokens;
  std::size_t past_tokens;
};

struct Qwen35AttentionLaunchPolicy {
  raftinfer::Qwen35AttentionImplementation implementation;
  raftinfer::Qwen35KvCacheDType kv_cache_dtype;
  raftinfer::Qwen35KvCacheLayout kv_cache_layout;
};

std::size_t qwen35_attention_workspace_floats(Qwen35AttentionShape shape);
std::size_t qwen35_attention_workspace_bytes(Qwen35AttentionShape shape);
std::size_t qwen35_attention_cache_bytes(Qwen35AttentionShape shape,
                                         Qwen35AttentionLaunchPolicy policy);
std::size_t
qwen35_attention_workspace_bytes(Qwen35AttentionShape shape,
                                 Qwen35AttentionLaunchPolicy policy);

void qwen35_causal_attention(const void *query, const void *key,
                             const void *value, const void *gate, void *output,
                             void *kv_cache, std::size_t kv_cache_bytes,
                             void *workspace, std::size_t workspace_bytes,
                             Qwen35AttentionShape shape,
                             RaftInferDataType activation_dtype,
                             Qwen35AttentionLaunchPolicy policy,
                             cudaStream_t stream);

void qwen35_causal_attention(const void *query, const void *key,
                             const void *value, const void *gate, void *output,
                             float *kv_cache, float *logits_workspace,
                             std::size_t logits_workspace_floats,
                             Qwen35AttentionShape shape, RaftInferDataType dtype,
                             cudaStream_t stream);

} // namespace raftinfer::kernels
