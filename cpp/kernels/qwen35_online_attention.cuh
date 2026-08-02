#pragma once

#include "qwen35_attention.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace raftinfer::kernels {

struct Qwen35OnlineDecodePlan {
  Qwen35DecodeAttentionMode resolved_mode{
      Qwen35DecodeAttentionMode::single_block};
  std::size_t partition_tokens{};
  std::size_t split_k_threshold_tokens{};
  std::size_t context_bucket_tokens{};
  std::size_t active_partition_capacity{1};

  bool operator==(const Qwen35OnlineDecodePlan &) const = default;
};

struct Qwen35OnlineDecodeWorkspaceLayout {
  std::size_t partial_count{};
  std::size_t max_offset_bytes{};
  std::size_t sum_offset_bytes{};
  std::size_t value_offset_bytes{};
  std::size_t bytes{};

  bool operator==(const Qwen35OnlineDecodeWorkspaceLayout &) const = default;
};

Qwen35OnlineDecodePlan qwen35_online_decode_plan(
    Qwen35AttentionShape shape, Qwen35DecodeAttentionMode requested,
    std::size_t context_tokens);

Qwen35OnlineDecodeWorkspaceLayout qwen35_online_decode_workspace_layout(
    Qwen35AttentionShape shape, Qwen35DecodeAttentionMode resolved_mode);

bool qwen35_online_attention_prefill_supported(
    Qwen35AttentionShape shape, RaftInferDataType activation_dtype,
    Qwen35AttentionLaunchPolicy policy) noexcept;

std::size_t
qwen35_online_attention_workspace_bytes(Qwen35AttentionShape shape) noexcept;

void qwen35_online_attention_prefill(
    const void *query, const void *key, const void *value, const void *gate,
    void *output, void *kv_cache, std::size_t kv_cache_bytes,
    Qwen35AttentionShape shape, RaftInferDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, Qwen35KvCacheLayout cache_layout,
    cudaStream_t stream);

void qwen35_online_attention_decode(
    const void *query, const void *key, const void *value, const void *gate,
    void *output, void *kv_cache, std::size_t kv_cache_bytes,
    Qwen35AttentionShape shape, RaftInferDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, Qwen35KvCacheLayout cache_layout,
    const std::uint32_t *device_position, cudaStream_t stream);

} // namespace raftinfer::kernels
