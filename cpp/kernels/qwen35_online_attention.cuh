#pragma once

#include "qwen35_attention.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace brt::kernels {

bool qwen35_online_attention_prefill_supported(
    Qwen35AttentionShape shape, BrtDataType activation_dtype,
    Qwen35AttentionLaunchPolicy policy) noexcept;

std::size_t
qwen35_online_attention_workspace_bytes(Qwen35AttentionShape shape) noexcept;

void qwen35_online_attention_prefill(
    const void *query, const void *key, const void *value, const void *gate,
    void *output, void *kv_cache, std::size_t kv_cache_bytes,
    Qwen35AttentionShape shape, BrtDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, Qwen35KvCacheLayout cache_layout,
    cudaStream_t stream);

void qwen35_online_attention_decode(
    const void *query, const void *key, const void *value, const void *gate,
    void *output, void *kv_cache, std::size_t kv_cache_bytes,
    Qwen35AttentionShape shape, BrtDataType activation_dtype,
    Qwen35KvCacheDType cache_dtype, Qwen35KvCacheLayout cache_layout,
    const std::uint32_t *device_position, cudaStream_t stream);

} // namespace brt::kernels
