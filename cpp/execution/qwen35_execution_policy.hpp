#pragma once

#include <cstdint>

namespace raftinfer {

enum class Qwen35AttentionImplementation : std::uint8_t {
  materialized_reference,
  online_tiled,
};

enum class Qwen35DecodeAttentionMode : std::uint8_t {
  auto_select,
  single_block,
  split_k_256,
  split_k_512,
};

enum class Qwen35KvCacheDType : std::uint8_t {
  f32,
  bf16,
};

enum class Qwen35KvCacheLayout : std::uint8_t {
  token_major,
  head_major,
};

struct Qwen35ExecutionPolicy {
  Qwen35AttentionImplementation attention{
      Qwen35AttentionImplementation::online_tiled};
  Qwen35KvCacheDType kv_cache{Qwen35KvCacheDType::f32};
  Qwen35KvCacheLayout kv_cache_layout{Qwen35KvCacheLayout::token_major};
  Qwen35DecodeAttentionMode decode_attention{
      Qwen35DecodeAttentionMode::auto_select};
  bool decode_graph{true};
  bool grouped_input_casts{true};
};

Qwen35ExecutionPolicy qwen35_execution_policy_from_environment(
    Qwen35ExecutionPolicy policy);

} // namespace raftinfer
