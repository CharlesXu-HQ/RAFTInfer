#pragma once

#include <cstdint>

namespace brt {

enum class Qwen35AttentionImplementation : std::uint8_t {
  materialized_reference,
  online_tiled,
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
  bool decode_graph{true};
  bool grouped_input_casts{true};
};

} // namespace brt
