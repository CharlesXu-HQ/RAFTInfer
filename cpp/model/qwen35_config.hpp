#pragma once

#include "gguf_types.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace brt::model {

class ConfigError : public std::runtime_error {
public:
  explicit ConfigError(const std::string &message)
      : std::runtime_error(message) {}
};

enum class Qwen35BlockKind {
  linear_attention,
  full_attention,
};

struct Qwen35BlockPlan {
  std::uint32_t index{};
  Qwen35BlockKind kind{};
};

struct Qwen35Config {
  std::uint32_t vocabulary_size{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t context_length{};
  std::uint32_t full_attention_head_count{};
  std::uint32_t full_attention_kv_head_count{};
  std::uint32_t full_attention_head_dimension{};
  std::uint32_t linear_key_head_count{};
  std::uint32_t linear_value_head_count{};
  std::uint32_t linear_head_dimension{};
  std::uint32_t linear_convolution_width{};
  std::uint32_t rotary_dimension{};
  float rms_norm_epsilon{};
  float rope_frequency_base{};
  std::vector<Qwen35BlockPlan> blocks;
};

Qwen35Config derive_qwen35_config(const gguf::Catalog &catalog);

} // namespace brt::model
