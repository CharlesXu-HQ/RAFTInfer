#include "qwen35_config.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace brt::model {
namespace {

const gguf::MetadataValue &require(const gguf::Catalog &catalog,
                                   const std::string &key) {
  const auto *value = catalog.find_metadata(key);
  if (value == nullptr) {
    throw ConfigError("missing Qwen3.5 metadata: " + key);
  }
  return *value;
}

std::uint32_t require_u32_value(const gguf::MetadataValue &value,
                                const std::string &key) {
  if (const auto *typed = value.get_if<std::uint32_t>()) {
    return *typed;
  }
  if (const auto *typed = value.get_if<std::uint64_t>();
      typed != nullptr && *typed <= std::numeric_limits<std::uint32_t>::max()) {
    return static_cast<std::uint32_t>(*typed);
  }
  throw ConfigError("Qwen3.5 metadata must be an unsigned 32-bit value: " +
                    key);
}

std::uint32_t require_u32(const gguf::Catalog &catalog,
                          const std::string &key) {
  const auto &value = require(catalog, key);
  return require_u32_value(value, key);
}

std::uint32_t optional_u32(const gguf::Catalog &catalog,
                           const std::string &key,
                           std::uint32_t default_value) {
  const auto *value = catalog.find_metadata(key);
  if (value == nullptr) {
    return default_value;
  }
  return require_u32_value(*value, key);
}

float require_float(const gguf::Catalog &catalog, const std::string &key) {
  const auto &value = require(catalog, key);
  if (const auto *typed = value.get_if<float>()) {
    return *typed;
  }
  if (const auto *typed = value.get_if<double>();
      typed != nullptr && std::isfinite(*typed) &&
      std::abs(*typed) <= std::numeric_limits<float>::max()) {
    return static_cast<float>(*typed);
  }
  throw ConfigError("Qwen3.5 metadata must be a finite float: " + key);
}

void require_nonzero(std::uint32_t value, const std::string &key) {
  if (value == 0) {
    throw ConfigError("Qwen3.5 metadata must be nonzero: " + key);
  }
}

Qwen35BlockKind parse_layer_type(const gguf::MetadataValue &value,
                                 std::uint32_t index) {
  const auto *name = value.get_if<std::string>();
  if (name == nullptr) {
    throw ConfigError("qwen35.layer_types must be a string array");
  }
  if (*name == "linear_attention") {
    return Qwen35BlockKind::linear_attention;
  }
  if (*name == "full_attention") {
    return Qwen35BlockKind::full_attention;
  }
  throw ConfigError("qwen35.layer_types has an invalid layer type at block " +
                    std::to_string(index));
}

} // namespace

Qwen35Config derive_qwen35_config(const gguf::Catalog &catalog) {
  const auto &architecture = require(catalog, "general.architecture");
  const auto *architecture_name = architecture.get_if<std::string>();
  if (architecture_name == nullptr || *architecture_name != "qwen35") {
    throw ConfigError("general.architecture must be qwen35");
  }

  Qwen35Config config;
  config.hidden_size = require_u32(catalog, "qwen35.embedding_length");
  config.intermediate_size = require_u32(catalog, "qwen35.feed_forward_length");
  config.context_length = require_u32(catalog, "qwen35.context_length");
  const auto total_block_count = require_u32(catalog, "qwen35.block_count");
  const auto nextn_predict_layers =
      optional_u32(catalog, "qwen35.nextn_predict_layers", 0);
  config.full_attention_head_count =
      require_u32(catalog, "qwen35.attention.head_count");
  config.full_attention_kv_head_count =
      require_u32(catalog, "qwen35.attention.head_count_kv");
  const auto attention_key_length =
      require_u32(catalog, "qwen35.attention.key_length");
  const auto attention_value_length =
      require_u32(catalog, "qwen35.attention.value_length");
  config.full_attention_head_dimension = attention_key_length;
  config.rms_norm_epsilon =
      require_float(catalog, "qwen35.attention.layer_norm_rms_epsilon");
  config.rope_frequency_base = require_float(catalog, "qwen35.rope.freq_base");
  config.rotary_dimension = require_u32(catalog, "qwen35.rope.dimension_count");
  config.linear_convolution_width =
      require_u32(catalog, "qwen35.ssm.conv_kernel");
  config.linear_head_dimension = require_u32(catalog, "qwen35.ssm.state_size");
  config.linear_key_head_count = require_u32(catalog, "qwen35.ssm.group_count");
  const auto linear_time_step_rank =
      require_u32(catalog, "qwen35.ssm.time_step_rank");
  const auto linear_inner_size = require_u32(catalog, "qwen35.ssm.inner_size");
  const auto full_attention_interval =
      require_u32(catalog, "qwen35.full_attention_interval");

  const auto &tokens = require(catalog, "tokenizer.ggml.tokens");
  const auto *token_array = tokens.get_if<gguf::MetadataArray>();
  if (token_array == nullptr ||
      token_array->element_type != gguf::MetadataType::string ||
      token_array->values.empty() ||
      token_array->values.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw ConfigError("tokenizer.ggml.tokens must be a nonempty string array");
  }
  config.vocabulary_size =
      static_cast<std::uint32_t>(token_array->values.size());

  require_nonzero(config.hidden_size, "qwen35.embedding_length");
  require_nonzero(config.intermediate_size, "qwen35.feed_forward_length");
  require_nonzero(config.context_length, "qwen35.context_length");
  require_nonzero(total_block_count, "qwen35.block_count");
  require_nonzero(config.full_attention_head_count,
                  "qwen35.attention.head_count");
  require_nonzero(config.full_attention_kv_head_count,
                  "qwen35.attention.head_count_kv");
  require_nonzero(attention_key_length, "qwen35.attention.key_length");
  require_nonzero(attention_value_length, "qwen35.attention.value_length");
  require_nonzero(config.linear_convolution_width, "qwen35.ssm.conv_kernel");
  require_nonzero(config.linear_head_dimension, "qwen35.ssm.state_size");
  require_nonzero(config.linear_key_head_count, "qwen35.ssm.group_count");
  require_nonzero(linear_time_step_rank, "qwen35.ssm.time_step_rank");
  require_nonzero(linear_inner_size, "qwen35.ssm.inner_size");
  require_nonzero(full_attention_interval, "qwen35.full_attention_interval");

  if (attention_key_length != attention_value_length) {
    throw ConfigError("Qwen3.5 attention key/value dimensions must match");
  }
  if (config.full_attention_head_count % config.full_attention_kv_head_count !=
      0) {
    throw ConfigError("Qwen3.5 query heads must be divisible by KV heads");
  }
  const std::uint64_t full_width =
      static_cast<std::uint64_t>(config.full_attention_head_count) *
      config.full_attention_head_dimension;
  if (full_width != config.hidden_size) {
    throw ConfigError("Qwen3.5 attention heads do not match the hidden size");
  }
  if (linear_inner_size % config.linear_head_dimension != 0) {
    throw ConfigError("Qwen3.5 SSM inner size must be divisible by state size");
  }
  config.linear_value_head_count =
      linear_inner_size / config.linear_head_dimension;
  if (config.linear_value_head_count < config.linear_key_head_count ||
      config.linear_value_head_count % config.linear_key_head_count != 0) {
    throw ConfigError(
        "Qwen3.5 linear value heads must be a multiple of key heads");
  }
  if (linear_time_step_rank != config.linear_value_head_count) {
    throw ConfigError(
        "Qwen3.5 SSM time-step rank must match the value-head count");
  }
  if (config.rotary_dimension == 0 ||
      config.rotary_dimension > config.full_attention_head_dimension ||
      config.rotary_dimension % 2 != 0) {
    throw ConfigError("Qwen3.5 rotary dimension is invalid");
  }
  if (!std::isfinite(config.rms_norm_epsilon) ||
      config.rms_norm_epsilon <= 0.0F) {
    throw ConfigError("Qwen3.5 RMSNorm epsilon must be positive");
  }
  if (!std::isfinite(config.rope_frequency_base) ||
      config.rope_frequency_base <= 0.0F) {
    throw ConfigError("Qwen3.5 RoPE frequency base must be positive");
  }
  if (nextn_predict_layers >= total_block_count) {
    throw ConfigError(
        "qwen35.nextn_predict_layers must be smaller than block count");
  }
  const auto main_block_count = total_block_count - nextn_predict_layers;
  if (main_block_count % full_attention_interval != 0) {
    throw ConfigError(
        "Qwen3.5 block count must be divisible by the attention interval");
  }

  const auto *layer_types_value = catalog.find_metadata("qwen35.layer_types");
  const gguf::MetadataArray *layer_types = nullptr;
  if (layer_types_value != nullptr) {
    layer_types = layer_types_value->get_if<gguf::MetadataArray>();
    if (layer_types == nullptr ||
        layer_types->element_type != gguf::MetadataType::string ||
        layer_types->values.size() != main_block_count) {
      throw ConfigError(
          "qwen35.layer_types must be a string array matching block count");
    }
  }

  config.blocks.reserve(main_block_count);
  for (std::uint32_t index = 0; index < main_block_count; ++index) {
    const bool interval_full_attention =
        (index + 1) % full_attention_interval == 0;
    const auto kind =
        layer_types == nullptr
            ? (interval_full_attention ? Qwen35BlockKind::full_attention
                                       : Qwen35BlockKind::linear_attention)
            : parse_layer_type(layer_types->values[index], index);
    if (layer_types != nullptr && ((kind == Qwen35BlockKind::full_attention) !=
                                   interval_full_attention)) {
      throw ConfigError(
          "qwen35.layer_types disagrees with qwen35.full_attention_interval");
    }
    config.blocks.push_back(Qwen35BlockPlan{.index = index, .kind = kind});
  }
  return config;
}

} // namespace brt::model
