#include "../model/qwen35_config.hpp"

#include "assert_enabled.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void put_u32(brt::gguf::Catalog &catalog, const std::string &key,
             std::uint32_t value) {
  catalog.metadata.emplace(key, brt::gguf::MetadataValue{value});
}

void put_f32(brt::gguf::Catalog &catalog, const std::string &key, float value) {
  catalog.metadata.emplace(key, brt::gguf::MetadataValue{value});
}

std::vector<brt::gguf::MetadataValue> layer_types_interval4() {
  std::vector<brt::gguf::MetadataValue> values;
  values.reserve(32);
  for (std::uint32_t index = 0; index < 32; ++index) {
    values.emplace_back(std::string{(index + 1) % 4 == 0 ? "full_attention"
                                                         : "linear_attention"});
  }
  return values;
}

void put_layer_types(brt::gguf::Catalog &catalog,
                     std::vector<brt::gguf::MetadataValue> values) {
  catalog.metadata.emplace("qwen35.layer_types",
                           brt::gguf::MetadataValue{brt::gguf::MetadataArray{
                               .element_type = brt::gguf::MetadataType::string,
                               .values = std::move(values),
                           }});
}

void replace_layer_types(brt::gguf::Catalog &catalog,
                         std::vector<brt::gguf::MetadataValue> values) {
  catalog.metadata.at("qwen35.layer_types") =
      brt::gguf::MetadataValue{brt::gguf::MetadataArray{
          .element_type = brt::gguf::MetadataType::string,
          .values = std::move(values),
      }};
}

brt::gguf::Catalog official_qwen35_9b_metadata() {
  brt::gguf::Catalog catalog;
  catalog.metadata.emplace("general.architecture",
                           brt::gguf::MetadataValue{std::string{"qwen35"}});
  put_u32(catalog, "qwen35.block_count", 32);
  put_u32(catalog, "qwen35.context_length", 262144);
  put_u32(catalog, "qwen35.embedding_length", 4096);
  put_u32(catalog, "qwen35.feed_forward_length", 12288);
  put_u32(catalog, "qwen35.attention.head_count", 16);
  put_u32(catalog, "qwen35.attention.head_count_kv", 4);
  put_u32(catalog, "qwen35.attention.key_length", 256);
  put_u32(catalog, "qwen35.attention.value_length", 256);
  put_f32(catalog, "qwen35.attention.layer_norm_rms_epsilon", 1.0e-6F);
  put_f32(catalog, "qwen35.rope.freq_base", 10'000'000.0F);
  put_u32(catalog, "qwen35.rope.dimension_count", 64);
  put_u32(catalog, "qwen35.ssm.conv_kernel", 4);
  put_u32(catalog, "qwen35.ssm.state_size", 128);
  put_u32(catalog, "qwen35.ssm.group_count", 16);
  put_u32(catalog, "qwen35.ssm.time_step_rank", 32);
  put_u32(catalog, "qwen35.ssm.inner_size", 4096);
  put_u32(catalog, "qwen35.full_attention_interval", 4);

  brt::gguf::MetadataArray tokens{
      .element_type = brt::gguf::MetadataType::string,
      .values = {},
  };
  tokens.values.reserve(248320);
  for (std::uint32_t index = 0; index < 248320; ++index) {
    tokens.values.emplace_back(std::string{});
  }
  catalog.metadata.emplace("tokenizer.ggml.tokens",
                           brt::gguf::MetadataValue{std::move(tokens)});
  put_layer_types(catalog, layer_types_interval4());
  return catalog;
}

template <class Fn> void expect_config_error(Fn &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const brt::model::ConfigError &) {
    thrown = true;
  }
  assert(thrown);
}

} // namespace

int main() {
  const auto catalog = official_qwen35_9b_metadata();
  const auto config = brt::model::derive_qwen35_config(catalog);

  assert(config.vocabulary_size == 248320);
  assert(config.hidden_size == 4096);
  assert(config.intermediate_size == 12288);
  assert(config.context_length == 262144);
  assert(config.blocks.size() == 32);
  assert(config.full_attention_head_count == 16);
  assert(config.full_attention_kv_head_count == 4);
  assert(config.full_attention_head_dimension == 256);
  assert(config.linear_key_head_count == 16);
  assert(config.linear_value_head_count == 32);
  assert(config.linear_head_dimension == 128);
  assert(config.linear_convolution_width == 4);
  assert(config.rotary_dimension == 64);
  assert(std::abs(config.rms_norm_epsilon - 1.0e-6F) < 1.0e-12F);

  std::size_t full_count = 0;
  for (std::size_t index = 0; index < config.blocks.size(); ++index) {
    assert(config.blocks[index].index == index);
    const bool expected_full = (index + 1) % 4 == 0;
    assert((config.blocks[index].kind ==
            brt::model::Qwen35BlockKind::full_attention) == expected_full);
    full_count += expected_full ? 1 : 0;
  }
  assert(full_count == 8);

  auto nextn_catalog = catalog;
  nextn_catalog.metadata.at("qwen35.block_count") =
      brt::gguf::MetadataValue{std::uint32_t{33}};
  put_u32(nextn_catalog, "qwen35.nextn_predict_layers", 1);
  nextn_catalog.metadata.erase("qwen35.layer_types");
  const auto nextn_config = brt::model::derive_qwen35_config(nextn_catalog);
  assert(nextn_config.blocks.size() == 32);
  assert(nextn_config.blocks.back().index == 31);
  assert(nextn_config.blocks.back().kind ==
         brt::model::Qwen35BlockKind::full_attention);

  auto nextn_with_main_layer_types = catalog;
  nextn_with_main_layer_types.metadata.at("qwen35.block_count") =
      brt::gguf::MetadataValue{std::uint32_t{33}};
  put_u32(nextn_with_main_layer_types, "qwen35.nextn_predict_layers", 1);
  const auto nextn_typed_config =
      brt::model::derive_qwen35_config(nextn_with_main_layer_types);
  assert(nextn_typed_config.blocks.size() == 32);

  auto nextn_with_total_layer_types = nextn_with_main_layer_types;
  auto extra_layer_values = layer_types_interval4();
  extra_layer_values.emplace_back(std::string{"linear_attention"});
  replace_layer_types(nextn_with_total_layer_types,
                      std::move(extra_layer_values));
  expect_config_error([&] {
    (void)brt::model::derive_qwen35_config(nextn_with_total_layer_types);
  });

  auto invalid_nextn = catalog;
  put_u32(invalid_nextn, "qwen35.nextn_predict_layers", 32);
  expect_config_error(
      [&] { (void)brt::model::derive_qwen35_config(invalid_nextn); });

  auto wrong_architecture = catalog;
  wrong_architecture.metadata.at("general.architecture") =
      brt::gguf::MetadataValue{std::string{"qwen3"}};
  expect_config_error(
      [&] { (void)brt::model::derive_qwen35_config(wrong_architecture); });

  auto invalid_interval = catalog;
  invalid_interval.metadata.at("qwen35.full_attention_interval") =
      brt::gguf::MetadataValue{std::uint32_t{0}};
  expect_config_error(
      [&] { (void)brt::model::derive_qwen35_config(invalid_interval); });

  auto invalid_heads = catalog;
  invalid_heads.metadata.at("qwen35.attention.head_count") =
      brt::gguf::MetadataValue{std::uint32_t{15}};
  expect_config_error(
      [&] { (void)brt::model::derive_qwen35_config(invalid_heads); });

  auto invalid_layer_type = catalog;
  auto invalid_layer_values = layer_types_interval4();
  invalid_layer_values[0] =
      brt::gguf::MetadataValue{std::string{"sliding_attention"}};
  replace_layer_types(invalid_layer_type, std::move(invalid_layer_values));
  expect_config_error(
      [&] { (void)brt::model::derive_qwen35_config(invalid_layer_type); });

  auto disagreeing_layer_type = catalog;
  auto disagreeing_layer_values = layer_types_interval4();
  disagreeing_layer_values[3] =
      brt::gguf::MetadataValue{std::string{"linear_attention"}};
  replace_layer_types(disagreeing_layer_type,
                      std::move(disagreeing_layer_values));
  expect_config_error(
      [&] { (void)brt::model::derive_qwen35_config(disagreeing_layer_type); });

  auto wrong_layer_count = catalog;
  auto short_layer_values = layer_types_interval4();
  short_layer_values.pop_back();
  replace_layer_types(wrong_layer_count, std::move(short_layer_values));
  expect_config_error(
      [&] { (void)brt::model::derive_qwen35_config(wrong_layer_count); });
}
