#include "../model/qwen35_manifest.hpp"

#include "assert_enabled.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

raftinfer::model::Qwen35Config small_config() {
  return raftinfer::model::Qwen35Config{
      .vocabulary_size = 16,
      .hidden_size = 8,
      .intermediate_size = 16,
      .context_length = 128,
      .full_attention_head_count = 2,
      .full_attention_kv_head_count = 1,
      .full_attention_head_dimension = 4,
      .linear_key_head_count = 1,
      .linear_value_head_count = 2,
      .linear_head_dimension = 4,
      .linear_convolution_width = 4,
      .rotary_dimension = 2,
      .rms_norm_epsilon = 1.0e-6F,
      .rope_frequency_base = 10'000.0F,
      .blocks =
          {
              {0, raftinfer::model::Qwen35BlockKind::linear_attention},
              {1, raftinfer::model::Qwen35BlockKind::linear_attention},
              {2, raftinfer::model::Qwen35BlockKind::linear_attention},
              {3, raftinfer::model::Qwen35BlockKind::full_attention},
          },
  };
}

void add_tensor(raftinfer::gguf::Catalog &catalog, std::string name,
                std::vector<std::uint64_t> dimensions) {
  catalog.tensors.push_back(raftinfer::gguf::TensorInfo{
      .name = std::move(name),
      .dimensions = std::move(dimensions),
      .type = 1,
      .offset = 0,
      .byte_size = 0,
  });
}

std::string block_name(std::uint32_t index, const std::string &suffix) {
  return "blk." + std::to_string(index) + "." + suffix;
}

raftinfer::gguf::Catalog complete_manifest(const raftinfer::model::Qwen35Config &config) {
  raftinfer::gguf::Catalog catalog;
  add_tensor(catalog, "token_embd.weight",
             {config.hidden_size, config.vocabulary_size});
  add_tensor(catalog, "output_norm.weight", {config.hidden_size});
  add_tensor(catalog, "output.weight",
             {config.hidden_size, config.vocabulary_size});

  const std::uint64_t linear_key_width =
      config.linear_key_head_count * config.linear_head_dimension;
  const std::uint64_t linear_value_width =
      config.linear_value_head_count * config.linear_head_dimension;
  const std::uint64_t linear_qkv_width =
      linear_key_width * 2 + linear_value_width;
  const std::uint64_t attention_kv_width = config.full_attention_kv_head_count *
                                           config.full_attention_head_dimension;

  for (const auto &block : config.blocks) {
    add_tensor(catalog, block_name(block.index, "attn_norm.weight"),
               {config.hidden_size});
    add_tensor(catalog, block_name(block.index, "post_attention_norm.weight"),
               {config.hidden_size});
    add_tensor(catalog, block_name(block.index, "ffn_gate.weight"),
               {config.hidden_size, config.intermediate_size});
    add_tensor(catalog, block_name(block.index, "ffn_down.weight"),
               {config.intermediate_size, config.hidden_size});
    add_tensor(catalog, block_name(block.index, "ffn_up.weight"),
               {config.hidden_size, config.intermediate_size});

    if (block.kind == raftinfer::model::Qwen35BlockKind::full_attention) {
      add_tensor(catalog, block_name(block.index, "attn_q.weight"),
                 {config.hidden_size, config.full_attention_head_count *
                                          config.full_attention_head_dimension *
                                          2});
      add_tensor(catalog, block_name(block.index, "attn_k.weight"),
                 {config.hidden_size, attention_kv_width});
      add_tensor(catalog, block_name(block.index, "attn_v.weight"),
                 {config.hidden_size, attention_kv_width});
      add_tensor(catalog, block_name(block.index, "attn_output.weight"),
                 {config.full_attention_head_count *
                      config.full_attention_head_dimension,
                  config.hidden_size});
      add_tensor(catalog, block_name(block.index, "attn_q_norm.weight"),
                 {config.full_attention_head_dimension});
      add_tensor(catalog, block_name(block.index, "attn_k_norm.weight"),
                 {config.full_attention_head_dimension});
    } else {
      add_tensor(catalog, block_name(block.index, "attn_qkv.weight"),
                 {config.hidden_size, linear_qkv_width});
      add_tensor(catalog, block_name(block.index, "attn_gate.weight"),
                 {config.hidden_size, linear_value_width});
      add_tensor(catalog, block_name(block.index, "ssm_conv1d.weight"),
                 {config.linear_convolution_width, linear_qkv_width});
      add_tensor(catalog, block_name(block.index, "ssm_dt.bias"),
                 {config.linear_value_head_count});
      add_tensor(catalog, block_name(block.index, "ssm_a"),
                 {config.linear_value_head_count});
      add_tensor(catalog, block_name(block.index, "ssm_beta.weight"),
                 {config.hidden_size, config.linear_value_head_count});
      add_tensor(catalog, block_name(block.index, "ssm_alpha.weight"),
                 {config.hidden_size, config.linear_value_head_count});
      add_tensor(catalog, block_name(block.index, "ssm_norm.weight"),
                 {config.linear_head_dimension});
      add_tensor(catalog, block_name(block.index, "ssm_out.weight"),
                 {linear_value_width, config.hidden_size});
    }
  }
  return catalog;
}

template <class Fn> void expect_manifest_error(Fn &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const raftinfer::model::ManifestError &) {
    thrown = true;
  }
  assert(thrown);
}

} // namespace

int main() {
  const auto config = small_config();
  const auto catalog = complete_manifest(config);
  const auto manifest = raftinfer::model::validate_qwen35_manifest(catalog, config);

  assert(manifest.token_embedding->name == "token_embd.weight");
  assert(manifest.output_norm->name == "output_norm.weight");
  assert(manifest.output->name == "output.weight");
  assert(manifest.layers.size() == 4);
  assert(manifest.layers[0].linear_attention.has_value());
  assert(!manifest.layers[0].full_attention.has_value());
  assert(manifest.layers[3].full_attention.has_value());
  assert(!manifest.layers[3].linear_attention.has_value());
  assert(manifest.layers[3].full_attention->query->name ==
         "blk.3.attn_q.weight");
  assert(manifest.layers[0].linear_attention->recurrent_a->name ==
         "blk.0.ssm_a");

  auto missing = catalog;
  missing.tensors.erase(missing.tensors.begin());
  add_tensor(missing, "v.token_embd.weight",
             {config.hidden_size, config.vocabulary_size});
  expect_manifest_error(
      [&] { (void)raftinfer::model::validate_qwen35_manifest(missing, config); });

  auto wrong_shape = catalog;
  bool tensor_found = false;
  for (auto &tensor : wrong_shape.tensors) {
    if (tensor.name == "blk.3.attn_q_norm.weight") {
      tensor.dimensions = {config.full_attention_head_dimension + 1};
      tensor_found = true;
      break;
    }
  }
  assert(tensor_found);
  expect_manifest_error(
      [&] { (void)raftinfer::model::validate_qwen35_manifest(wrong_shape, config); });
}
