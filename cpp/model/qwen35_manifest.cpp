#include "qwen35_manifest.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace brt::model {
namespace {

const gguf::TensorInfo *
require_tensor(const gguf::Catalog &catalog, const std::string &name,
               std::vector<std::uint64_t> expected_dimensions) {
  const auto *tensor = catalog.find_tensor(name);
  if (tensor == nullptr) {
    throw ManifestError("missing Qwen3.5 tensor: " + name);
  }
  if (tensor->dimensions != expected_dimensions) {
    throw ManifestError("Qwen3.5 tensor has an invalid shape: " + name);
  }
  return tensor;
}

std::string block_name(std::uint32_t index, const std::string &suffix) {
  return "blk." + std::to_string(index) + "." + suffix;
}

} // namespace

Qwen35Manifest validate_qwen35_manifest(const gguf::Catalog &catalog,
                                        const Qwen35Config &config) {
  if (config.blocks.empty()) {
    throw ManifestError("Qwen3.5 block plan must not be empty");
  }

  Qwen35Manifest manifest;
  manifest.token_embedding =
      require_tensor(catalog, "token_embd.weight",
                     {config.hidden_size, config.vocabulary_size});
  manifest.output_norm =
      require_tensor(catalog, "output_norm.weight", {config.hidden_size});
  manifest.output = require_tensor(
      catalog, "output.weight", {config.hidden_size, config.vocabulary_size});
  manifest.layers.reserve(config.blocks.size());

  const std::uint64_t full_query_width =
      static_cast<std::uint64_t>(config.full_attention_head_count) *
      config.full_attention_head_dimension;
  const std::uint64_t full_kv_width =
      static_cast<std::uint64_t>(config.full_attention_kv_head_count) *
      config.full_attention_head_dimension;
  const std::uint64_t linear_key_width =
      static_cast<std::uint64_t>(config.linear_key_head_count) *
      config.linear_head_dimension;
  const std::uint64_t linear_value_width =
      static_cast<std::uint64_t>(config.linear_value_head_count) *
      config.linear_head_dimension;
  const std::uint64_t linear_qkv_width =
      linear_key_width * 2 + linear_value_width;

  for (std::size_t position = 0; position < config.blocks.size(); ++position) {
    const auto &block = config.blocks[position];
    if (block.index != position) {
      throw ManifestError("Qwen3.5 block plan indices must be contiguous");
    }

    Qwen35LayerManifest layer;
    layer.index = block.index;
    layer.common.input_norm =
        require_tensor(catalog, block_name(block.index, "attn_norm.weight"),
                       {config.hidden_size});
    layer.common.post_attention_norm = require_tensor(
        catalog, block_name(block.index, "post_attention_norm.weight"),
        {config.hidden_size});
    layer.common.ffn_gate =
        require_tensor(catalog, block_name(block.index, "ffn_gate.weight"),
                       {config.hidden_size, config.intermediate_size});
    layer.common.ffn_down =
        require_tensor(catalog, block_name(block.index, "ffn_down.weight"),
                       {config.intermediate_size, config.hidden_size});
    layer.common.ffn_up =
        require_tensor(catalog, block_name(block.index, "ffn_up.weight"),
                       {config.hidden_size, config.intermediate_size});

    if (block.kind == Qwen35BlockKind::full_attention) {
      layer.full_attention = Qwen35FullAttentionTensors{
          .query =
              require_tensor(catalog, block_name(block.index, "attn_q.weight"),
                             {config.hidden_size, full_query_width * 2}),
          .key =
              require_tensor(catalog, block_name(block.index, "attn_k.weight"),
                             {config.hidden_size, full_kv_width}),
          .value =
              require_tensor(catalog, block_name(block.index, "attn_v.weight"),
                             {config.hidden_size, full_kv_width}),
          .output = require_tensor(
              catalog, block_name(block.index, "attn_output.weight"),
              {full_query_width, config.hidden_size}),
          .query_norm = require_tensor(
              catalog, block_name(block.index, "attn_q_norm.weight"),
              {config.full_attention_head_dimension}),
          .key_norm = require_tensor(
              catalog, block_name(block.index, "attn_k_norm.weight"),
              {config.full_attention_head_dimension}),
      };
    } else {
      layer.linear_attention = Qwen35LinearAttentionTensors{
          .qkv = require_tensor(catalog,
                                block_name(block.index, "attn_qkv.weight"),
                                {config.hidden_size, linear_qkv_width}),
          .gate = require_tensor(catalog,
                                 block_name(block.index, "attn_gate.weight"),
                                 {config.hidden_size, linear_value_width}),
          .convolution = require_tensor(
              catalog, block_name(block.index, "ssm_conv1d.weight"),
              {config.linear_convolution_width, linear_qkv_width}),
          .time_step_bias =
              require_tensor(catalog, block_name(block.index, "ssm_dt.bias"),
                             {config.linear_value_head_count}),
          .recurrent_a =
              require_tensor(catalog, block_name(block.index, "ssm_a"),
                             {config.linear_value_head_count}),
          .beta = require_tensor(
              catalog, block_name(block.index, "ssm_beta.weight"),
              {config.hidden_size, config.linear_value_head_count}),
          .alpha = require_tensor(
              catalog, block_name(block.index, "ssm_alpha.weight"),
              {config.hidden_size, config.linear_value_head_count}),
          .output_norm = require_tensor(
              catalog, block_name(block.index, "ssm_norm.weight"),
              {config.linear_head_dimension}),
          .output =
              require_tensor(catalog, block_name(block.index, "ssm_out.weight"),
                             {linear_value_width, config.hidden_size}),
      };
    }
    manifest.layers.push_back(std::move(layer));
  }
  return manifest;
}

} // namespace brt::model
