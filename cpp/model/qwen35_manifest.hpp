#pragma once

#include "gguf_types.hpp"
#include "qwen35_config.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace raftinfer::model {

class ManifestError : public std::runtime_error {
public:
  explicit ManifestError(const std::string &message)
      : std::runtime_error(message) {}
};

struct Qwen35CommonLayerTensors {
  const gguf::TensorInfo *input_norm{};
  const gguf::TensorInfo *post_attention_norm{};
  const gguf::TensorInfo *ffn_gate{};
  const gguf::TensorInfo *ffn_down{};
  const gguf::TensorInfo *ffn_up{};
};

struct Qwen35FullAttentionTensors {
  const gguf::TensorInfo *query{};
  const gguf::TensorInfo *key{};
  const gguf::TensorInfo *value{};
  const gguf::TensorInfo *output{};
  const gguf::TensorInfo *query_norm{};
  const gguf::TensorInfo *key_norm{};
};

struct Qwen35LinearAttentionTensors {
  const gguf::TensorInfo *qkv{};
  const gguf::TensorInfo *gate{};
  const gguf::TensorInfo *convolution{};
  const gguf::TensorInfo *time_step_bias{};
  const gguf::TensorInfo *recurrent_a{};
  const gguf::TensorInfo *beta{};
  const gguf::TensorInfo *alpha{};
  const gguf::TensorInfo *output_norm{};
  const gguf::TensorInfo *output{};
};

struct Qwen35LayerManifest {
  std::uint32_t index{};
  Qwen35CommonLayerTensors common;
  std::optional<Qwen35FullAttentionTensors> full_attention;
  std::optional<Qwen35LinearAttentionTensors> linear_attention;
};

struct Qwen35Manifest {
  const gguf::TensorInfo *token_embedding{};
  const gguf::TensorInfo *output_norm{};
  const gguf::TensorInfo *output{};
  std::vector<Qwen35LayerManifest> layers;
};

Qwen35Manifest validate_qwen35_manifest(const gguf::Catalog &catalog,
                                        const Qwen35Config &config);

} // namespace raftinfer::model
