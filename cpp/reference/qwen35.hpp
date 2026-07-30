#pragma once

#include "../model/qwen35_config.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace raftinfer::reference {

struct FullAttentionReferenceArgs {
  std::size_t tokens;
  std::size_t hidden_size;
  std::size_t query_heads;
  std::size_t kv_heads;
  std::size_t head_dim;
  std::size_t rotary_dim;
  std::size_t position_offset;
  float rope_base;
  float epsilon;

  static FullAttentionReferenceArgs
  from_config(const model::Qwen35Config &config, std::size_t tokens,
              std::size_t position_offset);
};

struct FullAttentionReferenceWeights {
  std::span<const float> query_norm_weight;
  std::span<const float> key_norm_weight;
  std::span<const float> output_weight;
};

struct GatedDeltaReferenceArgs {
  std::size_t tokens;
  std::size_t hidden_size;
  std::size_t key_heads;
  std::size_t value_heads;
  std::size_t key_dim;
  std::size_t value_dim;
  std::size_t conv_width;
  float epsilon;

  static GatedDeltaReferenceArgs from_config(const model::Qwen35Config &config,
                                             std::size_t tokens);
};

struct GatedDeltaReferenceWeights {
  std::span<const float> conv_weight;
  // GGUF stores the already transformed negative coefficient -exp(A_log).
  std::span<const float> recurrent_a;
  std::span<const float> dt_bias;
  std::span<const float> output_norm_weight;
};

struct GatedDeltaReferenceState {
  std::vector<float> convolution;
  std::vector<float> recurrent;

  explicit GatedDeltaReferenceState(const GatedDeltaReferenceArgs &args);
};

void qwen35_rms_norm(std::span<const float> input,
                     std::span<const float> weight, std::span<float> output,
                     std::size_t rows, std::size_t cols, float epsilon);

// Per-token input layout: [q, k, v, gate], where q has
// query_heads * head_dim elements, k/v each have kv_heads * head_dim elements,
// and gate has hidden_size elements.
void qwen35_gated_full_attention(std::span<const float> input,
                                 std::span<float> output,
                                 FullAttentionReferenceWeights weights,
                                 const FullAttentionReferenceArgs &args);

// Per-token input layout: [q, k, v, b, a, z]. The causal convolution applies
// only to q/k/v, whose combined width is
// 2 * key_heads * key_dim + value_heads * value_dim.
void qwen35_gated_delta_step(std::span<const float> input,
                             std::span<float> output,
                             GatedDeltaReferenceWeights weights,
                             const GatedDeltaReferenceArgs &args,
                             GatedDeltaReferenceState &state);

void qwen35_gated_delta_prefill(std::span<const float> input,
                                std::span<float> output,
                                GatedDeltaReferenceWeights weights,
                                const GatedDeltaReferenceArgs &args,
                                GatedDeltaReferenceState &state);

} // namespace raftinfer::reference
