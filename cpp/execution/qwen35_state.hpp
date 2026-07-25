#pragma once

#include "../model/qwen35_config.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace brt {

enum class Qwen35HostStorage {
  Full,
  LogicalOnly,
};

struct Qwen35StateLayout {
  std::uint32_t max_context_tokens{};
  std::uint32_t block_count{};
  std::uint32_t linear_layer_count{};
  std::uint32_t full_layer_count{};
  std::uint32_t linear_convolution_width{};
  std::uint32_t linear_qkv_channel_count{};
  std::uint32_t linear_value_head_count{};
  std::uint32_t linear_key_head_count{};
  std::uint32_t linear_head_dimension{};
  std::uint32_t full_attention_kv_head_count{};
  std::uint32_t full_attention_head_dimension{};
  std::size_t linear_convolution_floats_per_layer{};
  std::size_t linear_recurrent_floats_per_layer{};
  std::size_t full_kv_floats_per_layer{};
  std::size_t linear_convolution_bytes{};
  std::size_t linear_recurrent_bytes{};
  std::size_t full_kv_bytes{};
  std::vector<std::uint32_t> linear_slots_by_block;
  std::vector<std::uint32_t> full_slots_by_block;

  static Qwen35StateLayout create(const model::Qwen35Config &config,
                                  std::uint32_t max_context_tokens);
};

class Qwen35HostState {
public:
  explicit Qwen35HostState(
      Qwen35StateLayout layout,
      Qwen35HostStorage storage = Qwen35HostStorage::Full);

  const Qwen35StateLayout &layout() const noexcept;
  bool has_tensor_storage() const noexcept;
  std::uint32_t position() const noexcept;
  std::uint32_t full_kv_length(std::uint32_t layer) const;
  std::span<float> linear_convolution(std::uint32_t layer);
  std::span<const float> linear_convolution(std::uint32_t layer) const;
  std::span<float> linear_recurrent(std::uint32_t layer);
  std::span<const float> linear_recurrent(std::uint32_t layer) const;
  std::span<float> full_kv(std::uint32_t layer);
  std::span<const float> full_kv(std::uint32_t layer) const;
  void commit_tokens(std::uint32_t count);
  void reset() noexcept;

private:
  Qwen35StateLayout layout_;
  Qwen35HostStorage storage_{};
  std::uint32_t position_{};
  std::vector<std::uint32_t> full_kv_lengths_;
  std::vector<float> linear_convolution_;
  std::vector<float> linear_recurrent_;
  std::vector<float> full_kv_;
};

} // namespace brt
