#include "qwen35_state.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace brt {
namespace {

constexpr std::uint32_t kNoSlot = std::numeric_limits<std::uint32_t>::max();

std::size_t checked_mul(std::size_t lhs, std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::length_error("Qwen3.5 state layout size overflow");
  }
  return lhs * rhs;
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::length_error("Qwen3.5 state layout size overflow");
  }
  return lhs + rhs;
}

std::uint32_t checked_u32(std::size_t value) {
  if (value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("Qwen3.5 state layout size overflow");
  }
  return static_cast<std::uint32_t>(value);
}

void require_nonzero(std::uint32_t value, const char *name) {
  if (value == 0) {
    throw std::invalid_argument(name);
  }
}

std::size_t checked_bytes(std::size_t floats) {
  return checked_mul(floats, sizeof(float));
}

void require_vector_capacity(std::size_t size, std::size_t max_size) {
  if (size > max_size) {
    throw std::length_error("Qwen3.5 host state vector size overflow");
  }
}

std::span<float> slot_span(std::vector<float> &storage, std::size_t slot,
                           std::size_t floats_per_slot) noexcept {
  return std::span<float>{storage.data() + slot * floats_per_slot,
                          floats_per_slot};
}

std::span<const float> slot_span(const std::vector<float> &storage,
                                 std::size_t slot,
                                 std::size_t floats_per_slot) noexcept {
  return std::span<const float>{storage.data() + slot * floats_per_slot,
                                floats_per_slot};
}

Qwen35StateLayout require_valid_layout(Qwen35StateLayout layout) {
  if (layout.block_count == 0 || layout.linear_slots_by_block.empty() ||
      layout.full_slots_by_block.empty()) {
    throw std::invalid_argument("Qwen3.5 host state requires a nonempty layout");
  }
  if (layout.linear_slots_by_block.size() != layout.block_count ||
      layout.full_slots_by_block.size() != layout.block_count) {
    throw std::invalid_argument("Qwen3.5 host state layout slot map mismatch");
  }
  require_nonzero(layout.max_context_tokens,
                  "Qwen3.5 max context tokens must be nonzero");
  require_nonzero(layout.linear_convolution_width,
                  "Qwen3.5 linear convolution width must be nonzero");
  require_nonzero(layout.linear_value_head_count,
                  "Qwen3.5 linear value head count must be nonzero");
  require_nonzero(layout.linear_key_head_count,
                  "Qwen3.5 linear key head count must be nonzero");
  require_nonzero(layout.linear_head_dimension,
                  "Qwen3.5 linear head dimension must be nonzero");
  require_nonzero(layout.full_attention_kv_head_count,
                  "Qwen3.5 full attention KV head count must be nonzero");
  require_nonzero(layout.full_attention_head_dimension,
                  "Qwen3.5 full attention head dimension must be nonzero");

  const auto expected_qkv = checked_u32(checked_add(
      checked_mul(checked_mul(std::size_t{2}, layout.linear_key_head_count),
                  layout.linear_head_dimension),
      checked_mul(layout.linear_value_head_count,
                  layout.linear_head_dimension)));
  const auto expected_convolution =
      checked_mul(expected_qkv, layout.linear_convolution_width - std::size_t{1});
  const auto expected_recurrent =
      checked_mul(layout.linear_value_head_count,
                  checked_mul(layout.linear_head_dimension,
                              layout.linear_head_dimension));
  const auto expected_full_kv = checked_mul(
      checked_mul(std::size_t{2}, layout.max_context_tokens),
      checked_mul(layout.full_attention_kv_head_count,
                  layout.full_attention_head_dimension));
  if (layout.linear_qkv_channel_count != expected_qkv ||
      layout.linear_convolution_floats_per_layer != expected_convolution ||
      layout.linear_recurrent_floats_per_layer != expected_recurrent ||
      layout.full_kv_floats_per_layer != expected_full_kv) {
    throw std::invalid_argument("Qwen3.5 host state layout size mismatch");
  }

  std::uint32_t linear_slots = 0;
  std::uint32_t full_slots = 0;
  for (std::uint32_t block = 0; block < layout.block_count; ++block) {
    const auto linear_slot = layout.linear_slots_by_block[block];
    const auto full_slot = layout.full_slots_by_block[block];
    const bool has_linear = linear_slot != kNoSlot;
    const bool has_full = full_slot != kNoSlot;
    if (has_linear == has_full) {
      throw std::invalid_argument(
          "Qwen3.5 host state requires exactly one slot per block");
    }
    if (has_linear) {
      if (linear_slot >= layout.linear_layer_count) {
        throw std::invalid_argument("Qwen3.5 linear slot out of range");
      }
      for (std::uint32_t previous = 0; previous < block; ++previous) {
        if (layout.linear_slots_by_block[previous] == linear_slot) {
          throw std::invalid_argument("Qwen3.5 duplicate linear slot");
        }
      }
      ++linear_slots;
    } else {
      if (full_slot >= layout.full_layer_count) {
        throw std::invalid_argument("Qwen3.5 full slot out of range");
      }
      for (std::uint32_t previous = 0; previous < block; ++previous) {
        if (layout.full_slots_by_block[previous] == full_slot) {
          throw std::invalid_argument("Qwen3.5 duplicate full slot");
        }
      }
      ++full_slots;
    }
  }
  if (linear_slots != layout.linear_layer_count ||
      full_slots != layout.full_layer_count) {
    throw std::invalid_argument("Qwen3.5 host state slot count mismatch");
  }

  const auto linear_convolution_floats =
      checked_mul(layout.linear_layer_count,
                  layout.linear_convolution_floats_per_layer);
  const auto linear_recurrent_floats =
      checked_mul(layout.linear_layer_count,
                  layout.linear_recurrent_floats_per_layer);
  const auto full_kv_floats =
      checked_mul(layout.full_layer_count, layout.full_kv_floats_per_layer);
  const auto expected_convolution_bytes =
      checked_bytes(linear_convolution_floats);
  const auto expected_recurrent_bytes = checked_bytes(linear_recurrent_floats);
  const auto expected_full_kv_bytes = checked_bytes(full_kv_floats);
  (void)checked_add(checked_add(expected_convolution_bytes,
                                expected_recurrent_bytes),
                    expected_full_kv_bytes);
  if (layout.linear_convolution_bytes != expected_convolution_bytes ||
      layout.linear_recurrent_bytes != expected_recurrent_bytes ||
      layout.full_kv_bytes != expected_full_kv_bytes) {
    throw std::invalid_argument("Qwen3.5 host state byte-size mismatch");
  }

  require_vector_capacity(layout.full_layer_count,
                          std::vector<std::uint32_t>{}.max_size());
  require_vector_capacity(linear_convolution_floats,
                          std::vector<float>{}.max_size());
  require_vector_capacity(linear_recurrent_floats,
                          std::vector<float>{}.max_size());
  require_vector_capacity(full_kv_floats, std::vector<float>{}.max_size());
  return layout;
}

} // namespace

Qwen35StateLayout
Qwen35StateLayout::create(const model::Qwen35Config &config,
                          std::uint32_t max_context_tokens) {
  require_nonzero(max_context_tokens, "Qwen3.5 max context tokens must be nonzero");
  require_nonzero(config.linear_convolution_width,
                  "Qwen3.5 linear convolution width must be nonzero");
  require_nonzero(config.linear_value_head_count,
                  "Qwen3.5 linear value head count must be nonzero");
  require_nonzero(config.linear_key_head_count,
                  "Qwen3.5 linear key head count must be nonzero");
  require_nonzero(config.linear_head_dimension,
                  "Qwen3.5 linear head dimension must be nonzero");
  require_nonzero(config.full_attention_kv_head_count,
                  "Qwen3.5 full attention KV head count must be nonzero");
  require_nonzero(config.full_attention_head_dimension,
                  "Qwen3.5 full attention head dimension must be nonzero");

  if (config.blocks.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("Qwen3.5 block count overflow");
  }
  if (config.blocks.empty()) {
    throw std::invalid_argument("Qwen3.5 state layout requires blocks");
  }

  Qwen35StateLayout layout;
  layout.max_context_tokens = max_context_tokens;
  layout.block_count = static_cast<std::uint32_t>(config.blocks.size());
  layout.linear_convolution_width = config.linear_convolution_width;
  layout.linear_value_head_count = config.linear_value_head_count;
  layout.linear_key_head_count = config.linear_key_head_count;
  layout.linear_head_dimension = config.linear_head_dimension;
  layout.full_attention_kv_head_count = config.full_attention_kv_head_count;
  layout.full_attention_head_dimension = config.full_attention_head_dimension;

  for (std::uint32_t block_position = 0; block_position < layout.block_count;
       ++block_position) {
    const auto &block = config.blocks[block_position];
    if (block.index != block_position) {
      throw std::invalid_argument("Qwen3.5 blocks must be in index order");
    }
    if (block.kind == model::Qwen35BlockKind::linear_attention) {
      ++layout.linear_layer_count;
    } else {
      ++layout.full_layer_count;
    }
  }

  layout.linear_qkv_channel_count = checked_u32(checked_add(
      checked_mul(checked_mul(std::size_t{2}, config.linear_key_head_count),
                  config.linear_head_dimension),
      checked_mul(config.linear_value_head_count,
                  config.linear_head_dimension)));
  layout.linear_convolution_floats_per_layer =
      checked_mul(layout.linear_qkv_channel_count,
                  config.linear_convolution_width - std::size_t{1});
  layout.linear_recurrent_floats_per_layer =
      checked_mul(config.linear_value_head_count,
                  checked_mul(config.linear_head_dimension,
                              config.linear_head_dimension));
  layout.full_kv_floats_per_layer = checked_mul(
      checked_mul(std::size_t{2}, max_context_tokens),
      checked_mul(config.full_attention_kv_head_count,
                  config.full_attention_head_dimension));

  layout.linear_convolution_bytes = checked_bytes(checked_mul(
      layout.linear_layer_count, layout.linear_convolution_floats_per_layer));
  layout.linear_recurrent_bytes = checked_bytes(checked_mul(
      layout.linear_layer_count, layout.linear_recurrent_floats_per_layer));
  layout.full_kv_bytes =
      checked_bytes(checked_mul(layout.full_layer_count,
                                layout.full_kv_floats_per_layer));
  (void)checked_add(checked_add(layout.linear_convolution_bytes,
                                layout.linear_recurrent_bytes),
                    layout.full_kv_bytes);
  layout.linear_slots_by_block.assign(config.blocks.size(), kNoSlot);
  layout.full_slots_by_block.assign(config.blocks.size(), kNoSlot);
  std::uint32_t linear_slot = 0;
  std::uint32_t full_slot = 0;
  for (std::uint32_t block_position = 0; block_position < layout.block_count;
       ++block_position) {
    if (config.blocks[block_position].kind ==
        model::Qwen35BlockKind::linear_attention) {
      layout.linear_slots_by_block[block_position] = linear_slot++;
    } else {
      layout.full_slots_by_block[block_position] = full_slot++;
    }
  }
  return layout;
}

Qwen35HostState::Qwen35HostState(Qwen35StateLayout layout)
    : layout_(require_valid_layout(std::move(layout))),
      full_kv_lengths_(layout_.full_layer_count),
      linear_convolution_(checked_mul(
          layout_.linear_layer_count, layout_.linear_convolution_floats_per_layer)),
      linear_recurrent_(checked_mul(
          layout_.linear_layer_count, layout_.linear_recurrent_floats_per_layer)),
      full_kv_(checked_mul(layout_.full_layer_count,
                           layout_.full_kv_floats_per_layer)) {}

const Qwen35StateLayout &Qwen35HostState::layout() const noexcept {
  return layout_;
}

std::uint32_t Qwen35HostState::position() const noexcept { return position_; }

std::uint32_t Qwen35HostState::full_kv_length(std::uint32_t layer) const {
  if (layer >= layout_.full_slots_by_block.size()) {
    throw std::out_of_range("Qwen3.5 full KV layer index out of range");
  }
  const auto slot = layout_.full_slots_by_block[layer];
  if (slot == kNoSlot) {
    throw std::invalid_argument("Qwen3.5 layer does not use full attention");
  }
  return full_kv_lengths_[slot];
}

std::span<float> Qwen35HostState::linear_convolution(std::uint32_t layer) {
  if (layer >= layout_.linear_slots_by_block.size()) {
    throw std::out_of_range("Qwen3.5 linear layer index out of range");
  }
  const auto slot = layout_.linear_slots_by_block[layer];
  if (slot == kNoSlot) {
    throw std::invalid_argument("Qwen3.5 layer does not use linear attention");
  }
  return slot_span(linear_convolution_, slot,
                   layout_.linear_convolution_floats_per_layer);
}

std::span<const float>
Qwen35HostState::linear_convolution(std::uint32_t layer) const {
  if (layer >= layout_.linear_slots_by_block.size()) {
    throw std::out_of_range("Qwen3.5 linear layer index out of range");
  }
  const auto slot = layout_.linear_slots_by_block[layer];
  if (slot == kNoSlot) {
    throw std::invalid_argument("Qwen3.5 layer does not use linear attention");
  }
  return slot_span(linear_convolution_, slot,
                   layout_.linear_convolution_floats_per_layer);
}

std::span<float> Qwen35HostState::linear_recurrent(std::uint32_t layer) {
  if (layer >= layout_.linear_slots_by_block.size()) {
    throw std::out_of_range("Qwen3.5 linear layer index out of range");
  }
  const auto slot = layout_.linear_slots_by_block[layer];
  if (slot == kNoSlot) {
    throw std::invalid_argument("Qwen3.5 layer does not use linear attention");
  }
  return slot_span(linear_recurrent_, slot,
                   layout_.linear_recurrent_floats_per_layer);
}

std::span<const float>
Qwen35HostState::linear_recurrent(std::uint32_t layer) const {
  if (layer >= layout_.linear_slots_by_block.size()) {
    throw std::out_of_range("Qwen3.5 linear layer index out of range");
  }
  const auto slot = layout_.linear_slots_by_block[layer];
  if (slot == kNoSlot) {
    throw std::invalid_argument("Qwen3.5 layer does not use linear attention");
  }
  return slot_span(linear_recurrent_, slot,
                   layout_.linear_recurrent_floats_per_layer);
}

std::span<float> Qwen35HostState::full_kv(std::uint32_t layer) {
  if (layer >= layout_.full_slots_by_block.size()) {
    throw std::out_of_range("Qwen3.5 full KV layer index out of range");
  }
  const auto slot = layout_.full_slots_by_block[layer];
  if (slot == kNoSlot) {
    throw std::invalid_argument("Qwen3.5 layer does not use full attention");
  }
  return slot_span(full_kv_, slot, layout_.full_kv_floats_per_layer);
}

std::span<const float> Qwen35HostState::full_kv(std::uint32_t layer) const {
  if (layer >= layout_.full_slots_by_block.size()) {
    throw std::out_of_range("Qwen3.5 full KV layer index out of range");
  }
  const auto slot = layout_.full_slots_by_block[layer];
  if (slot == kNoSlot) {
    throw std::invalid_argument("Qwen3.5 layer does not use full attention");
  }
  return slot_span(full_kv_, slot, layout_.full_kv_floats_per_layer);
}

void Qwen35HostState::commit_tokens(std::uint32_t count) {
  if (count > layout_.max_context_tokens ||
      position_ > layout_.max_context_tokens - count) {
    throw std::length_error("Qwen3.5 state commit exceeds max context");
  }
  const auto next_position = position_ + count;
  for (const auto length : full_kv_lengths_) {
    if (count > layout_.max_context_tokens ||
        length > layout_.max_context_tokens - count) {
      throw std::length_error("Qwen3.5 full KV length exceeds max context");
    }
  }
  position_ = next_position;
  for (auto &length : full_kv_lengths_) {
    length += count;
  }
}

void Qwen35HostState::reset() noexcept {
  position_ = 0;
  std::fill(full_kv_lengths_.begin(), full_kv_lengths_.end(), 0);
  std::fill(linear_convolution_.begin(), linear_convolution_.end(), 0.0F);
  std::fill(linear_recurrent_.begin(), linear_recurrent_.end(), 0.0F);
  std::fill(full_kv_.begin(), full_kv_.end(), 0.0F);
}

} // namespace brt
