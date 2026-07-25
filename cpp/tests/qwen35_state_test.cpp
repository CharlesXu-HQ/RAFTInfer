#include "../execution/qwen35_state.hpp"

#include "assert_enabled.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kNoSlot = std::numeric_limits<std::uint32_t>::max();
std::atomic<std::size_t> g_allocation_count{0};
}

void *operator new(std::size_t size) {
  g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void *pointer = std::malloc(size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void operator delete(void *pointer) noexcept { std::free(pointer); }

void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }

namespace {

brt::model::Qwen35Config small_hybrid_config() {
  brt::model::Qwen35Config config;
  config.hidden_size = 16;
  config.context_length = 16;
  config.full_attention_head_count = 4;
  config.full_attention_kv_head_count = 2;
  config.full_attention_head_dimension = 4;
  config.linear_key_head_count = 2;
  config.linear_value_head_count = 4;
  config.linear_head_dimension = 3;
  config.linear_convolution_width = 4;
  config.blocks = {
      {.index = 0,
       .kind = brt::model::Qwen35BlockKind::linear_attention},
      {.index = 1,
       .kind = brt::model::Qwen35BlockKind::linear_attention},
      {.index = 2, .kind = brt::model::Qwen35BlockKind::full_attention},
      {.index = 3,
       .kind = brt::model::Qwen35BlockKind::linear_attention},
  };
  return config;
}

brt::model::Qwen35Config official_shape_config() {
  auto config = small_hybrid_config();
  config.hidden_size = 4096;
  config.context_length = 262144;
  config.full_attention_head_count = 16;
  config.full_attention_kv_head_count = 4;
  config.full_attention_head_dimension = 256;
  config.linear_key_head_count = 16;
  config.linear_value_head_count = 32;
  config.linear_head_dimension = 128;
  config.linear_convolution_width = 4;
  config.blocks.clear();
  config.blocks.reserve(32);
  for (std::uint32_t index = 0; index < 32; ++index) {
    config.blocks.push_back(
        {.index = index,
         .kind = (index + 1) % 4 == 0
                     ? brt::model::Qwen35BlockKind::full_attention
                     : brt::model::Qwen35BlockKind::linear_attention});
  }
  return config;
}

brt::model::Qwen35Config two_full_layer_config() {
  auto config = small_hybrid_config();
  config.blocks = {
      {.index = 0, .kind = brt::model::Qwen35BlockKind::full_attention},
      {.index = 1,
       .kind = brt::model::Qwen35BlockKind::linear_attention},
      {.index = 2, .kind = brt::model::Qwen35BlockKind::full_attention},
  };
  return config;
}

template <class Exception, class Fn>
void expect_throw(const char *case_name, Fn &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const Exception &) {
    thrown = true;
  }
  if (!thrown) {
    std::cerr << "expected exception was not thrown: " << case_name << '\n';
  }
  assert(thrown);
}

void expect_all_zero(std::span<const float> values) {
  for (const float value : values) {
    assert(value == 0.0F);
  }
}

brt::Qwen35StateLayout valid_small_layout() {
  return brt::Qwen35StateLayout::create(small_hybrid_config(), 8);
}

} // namespace

int main() {
  const auto small_layout = valid_small_layout();
  assert(small_layout.max_context_tokens == 8);
  assert(small_layout.block_count == 4);
  assert(small_layout.linear_layer_count == 3);
  assert(small_layout.full_layer_count == 1);
  assert(small_layout.linear_slots_by_block ==
         (std::vector<std::uint32_t>{0, 1, kNoSlot, 2}));
  assert(small_layout.full_slots_by_block ==
         (std::vector<std::uint32_t>{kNoSlot, kNoSlot, 0, kNoSlot}));
  assert(small_layout.linear_qkv_channel_count == 24);
  assert(small_layout.linear_convolution_floats_per_layer == 72);
  assert(small_layout.linear_recurrent_floats_per_layer == 36);
  assert(small_layout.full_kv_floats_per_layer == 128);

  const auto official_layout =
      brt::Qwen35StateLayout::create(official_shape_config(), 1024);
  assert(official_layout.linear_layer_count == 24);
  assert(official_layout.full_layer_count == 8);
  assert(official_layout.linear_slots_by_block[0] == 0);
  assert(official_layout.linear_slots_by_block[2] == 2);
  assert(official_layout.full_slots_by_block[3] == 0);
  assert(official_layout.linear_slots_by_block[4] == 3);
  assert(official_layout.full_slots_by_block[31] == 7);
  assert(official_layout.linear_qkv_channel_count == 8192);
  assert(official_layout.linear_recurrent_floats_per_layer == 524288);

  expect_throw<std::invalid_argument>(
      "zero max context",
      [] { (void)brt::Qwen35StateLayout::create(small_hybrid_config(), 0); });
  auto empty = small_hybrid_config();
  empty.blocks.clear();
  expect_throw<std::invalid_argument>(
      "empty block plan in layout create",
      [&] { (void)brt::Qwen35StateLayout::create(empty, 8); });
  expect_throw<std::invalid_argument>(
      "empty layout in host state constructor",
      [] { (void)brt::Qwen35HostState{brt::Qwen35StateLayout{}}; });
  expect_throw<std::invalid_argument>(
      "unknown host storage mode",
      [&] {
        (void)brt::Qwen35HostState{
            small_layout, static_cast<brt::Qwen35HostStorage>(99)};
      });

  auto zero_count_with_slot = valid_small_layout();
  zero_count_with_slot.linear_layer_count = 0;
  expect_throw<std::invalid_argument>(
      "host state rejects zero linear count with slots",
      [&] { (void)brt::Qwen35HostState{zero_count_with_slot}; });

  auto out_of_range_slot = valid_small_layout();
  out_of_range_slot.linear_slots_by_block[0] =
      out_of_range_slot.linear_layer_count;
  expect_throw<std::invalid_argument>(
      "host state rejects out-of-range linear slot",
      [&] { (void)brt::Qwen35HostState{out_of_range_slot}; });

  auto duplicate_slot = valid_small_layout();
  duplicate_slot.linear_slots_by_block[1] = 0;
  expect_throw<std::invalid_argument>(
      "host state rejects duplicate linear slot",
      [&] { (void)brt::Qwen35HostState{duplicate_slot}; });

  auto non_exhaustive_slots = valid_small_layout();
  ++non_exhaustive_slots.linear_layer_count;
  expect_throw<std::invalid_argument>(
      "host state rejects non-exhaustive linear slots",
      [&] { (void)brt::Qwen35HostState{non_exhaustive_slots}; });

  auto both_type_slots = valid_small_layout();
  both_type_slots.full_slots_by_block[0] = 0;
  expect_throw<std::invalid_argument>(
      "host state rejects block with both slot types",
      [&] { (void)brt::Qwen35HostState{both_type_slots}; });

  auto neither_type_slot = valid_small_layout();
  neither_type_slot.linear_slots_by_block[0] = kNoSlot;
  expect_throw<std::invalid_argument>(
      "host state rejects block with neither slot type",
      [&] { (void)brt::Qwen35HostState{neither_type_slot}; });

  auto zero_context_layout = valid_small_layout();
  zero_context_layout.max_context_tokens = 0;
  expect_throw<std::invalid_argument>(
      "host state rejects zero context in public layout",
      [&] { (void)brt::Qwen35HostState{zero_context_layout}; });

  auto malformed_dimension_layout = valid_small_layout();
  malformed_dimension_layout.linear_head_dimension = 0;
  expect_throw<std::invalid_argument>(
      "host state rejects malformed public dimensions",
      [&] { (void)brt::Qwen35HostState{malformed_dimension_layout}; });

  auto constructor_overflow_layout = valid_small_layout();
  constructor_overflow_layout.linear_key_head_count =
      std::numeric_limits<std::uint32_t>::max();
  constructor_overflow_layout.linear_head_dimension =
      std::numeric_limits<std::uint32_t>::max();
  expect_throw<std::length_error>(
      "host state rejects constructor-side layout formula overflow",
      [&] { (void)brt::Qwen35HostState{constructor_overflow_layout}; });

  auto too_large_vector_layout = valid_small_layout();
  too_large_vector_layout.max_context_tokens = std::uint32_t{1} << 31;
  too_large_vector_layout.full_attention_kv_head_count = 1;
  too_large_vector_layout.full_attention_head_dimension =
      std::uint32_t{1} << 30;
  too_large_vector_layout.full_kv_floats_per_layer = std::size_t{1} << 62;
  expect_throw<std::length_error>(
      "host state rejects vector max_size overflow before allocation",
      [&] { (void)brt::Qwen35HostState{too_large_vector_layout}; });

  auto overflowing = small_hybrid_config();
  overflowing.blocks = {{.index = 0,
                         .kind = brt::model::Qwen35BlockKind::full_attention}};
  overflowing.full_attention_kv_head_count =
      std::numeric_limits<std::uint32_t>::max();
  overflowing.full_attention_head_dimension =
      std::numeric_limits<std::uint32_t>::max();
  expect_throw<std::length_error>(
      "full KV byte overflow",
      [&] { (void)brt::Qwen35StateLayout::create(overflowing, 2); });

  brt::Qwen35HostState state{small_layout};
  assert(state.has_tensor_storage());
  const auto *convolution_base = state.linear_convolution(0).data();
  const auto *recurrent_base = state.linear_recurrent(1).data();
  const auto *kv_base = state.full_kv(2).data();
  const auto convolution_capacity = state.linear_convolution(0).size();
  const auto recurrent_capacity = state.linear_recurrent(1).size();
  const auto kv_capacity = state.full_kv(2).size();

  assert(state.position() == 0);
  assert(state.full_kv_length(2) == 0);
  assert(state.linear_convolution(0).size() == 72);
  assert(state.linear_recurrent(1).size() == 36);
  assert(state.full_kv(2).size() == 128);

  g_allocation_count.store(0, std::memory_order_relaxed);
  (void)state.position();
  (void)state.full_kv_length(2);
  (void)state.linear_convolution(0);
  (void)state.linear_recurrent(1);
  (void)state.full_kv(2);
  state.commit_tokens(1);
  state.reset();
  assert(g_allocation_count.load(std::memory_order_relaxed) == 0);
  assert(state.position() == 0);
  assert(state.full_kv_length(2) == 0);

  expect_throw<std::invalid_argument>(
      "linear convolution accessor rejects full layer",
      [&] { (void)state.linear_convolution(2); });
  expect_throw<std::invalid_argument>(
      "linear recurrent accessor rejects full layer",
      [&] { (void)state.linear_recurrent(2); });
  expect_throw<std::invalid_argument>(
      "full KV accessor rejects linear layer", [&] { (void)state.full_kv(1); });
  expect_throw<std::out_of_range>(
      "full KV length rejects out-of-range layer",
      [&] { (void)state.full_kv_length(4); });

  state.linear_convolution(0)[3] = 1.0F;
  state.linear_recurrent(1)[5] = 2.0F;
  state.full_kv(2)[7] = 3.0F;
  state.commit_tokens(3);
  assert(state.position() == 3);
  assert(state.full_kv_length(2) == 3);
  expect_throw<std::length_error>("single-full commit overflow leaves state",
                                  [&] { state.commit_tokens(6); });
  assert(state.position() == 3);
  assert(state.full_kv_length(2) == 3);

  brt::Qwen35HostState two_full_state{
      brt::Qwen35StateLayout::create(two_full_layer_config(), 4)};
  two_full_state.commit_tokens(3);
  assert(two_full_state.position() == 3);
  assert(two_full_state.full_kv_length(0) == 3);
  assert(two_full_state.full_kv_length(2) == 3);
  expect_throw<std::length_error>("multi-full commit overflow leaves all state",
                                  [&] { two_full_state.commit_tokens(2); });
  assert(two_full_state.position() == 3);
  assert(two_full_state.full_kv_length(0) == 3);
  assert(two_full_state.full_kv_length(2) == 3);

  state.reset();
  assert(state.position() == 0);
  assert(state.full_kv_length(2) == 0);
  assert(state.linear_convolution(0).data() == convolution_base);
  assert(state.linear_recurrent(1).data() == recurrent_base);
  assert(state.full_kv(2).data() == kv_base);
  assert(state.linear_convolution(0).size() == convolution_capacity);
  assert(state.linear_recurrent(1).size() == recurrent_capacity);
  assert(state.full_kv(2).size() == kv_capacity);
  expect_all_zero(state.linear_convolution(0));
  expect_all_zero(state.linear_recurrent(1));
  expect_all_zero(state.full_kv(2));

  auto logical_layout = valid_small_layout();
  g_allocation_count.store(0, std::memory_order_relaxed);
  brt::Qwen35HostState logical_only{std::move(logical_layout),
                                    brt::Qwen35HostStorage::LogicalOnly};
  assert(g_allocation_count.load(std::memory_order_relaxed) == 1);
  assert(!logical_only.has_tensor_storage());
  assert(logical_only.position() == 0);
  assert(logical_only.full_kv_length(2) == 0);
  expect_throw<std::logic_error>(
      "logical-only state rejects convolution access",
      [&] { (void)logical_only.linear_convolution(0); });
  expect_throw<std::logic_error>(
      "logical-only state rejects recurrent access",
      [&] { (void)logical_only.linear_recurrent(0); });
  expect_throw<std::logic_error>("logical-only state rejects KV access",
                                 [&] { (void)logical_only.full_kv(2); });
  g_allocation_count.store(0, std::memory_order_relaxed);
  logical_only.commit_tokens(3);
  logical_only.reset();
  assert(g_allocation_count.load(std::memory_order_relaxed) == 0);
  assert(logical_only.position() == 0);
  assert(logical_only.full_kv_length(2) == 0);
}
