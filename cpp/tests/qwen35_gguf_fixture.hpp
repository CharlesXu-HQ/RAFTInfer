#pragma once

#include "../model/gguf_types.hpp"

#include <bit>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace brt::test {

struct Qwen35GgufFixtureOptions {
  std::uint32_t vocabulary_size{16};
  std::uint32_t hidden_size{8};
  std::uint32_t intermediate_size{16};
  std::uint32_t context_length{128};
  std::uint32_t full_head_count{2};
  std::uint32_t full_kv_head_count{1};
  std::uint32_t full_head_dimension{4};
  std::uint32_t linear_key_head_count{1};
  std::uint32_t linear_value_head_count{2};
  std::uint32_t linear_head_dimension{4};
  std::uint32_t rotary_dimension{2};
  std::uint32_t block_count{4};
};

namespace detail {

template <class T> void append(std::vector<std::uint8_t> &bytes, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  if constexpr (sizeof(T) > 1) {
    static_assert(std::endian::native == std::endian::little);
  }
  const auto *source = reinterpret_cast<const std::uint8_t *>(&value);
  bytes.insert(bytes.end(), source, source + sizeof(value));
}

inline void append_string(std::vector<std::uint8_t> &bytes,
                          const std::string &value) {
  append<std::uint64_t>(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

inline void append_key(std::vector<std::uint8_t> &bytes, const std::string &key,
                       gguf::MetadataType type) {
  append_string(bytes, key);
  append<std::uint32_t>(bytes, static_cast<std::uint32_t>(type));
}

inline void append_u32_metadata(std::vector<std::uint8_t> &bytes,
                                const std::string &key, std::uint32_t value) {
  append_key(bytes, key, gguf::MetadataType::uint32);
  append(bytes, value);
}

inline void append_float_metadata(std::vector<std::uint8_t> &bytes,
                                  const std::string &key, float value) {
  append_key(bytes, key, gguf::MetadataType::float32);
  append(bytes, value);
}

inline void append_string_metadata(std::vector<std::uint8_t> &bytes,
                                   const std::string &key,
                                   const std::string &value) {
  append_key(bytes, key, gguf::MetadataType::string);
  append_string(bytes, value);
}

inline void
append_string_array_metadata(std::vector<std::uint8_t> &bytes,
                             const std::string &key,
                             const std::vector<std::string> &values) {
  append_key(bytes, key, gguf::MetadataType::array);
  append<std::uint32_t>(bytes,
                        static_cast<std::uint32_t>(gguf::MetadataType::string));
  append<std::uint64_t>(bytes, values.size());
  for (const auto &value : values) {
    append_string(bytes, value);
  }
}

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t type{};
  std::uint64_t offset{};
  std::uint64_t byte_size{};
};

inline void add_tensor(std::vector<Tensor> &tensors, std::uint64_t &next_offset,
                       std::string name,
                       std::vector<std::uint64_t> dimensions,
                       std::uint32_t tensor_type) {
  std::uint64_t elements = 1;
  for (const auto dimension : dimensions) {
    elements *= dimension;
  }
  const std::uint64_t element_size = tensor_type == 0 ? 4 : 2;
  const std::uint64_t byte_size = elements * element_size;
  tensors.push_back(Tensor{
      .name = std::move(name),
      .dimensions = std::move(dimensions),
      .type = tensor_type,
      .offset = next_offset,
      .byte_size = byte_size,
  });
  next_offset = (next_offset + byte_size + 31) / 32 * 32;
}

inline std::string block_name(std::uint32_t index, const std::string &suffix) {
  return "blk." + std::to_string(index) + "." + suffix;
}

} // namespace detail

inline std::vector<std::uint8_t>
make_qwen35_gguf_fixture(std::uint32_t tensor_type = 1,
                         bool f32_auxiliary_tensors = false,
                         Qwen35GgufFixtureOptions options = {}) {
  constexpr std::uint32_t f32_tensor_type = 0;
  const std::uint32_t linear_key_width =
      options.linear_key_head_count * options.linear_head_dimension;
  const std::uint32_t linear_value_width =
      options.linear_value_head_count * options.linear_head_dimension;
  const std::uint32_t linear_qkv_width =
      linear_key_width * 2 + linear_value_width;

  std::vector<std::uint8_t> metadata;
  std::uint64_t metadata_count = 0;
  const auto u32 = [&](const std::string &key, std::uint32_t value) {
    detail::append_u32_metadata(metadata, key, value);
    ++metadata_count;
  };
  const auto f32 = [&](const std::string &key, float value) {
    detail::append_float_metadata(metadata, key, value);
    ++metadata_count;
  };
  const auto string = [&](const std::string &key, const std::string &value) {
    detail::append_string_metadata(metadata, key, value);
    ++metadata_count;
  };
  const auto strings = [&](const std::string &key,
                           const std::vector<std::string> &values) {
    detail::append_string_array_metadata(metadata, key, values);
    ++metadata_count;
  };

  string("general.architecture", "qwen35");
  u32("general.alignment", 32);
  u32("qwen35.embedding_length", options.hidden_size);
  u32("qwen35.feed_forward_length", options.intermediate_size);
  u32("qwen35.context_length", options.context_length);
  u32("qwen35.block_count", options.block_count);
  u32("qwen35.attention.head_count", options.full_head_count);
  u32("qwen35.attention.head_count_kv", options.full_kv_head_count);
  u32("qwen35.attention.key_length", options.full_head_dimension);
  u32("qwen35.attention.value_length", options.full_head_dimension);
  f32("qwen35.attention.layer_norm_rms_epsilon", 1.0e-6F);
  f32("qwen35.rope.freq_base", 10'000.0F);
  u32("qwen35.rope.dimension_count", options.rotary_dimension);
  u32("qwen35.ssm.conv_kernel", 4);
  u32("qwen35.ssm.state_size", options.linear_head_dimension);
  u32("qwen35.ssm.group_count", options.linear_key_head_count);
  u32("qwen35.ssm.time_step_rank", options.linear_value_head_count);
  u32("qwen35.ssm.inner_size", linear_value_width);
  u32("qwen35.full_attention_interval", options.block_count);
  string("tokenizer.ggml.model", "gpt2");
  string("tokenizer.ggml.pre", "qwen2");
  strings("tokenizer.ggml.tokens", {"<eos>", "a", "b", "c", "d", "e", "f", "g",
                                    "h", "i", "j", "k", "l", "m", "n", "o"});
  strings("tokenizer.ggml.merges", {"a b", "b c"});
  u32("tokenizer.ggml.eos_token_id", 0);
  string("tokenizer.chat_template", "{{ messages }}");

  std::vector<detail::Tensor> tensors;
  std::uint64_t next_offset = 0;
  const auto main_type = tensor_type;
  const auto aux_type = f32_auxiliary_tensors ? f32_tensor_type : main_type;
  const auto add_main = [&](std::string name,
                            std::vector<std::uint64_t> dimensions) {
    detail::add_tensor(tensors, next_offset, std::move(name),
                       std::move(dimensions), main_type);
  };
  const auto add_aux = [&](std::string name,
                           std::vector<std::uint64_t> dimensions) {
    detail::add_tensor(tensors, next_offset, std::move(name),
                       std::move(dimensions), aux_type);
  };

  add_main("token_embd.weight", {options.hidden_size, options.vocabulary_size});
  add_aux("output_norm.weight", {options.hidden_size});
  add_main("output.weight", {options.hidden_size, options.vocabulary_size});

  for (std::uint32_t block = 0; block < options.block_count; ++block) {
    add_aux(detail::block_name(block, "attn_norm.weight"), {options.hidden_size});
    add_aux(detail::block_name(block, "post_attention_norm.weight"),
            {options.hidden_size});
    add_main(detail::block_name(block, "ffn_gate.weight"),
             {options.hidden_size, options.intermediate_size});
    add_main(detail::block_name(block, "ffn_down.weight"),
             {options.intermediate_size, options.hidden_size});
    add_main(detail::block_name(block, "ffn_up.weight"),
             {options.hidden_size, options.intermediate_size});

    if (block + 1 == options.block_count) {
      add_main(detail::block_name(block, "attn_q.weight"),
               {options.hidden_size,
                options.full_head_count * options.full_head_dimension * 2});
      add_main(detail::block_name(block, "attn_k.weight"),
               {options.hidden_size,
                options.full_kv_head_count * options.full_head_dimension});
      add_main(detail::block_name(block, "attn_v.weight"),
               {options.hidden_size,
                options.full_kv_head_count * options.full_head_dimension});
      add_main(detail::block_name(block, "attn_output.weight"),
               {options.full_head_count * options.full_head_dimension,
                options.hidden_size});
      add_aux(detail::block_name(block, "attn_q_norm.weight"),
              {options.full_head_dimension});
      add_aux(detail::block_name(block, "attn_k_norm.weight"),
              {options.full_head_dimension});
    } else {
      add_main(detail::block_name(block, "attn_qkv.weight"),
               {options.hidden_size, linear_qkv_width});
      add_main(detail::block_name(block, "attn_gate.weight"),
               {options.hidden_size, linear_value_width});
      add_aux(detail::block_name(block, "ssm_conv1d.weight"),
              {4, linear_qkv_width});
      add_aux(detail::block_name(block, "ssm_dt.bias"),
              {options.linear_value_head_count});
      add_aux(detail::block_name(block, "ssm_a"),
              {options.linear_value_head_count});
      add_main(detail::block_name(block, "ssm_beta.weight"),
               {options.hidden_size, options.linear_value_head_count});
      add_main(detail::block_name(block, "ssm_alpha.weight"),
               {options.hidden_size, options.linear_value_head_count});
      add_aux(detail::block_name(block, "ssm_norm.weight"),
              {options.linear_head_dimension});
      add_main(detail::block_name(block, "ssm_out.weight"),
               {linear_value_width, options.hidden_size});
    }
  }

  std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
  detail::append<std::uint32_t>(bytes, 3);
  detail::append<std::uint64_t>(bytes, tensors.size());
  detail::append<std::uint64_t>(bytes, metadata_count);
  bytes.insert(bytes.end(), metadata.begin(), metadata.end());
  for (const auto &tensor : tensors) {
    detail::append_string(bytes, tensor.name);
    detail::append<std::uint32_t>(bytes, tensor.dimensions.size());
    for (const auto dimension : tensor.dimensions) {
      detail::append(bytes, dimension);
    }
    detail::append<std::uint32_t>(bytes, tensor.type);
    detail::append(bytes, tensor.offset);
  }
  while (bytes.size() % 32 != 0) {
    bytes.push_back(0);
  }
  bytes.resize(bytes.size() + next_offset);
  return bytes;
}

inline std::vector<std::uint8_t>
make_qwen35_mixed_f32_aux_gguf_fixture(std::uint32_t primary_tensor_type = 1) {
  return make_qwen35_gguf_fixture(primary_tensor_type, true);
}

} // namespace brt::test
