#include "../model/gguf_reader.hpp"
#include "../model/gguf_types.hpp"
#include "../model/model.hpp"

#include "assert_enabled.hpp"
#include "qwen35_gguf_fixture.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace {

template <class T> void append(std::vector<std::uint8_t> &bytes, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto *source = reinterpret_cast<const std::uint8_t *>(&value);
  bytes.insert(bytes.end(), source, source + sizeof(value));
}

void append_string(std::vector<std::uint8_t> &bytes, const std::string &value) {
  append<std::uint64_t>(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void append_string_metadata(std::vector<std::uint8_t> &bytes,
                            const std::string &key, const std::string &value) {
  append_string(bytes, key);
  append<std::uint32_t>(
      bytes, static_cast<std::uint32_t>(raftinfer::gguf::MetadataType::string));
  append_string(bytes, value);
}

void append_u32_metadata(std::vector<std::uint8_t> &bytes,
                         const std::string &key, std::uint32_t value) {
  append_string(bytes, key);
  append<std::uint32_t>(
      bytes, static_cast<std::uint32_t>(raftinfer::gguf::MetadataType::uint32));
  append<std::uint32_t>(bytes, value);
}

void append_string_array_metadata(std::vector<std::uint8_t> &bytes,
                                  const std::string &key,
                                  const std::vector<std::string> &values) {
  append_string(bytes, key);
  append<std::uint32_t>(
      bytes, static_cast<std::uint32_t>(raftinfer::gguf::MetadataType::array));
  append<std::uint32_t>(
      bytes, static_cast<std::uint32_t>(raftinfer::gguf::MetadataType::string));
  append<std::uint64_t>(bytes, values.size());
  for (const auto &value : values) {
    append_string(bytes, value);
  }
}

std::vector<std::uint8_t> valid_gguf() {
  std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
  append<std::uint32_t>(bytes, 3);
  append<std::uint64_t>(bytes, 1);
  append<std::uint64_t>(bytes, 4);
  append_string_metadata(bytes, "general.architecture", "qwen35");
  append_u32_metadata(bytes, "general.alignment", 32);
  append_u32_metadata(bytes, "qwen35.block_count", 32);
  append_string_array_metadata(bytes, "qwen35.layer_types",
                               {"linear_attention", "linear_attention",
                                "linear_attention", "full_attention"});

  append_string(bytes, "token_embd.weight");
  append<std::uint32_t>(bytes, 2);
  append<std::uint64_t>(bytes, 2);
  append<std::uint64_t>(bytes, 2);
  append<std::uint32_t>(bytes, 1);
  append<std::uint64_t>(bytes, 0);

  while (bytes.size() % 32 != 0) {
    bytes.push_back(0);
  }
  bytes.resize(bytes.size() + 8);
  return bytes;
}

template <class Fn> void expect_parse_error(Fn &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const raftinfer::gguf::ParseError &) {
    thrown = true;
  }
  assert(thrown);
}

template <class Fn> void expect_model_io_error(Fn &&fn) {
  bool thrown = false;
  try {
    fn();
  } catch (const raftinfer::model::ModelIoError &) {
    thrown = true;
  }
  assert(thrown);
}

} // namespace

int main() {
  using raftinfer::gguf::MetadataArray;
  using raftinfer::gguf::MetadataValue;

  MetadataValue name{std::string{"Qwen3.5-9B"}};
  assert(name.get<std::string>() == "Qwen3.5-9B");
  assert(name.get_if<std::uint32_t>() == nullptr);

  MetadataArray layers{
      .element_type = raftinfer::gguf::MetadataType::string,
      .values = {MetadataValue{std::string{"linear_attention"}},
                 MetadataValue{std::string{"full_attention"}}},
  };
  MetadataValue layer_types{layers};
  const auto &stored_layers = layer_types.get<MetadataArray>();
  assert(stored_layers.values.size() == 2);
  assert(stored_layers.values[1].get<std::string>() == "full_attention");

  raftinfer::gguf::Catalog catalog;
  catalog.metadata.emplace("general.architecture",
                           MetadataValue{std::string{"qwen35"}});
  catalog.tensors.push_back(raftinfer::gguf::TensorInfo{
      .name = "token_embd.weight",
      .dimensions = {4096, 248320},
      .type = 1,
      .offset = 0,
  });

  assert(catalog.require_metadata("general.architecture").get<std::string>() ==
         "qwen35");
  assert(catalog.find_metadata("missing") == nullptr);
  assert(catalog.require_tensor("token_embd.weight").dimensions[0] == 4096);
  assert(catalog.find_tensor("missing") == nullptr);

  const auto bytes = valid_gguf();
  const auto parsed = raftinfer::gguf::read_catalog(bytes);
  assert(parsed.version == 3);
  assert(parsed.alignment == 32);
  assert(parsed.tensor_data_offset % 32 == 0);
  assert(parsed.require_metadata("general.architecture").get<std::string>() ==
         "qwen35");
  const auto &parsed_layers =
      parsed.require_metadata("qwen35.layer_types").get<MetadataArray>();
  assert(parsed_layers.element_type == raftinfer::gguf::MetadataType::string);
  assert(parsed_layers.values.size() == 4);
  assert(parsed.require_tensor("token_embd.weight").dimensions ==
         std::vector<std::uint64_t>({2, 2}));
  assert(parsed.require_tensor("token_embd.weight").byte_size == 8);

  raftinfer::gguf::ReaderLimits tiny_catalog_limit;
  tiny_catalog_limit.max_catalog_bytes = 16;
  expect_parse_error([&] {
    (void)raftinfer::gguf::read_catalog(std::span{bytes}, tiny_catalog_limit);
  });

  const auto model_bytes = raftinfer::test::make_qwen35_gguf_fixture();
  const auto model_catalog = raftinfer::gguf::read_catalog(std::span{model_bytes});
  const auto temporary_path =
      std::filesystem::temp_directory_path() /
      ("raftinfer_gguf_reader_test_" + std::to_string(getpid()) + ".gguf");
  {
    std::ofstream output{temporary_path, std::ios::binary};
    output.write(reinterpret_cast<const char *>(model_bytes.data()),
                 static_cast<std::streamsize>(model_bytes.size()));
  }
  raftinfer::model::Model model{temporary_path.string()};
  assert(model.tensor_payload(model_catalog.require_tensor("token_embd.weight"))
             .size() == 256);

  auto outside = model_catalog.require_tensor("token_embd.weight");
  outside.offset = model_bytes.size();
  expect_model_io_error([&] { (void)model.tensor_payload(outside); });
  std::filesystem::remove(temporary_path);

  auto bad_magic = bytes;
  bad_magic[0] = 'X';
  expect_parse_error(
      [&] { (void)raftinfer::gguf::read_catalog(std::span{bad_magic}); });

  auto bad_version = bytes;
  bad_version[4] = 2;
  expect_parse_error(
      [&] { (void)raftinfer::gguf::read_catalog(std::span{bad_version}); });

  auto invalid_metadata_key = bytes;
  const std::string architecture_key = "general.architecture";
  const auto key_position =
      std::search(invalid_metadata_key.begin(), invalid_metadata_key.end(),
                  architecture_key.begin(), architecture_key.end());
  assert(key_position != invalid_metadata_key.end());
  *key_position = 'G';
  expect_parse_error(
      [&] { (void)raftinfer::gguf::read_catalog(std::span{invalid_metadata_key}); });

  for (std::size_t size = 0; size < 24; ++size) {
    expect_parse_error([&] {
      (void)raftinfer::gguf::read_catalog(
          std::span<const std::uint8_t>{bytes.data(), size});
    });
  }
}
