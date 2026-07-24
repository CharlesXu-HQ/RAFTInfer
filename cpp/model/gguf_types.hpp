#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace brt::gguf {

enum class MetadataType : std::uint32_t {
  uint8 = 0,
  int8 = 1,
  uint16 = 2,
  int16 = 3,
  uint32 = 4,
  int32 = 5,
  float32 = 6,
  boolean = 7,
  string = 8,
  array = 9,
  uint64 = 10,
  int64 = 11,
  float64 = 12,
};

struct MetadataArray;

class MetadataValue {
public:
  using Storage =
      std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
                   std::uint32_t, std::int32_t, float, bool, std::string,
                   std::shared_ptr<MetadataArray>, std::uint64_t, std::int64_t,
                   double>;

  MetadataValue(const MetadataValue &) = default;
  MetadataValue(MetadataValue &&) noexcept = default;
  MetadataValue &operator=(const MetadataValue &) = default;
  MetadataValue &operator=(MetadataValue &&) noexcept = default;

  template <class T>
    requires(!std::is_same_v<std::remove_cvref_t<T>, MetadataArray> &&
             !std::is_same_v<std::remove_cvref_t<T>, MetadataValue>)
  explicit MetadataValue(T &&value) : storage_(std::forward<T>(value)) {}

  explicit MetadataValue(MetadataArray value);

  template <class T> const T &get() const {
    if constexpr (std::is_same_v<T, MetadataArray>) {
      return *std::get<std::shared_ptr<MetadataArray>>(storage_);
    } else {
      return std::get<T>(storage_);
    }
  }

  template <class T> const T *get_if() const {
    if constexpr (std::is_same_v<T, MetadataArray>) {
      const auto *value =
          std::get_if<std::shared_ptr<MetadataArray>>(&storage_);
      return value == nullptr ? nullptr : value->get();
    } else {
      return std::get_if<T>(&storage_);
    }
  }

private:
  Storage storage_;
};

struct MetadataArray {
  MetadataType element_type;
  std::vector<MetadataValue> values;
};

inline MetadataValue::MetadataValue(MetadataArray value)
    : storage_(std::make_shared<MetadataArray>(std::move(value))) {}

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t type{};
  std::uint64_t offset{};
  std::uint64_t byte_size{};
};

struct Catalog {
  std::uint32_t version{};
  std::uint32_t alignment{32};
  std::uint64_t tensor_data_offset{};
  std::unordered_map<std::string, MetadataValue> metadata;
  std::vector<TensorInfo> tensors;

  const MetadataValue *find_metadata(const std::string &key) const {
    const auto it = metadata.find(key);
    return it == metadata.end() ? nullptr : &it->second;
  }

  const MetadataValue &require_metadata(const std::string &key) const {
    const auto *value = find_metadata(key);
    if (value == nullptr) {
      throw std::out_of_range("missing GGUF metadata: " + key);
    }
    return *value;
  }

  const TensorInfo *find_tensor(const std::string &name) const {
    for (const auto &tensor : tensors) {
      if (tensor.name == name) {
        return &tensor;
      }
    }
    return nullptr;
  }

  const TensorInfo &require_tensor(const std::string &name) const {
    const auto *tensor = find_tensor(name);
    if (tensor == nullptr) {
      throw std::out_of_range("missing GGUF tensor: " + name);
    }
    return *tensor;
  }
};

struct ReaderLimits {
  std::uint64_t max_metadata_count{1'000'000};
  std::uint64_t max_tensor_count{1'000'000};
  std::uint64_t max_string_bytes{64 * 1024 * 1024};
  std::uint64_t max_array_elements{16 * 1024 * 1024};
  std::uint32_t max_tensor_rank{4};
};

} // namespace brt::gguf
