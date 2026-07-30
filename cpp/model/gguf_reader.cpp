#include "gguf_reader.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace raftinfer::gguf {
namespace {

template <class T> bool checked_add(T left, T right, T &result) {
  if (right > std::numeric_limits<T>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

template <class T> bool checked_multiply(T left, T right, T &result) {
  if (left != 0 && right > std::numeric_limits<T>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

class Reader {
public:
  Reader(std::span<const std::uint8_t> bytes, const ReaderLimits &limits)
      : bytes_(bytes), limits_(limits) {}

  std::size_t position() const { return position_; }

  std::uint8_t read_u8(const std::string &context) {
    require(1, context);
    return bytes_[position_++];
  }

  std::uint16_t read_u16(const std::string &context) {
    return read_unsigned<std::uint16_t>(context);
  }

  std::uint32_t read_u32(const std::string &context) {
    return read_unsigned<std::uint32_t>(context);
  }

  std::uint64_t read_u64(const std::string &context) {
    return read_unsigned<std::uint64_t>(context);
  }

  std::string read_string(const std::string &context) {
    const std::uint64_t length = read_u64(context + " length");
    if (length > limits_.max_string_bytes) {
      throw ParseError(context + " exceeds the GGUF string limit");
    }
    if (length > std::numeric_limits<std::size_t>::max()) {
      throw ParseError(context + " length does not fit this platform");
    }
    const auto size = static_cast<std::size_t>(length);
    require(size, context);
    charge_catalog_bytes(length, context + " storage");
    const char *first =
        reinterpret_cast<const char *>(bytes_.data() + position_);
    std::string result(first, size);
    position_ += size;
    return result;
  }

  MetadataValue read_value(MetadataType type, const std::string &context,
                           bool allow_array = true) {
    switch (type) {
    case MetadataType::uint8:
      charge_catalog_bytes(sizeof(std::uint8_t), context + " storage");
      return MetadataValue{read_u8(context)};
    case MetadataType::int8:
      charge_catalog_bytes(sizeof(std::int8_t), context + " storage");
      return MetadataValue{std::bit_cast<std::int8_t>(read_u8(context))};
    case MetadataType::uint16:
      charge_catalog_bytes(sizeof(std::uint16_t), context + " storage");
      return MetadataValue{read_u16(context)};
    case MetadataType::int16:
      charge_catalog_bytes(sizeof(std::int16_t), context + " storage");
      return MetadataValue{std::bit_cast<std::int16_t>(read_u16(context))};
    case MetadataType::uint32:
      charge_catalog_bytes(sizeof(std::uint32_t), context + " storage");
      return MetadataValue{read_u32(context)};
    case MetadataType::int32:
      charge_catalog_bytes(sizeof(std::int32_t), context + " storage");
      return MetadataValue{std::bit_cast<std::int32_t>(read_u32(context))};
    case MetadataType::float32:
      charge_catalog_bytes(sizeof(float), context + " storage");
      return MetadataValue{std::bit_cast<float>(read_u32(context))};
    case MetadataType::boolean: {
      charge_catalog_bytes(sizeof(bool), context + " storage");
      const auto value = read_u8(context);
      if (value > 1) {
        throw ParseError(context + " has an invalid boolean value");
      }
      return MetadataValue{value != 0};
    }
    case MetadataType::string:
      return MetadataValue{read_string(context)};
    case MetadataType::array:
      if (!allow_array) {
        throw ParseError(context + " contains a nested array");
      }
      return read_array(context);
    case MetadataType::uint64:
      charge_catalog_bytes(sizeof(std::uint64_t), context + " storage");
      return MetadataValue{read_u64(context)};
    case MetadataType::int64:
      charge_catalog_bytes(sizeof(std::int64_t), context + " storage");
      return MetadataValue{std::bit_cast<std::int64_t>(read_u64(context))};
    case MetadataType::float64:
      charge_catalog_bytes(sizeof(double), context + " storage");
      return MetadataValue{std::bit_cast<double>(read_u64(context))};
    }
    throw ParseError(context + " has an unsupported metadata type");
  }

  void charge_tensor_dimensions(std::uint32_t rank, const std::string &name) {
    std::uint64_t storage = 0;
    if (!checked_multiply<std::uint64_t>(
            rank, static_cast<std::uint64_t>(sizeof(std::uint64_t)), storage)) {
      throw ParseError("tensor " + name + " dimension storage overflows");
    }
    charge_catalog_bytes(storage, "tensor " + name + " dimension storage");
  }

private:
  template <class T> T read_unsigned(const std::string &context) {
    require(sizeof(T), context);
    T value = 0;
    for (std::size_t index = 0; index < sizeof(T); ++index) {
      value |= static_cast<T>(bytes_[position_ + index])
               << static_cast<unsigned>(index * 8);
    }
    position_ += sizeof(T);
    return value;
  }

  MetadataValue read_array(const std::string &context) {
    const auto raw_type = read_u32(context + " element type");
    const auto element_type = metadata_type(raw_type, context + " element");
    if (element_type == MetadataType::array) {
      throw ParseError(context + " contains a nested array");
    }
    const auto length = read_u64(context + " length");
    if (length > limits_.max_array_elements) {
      throw ParseError(context + " exceeds the GGUF array limit");
    }
    std::uint64_t element_storage = 0;
    if (!checked_multiply<std::uint64_t>(
            length, static_cast<std::uint64_t>(sizeof(MetadataValue)),
            element_storage)) {
      throw ParseError(context + " catalog storage overflows");
    }
    charge_catalog_bytes(element_storage, context + " storage");
    MetadataArray result{.element_type = element_type, .values = {}};
    result.values.reserve(static_cast<std::size_t>(length));
    for (std::uint64_t index = 0; index < length; ++index) {
      result.values.push_back(
          read_value(element_type, context + " element", false));
    }
    return MetadataValue{std::move(result)};
  }

  static MetadataType metadata_type(std::uint32_t raw,
                                    const std::string &context) {
    if (raw > static_cast<std::uint32_t>(MetadataType::float64)) {
      throw ParseError(context + " has an unknown metadata type");
    }
    return static_cast<MetadataType>(raw);
  }

  void require(std::size_t count, const std::string &context) const {
    if (count > bytes_.size() - position_) {
      throw ParseError("truncated GGUF while reading " + context);
    }
  }

  void charge_catalog_bytes(std::uint64_t bytes, const std::string &context) {
    std::uint64_t charged = 0;
    if (!checked_add(catalog_bytes_, bytes, charged)) {
      throw ParseError(context + " overflows the GGUF catalog byte counter");
    }
    if (charged > limits_.max_catalog_bytes) {
      throw ParseError(context + " exceeds the GGUF catalog byte limit");
    }
    catalog_bytes_ = charged;
  }

  std::span<const std::uint8_t> bytes_;
  const ReaderLimits &limits_;
  std::size_t position_{};
  std::uint64_t catalog_bytes_{};
};

MetadataType metadata_type(std::uint32_t raw, const std::string &context) {
  if (raw > static_cast<std::uint32_t>(MetadataType::float64)) {
    throw ParseError(context + " has an unknown metadata type");
  }
  return static_cast<MetadataType>(raw);
}

std::uint64_t tensor_byte_size(std::uint32_t type,
                               const std::vector<std::uint64_t> &dimensions,
                               const std::string &name) {
  std::uint64_t element_size = 0;
  switch (type) {
  case 0:
    element_size = 4;
    break;
  case 1:
  case 30:
    element_size = 2;
    break;
  default:
    throw ParseError("tensor " + name +
                     " uses a type unsupported by the M2A baseline");
  }

  std::uint64_t elements = 1;
  for (const auto dimension : dimensions) {
    if (dimension == 0 || !checked_multiply(elements, dimension, elements)) {
      throw ParseError("tensor " + name + " has invalid dimensions");
    }
  }
  std::uint64_t bytes = 0;
  if (!checked_multiply(elements, element_size, bytes)) {
    throw ParseError("tensor " + name + " byte size overflows");
  }
  return bytes;
}

std::uint32_t catalog_alignment(const Catalog &catalog) {
  const auto *value = catalog.find_metadata("general.alignment");
  if (value == nullptr) {
    return 32;
  }
  std::uint64_t alignment = 0;
  if (const auto *typed = value->get_if<std::uint32_t>()) {
    alignment = *typed;
  } else if (const auto *typed = value->get_if<std::uint64_t>()) {
    alignment = *typed;
  } else {
    throw ParseError("general.alignment must be an unsigned integer");
  }
  if (alignment == 0 || alignment % 8 != 0 ||
      alignment > std::numeric_limits<std::uint32_t>::max()) {
    throw ParseError("general.alignment must be a nonzero multiple of eight");
  }
  return static_cast<std::uint32_t>(alignment);
}

bool valid_metadata_key(const std::string &key) {
  if (key.empty() || key.size() > 65535) {
    return false;
  }
  bool segment_start = true;
  for (const unsigned char character : key) {
    if (character == '.') {
      if (segment_start) {
        return false;
      }
      segment_start = true;
      continue;
    }
    const bool lowercase = character >= 'a' && character <= 'z';
    const bool digit = character >= '0' && character <= '9';
    if (segment_start) {
      if (!lowercase && !digit) {
        return false;
      }
      segment_start = false;
    } else if (!lowercase && !digit && character != '_') {
      return false;
    }
  }
  return !segment_start;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  const std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  std::uint64_t result = 0;
  if (!checked_add(value, alignment - remainder, result)) {
    throw ParseError("GGUF tensor data offset overflows");
  }
  return result;
}

} // namespace

Catalog read_catalog(std::span<const std::uint8_t> bytes,
                     const ReaderLimits &limits) {
  Reader reader(bytes, limits);
  if (reader.read_u8("magic") != 'G' || reader.read_u8("magic") != 'G' ||
      reader.read_u8("magic") != 'U' || reader.read_u8("magic") != 'F') {
    throw ParseError("invalid GGUF magic");
  }

  Catalog catalog;
  catalog.version = reader.read_u32("version");
  if (catalog.version != 3) {
    throw ParseError("unsupported GGUF version");
  }

  const auto tensor_count = reader.read_u64("tensor count");
  const auto metadata_count = reader.read_u64("metadata count");
  if (tensor_count > limits.max_tensor_count) {
    throw ParseError("GGUF tensor count exceeds the configured limit");
  }
  if (metadata_count > limits.max_metadata_count) {
    throw ParseError("GGUF metadata count exceeds the configured limit");
  }

  catalog.metadata.reserve(static_cast<std::size_t>(metadata_count));
  for (std::uint64_t index = 0; index < metadata_count; ++index) {
    std::string key = reader.read_string("metadata key");
    if (!valid_metadata_key(key)) {
      throw ParseError("invalid GGUF metadata key: " + key);
    }
    const auto type =
        metadata_type(reader.read_u32("metadata type"), "metadata " + key);
    MetadataValue value = reader.read_value(type, "metadata " + key);
    if (!catalog.metadata.emplace(key, std::move(value)).second) {
      throw ParseError("duplicate GGUF metadata key: " + key);
    }
  }

  catalog.alignment = catalog_alignment(catalog);
  catalog.tensors.reserve(static_cast<std::size_t>(tensor_count));
  std::unordered_set<std::string> tensor_names;
  for (std::uint64_t index = 0; index < tensor_count; ++index) {
    TensorInfo tensor;
    tensor.name = reader.read_string("tensor name");
    if (!tensor_names.insert(tensor.name).second) {
      throw ParseError("duplicate GGUF tensor name: " + tensor.name);
    }
    const auto rank = reader.read_u32("tensor rank");
    if (rank == 0 || rank > limits.max_tensor_rank) {
      throw ParseError("tensor " + tensor.name + " has an invalid rank");
    }
    reader.charge_tensor_dimensions(rank, tensor.name);
    tensor.dimensions.reserve(rank);
    for (std::uint32_t dimension = 0; dimension < rank; ++dimension) {
      tensor.dimensions.push_back(reader.read_u64("tensor dimension"));
    }
    tensor.type = reader.read_u32("tensor type");
    tensor.offset = reader.read_u64("tensor offset");
    tensor.byte_size =
        tensor_byte_size(tensor.type, tensor.dimensions, tensor.name);
    catalog.tensors.push_back(std::move(tensor));
  }

  catalog.tensor_data_offset = align_up(
      static_cast<std::uint64_t>(reader.position()), catalog.alignment);
  if (catalog.tensor_data_offset > bytes.size()) {
    throw ParseError("GGUF tensor data offset is outside the file");
  }

  struct Span {
    std::uint64_t begin;
    std::uint64_t end;
    std::string name;
  };
  std::vector<Span> spans;
  spans.reserve(catalog.tensors.size());
  for (const auto &tensor : catalog.tensors) {
    if (tensor.offset % catalog.alignment != 0) {
      throw ParseError("tensor " + tensor.name + " is not aligned");
    }
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    if (!checked_add(catalog.tensor_data_offset, tensor.offset, begin) ||
        !checked_add(begin, tensor.byte_size, end) || end > bytes.size()) {
      throw ParseError("tensor " + tensor.name + " is outside the file");
    }
    spans.push_back(Span{begin, end, tensor.name});
  }
  std::sort(spans.begin(), spans.end(),
            [](const Span &left, const Span &right) {
              return left.begin < right.begin;
            });
  for (std::size_t index = 1; index < spans.size(); ++index) {
    if (spans[index].begin < spans[index - 1].end) {
      throw ParseError("GGUF tensor spans overlap: " + spans[index - 1].name +
                       " and " + spans[index].name);
    }
  }

  return catalog;
}

} // namespace raftinfer::gguf
