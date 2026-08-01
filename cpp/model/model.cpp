#include "model.hpp"

#include "gguf_reader.hpp"
#include "qwen35_config.hpp"
#include "qwen35_manifest.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace raftinfer::model {
namespace {

bool checked_add_u64(std::uint64_t left, std::uint64_t right,
                     std::uint64_t &result) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

class MappedFile {
public:
  explicit MappedFile(const std::string &path) {
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
      throw ModelIoError("cannot open GGUF file " + path + ": " +
                         std::strerror(errno));
    }

    struct stat status{};
    if (fstat(descriptor, &status) != 0) {
      const std::string message = std::strerror(errno);
      close(descriptor);
      throw ModelIoError("cannot inspect GGUF file " + path + ": " + message);
    }
    if (!S_ISREG(status.st_mode) || status.st_size <= 0) {
      close(descriptor);
      throw ModelIoError("GGUF path is not a nonempty regular file: " + path);
    }
    if (static_cast<std::uintmax_t>(status.st_size) >
        std::numeric_limits<std::size_t>::max()) {
      close(descriptor);
      throw ModelIoError("GGUF file is too large for this platform: " + path);
    }

    size_ = static_cast<std::size_t>(status.st_size);
    void *mapping = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor, 0);
    const int mapping_error = errno;
    close(descriptor);
    if (mapping == MAP_FAILED) {
      size_ = 0;
      throw ModelIoError("cannot map GGUF file " + path + ": " +
                         std::strerror(mapping_error));
    }
    data_ = static_cast<const std::uint8_t *>(mapping);
  }

  ~MappedFile() {
    if (data_ != nullptr) {
      munmap(const_cast<std::uint8_t *>(data_), size_);
    }
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  std::span<const std::uint8_t> bytes() const noexcept {
    return {data_, size_};
  }

private:
  const std::uint8_t *data_{};
  std::size_t size_{};
};

template <class T>
void append_integer(std::vector<std::uint8_t> &bytes, T value) {
  using Unsigned = std::make_unsigned_t<T>;
  const auto raw = static_cast<Unsigned>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    bytes.push_back(
        static_cast<std::uint8_t>(raw >> static_cast<unsigned>(index * 8)));
  }
}

void append_bytes(std::vector<std::uint8_t> &bytes, const std::string &value) {
  if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw ConfigError("tokenizer metadata string exceeds the ABI limit");
  }
  append_integer<std::uint32_t>(bytes,
                                static_cast<std::uint32_t>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void append_scalar(std::vector<std::uint8_t> &bytes, gguf::MetadataType type,
                   const gguf::MetadataValue &value) {
  switch (type) {
  case gguf::MetadataType::uint8:
    append_integer(bytes, value.get<std::uint8_t>());
    return;
  case gguf::MetadataType::int8:
    append_integer(bytes, value.get<std::int8_t>());
    return;
  case gguf::MetadataType::uint16:
    append_integer(bytes, value.get<std::uint16_t>());
    return;
  case gguf::MetadataType::int16:
    append_integer(bytes, value.get<std::int16_t>());
    return;
  case gguf::MetadataType::uint32:
    append_integer(bytes, value.get<std::uint32_t>());
    return;
  case gguf::MetadataType::int32:
    append_integer(bytes, value.get<std::int32_t>());
    return;
  case gguf::MetadataType::float32:
    append_integer(bytes, std::bit_cast<std::uint32_t>(value.get<float>()));
    return;
  case gguf::MetadataType::boolean:
    append_integer<std::uint8_t>(bytes, value.get<bool>() ? 1 : 0);
    return;
  case gguf::MetadataType::string:
    append_bytes(bytes, value.get<std::string>());
    return;
  case gguf::MetadataType::uint64:
    append_integer(bytes, value.get<std::uint64_t>());
    return;
  case gguf::MetadataType::int64:
    append_integer(bytes, value.get<std::int64_t>());
    return;
  case gguf::MetadataType::float64:
    append_integer(bytes, std::bit_cast<std::uint64_t>(value.get<double>()));
    return;
  case gguf::MetadataType::array:
    break;
  }
  throw ConfigError("nested tokenizer metadata arrays are unsupported");
}

gguf::MetadataType value_type(const gguf::MetadataValue &value) {
#define RAFTINFER_GGUF_VALUE_TYPE(cpp_type, metadata_type)                           \
  if (value.get_if<cpp_type>() != nullptr) {                                   \
    return gguf::MetadataType::metadata_type;                                  \
  }
  RAFTINFER_GGUF_VALUE_TYPE(std::uint8_t, uint8)
  RAFTINFER_GGUF_VALUE_TYPE(std::int8_t, int8)
  RAFTINFER_GGUF_VALUE_TYPE(std::uint16_t, uint16)
  RAFTINFER_GGUF_VALUE_TYPE(std::int16_t, int16)
  RAFTINFER_GGUF_VALUE_TYPE(std::uint32_t, uint32)
  RAFTINFER_GGUF_VALUE_TYPE(std::int32_t, int32)
  RAFTINFER_GGUF_VALUE_TYPE(float, float32)
  RAFTINFER_GGUF_VALUE_TYPE(bool, boolean)
  RAFTINFER_GGUF_VALUE_TYPE(std::string, string)
  RAFTINFER_GGUF_VALUE_TYPE(gguf::MetadataArray, array)
  RAFTINFER_GGUF_VALUE_TYPE(std::uint64_t, uint64)
  RAFTINFER_GGUF_VALUE_TYPE(std::int64_t, int64)
  RAFTINFER_GGUF_VALUE_TYPE(double, float64)
#undef RAFTINFER_GGUF_VALUE_TYPE
  throw ConfigError("tokenizer metadata has an unknown value type");
}

void append_value(std::vector<std::uint8_t> &bytes,
                  const gguf::MetadataValue &value) {
  const auto type = value_type(value);
  append_integer(bytes, static_cast<std::uint32_t>(type));
  if (type != gguf::MetadataType::array) {
    append_scalar(bytes, type, value);
    return;
  }

  const auto &array = value.get<gguf::MetadataArray>();
  append_integer(bytes, static_cast<std::uint32_t>(array.element_type));
  append_integer<std::uint64_t>(bytes, array.values.size());
  for (const auto &element : array.values) {
    append_scalar(bytes, array.element_type, element);
  }
}

std::vector<std::uint8_t>
serialize_tokenizer_spec(const gguf::Catalog &catalog) {
  std::vector<std::pair<std::string, const gguf::MetadataValue *>> entries;
  for (const auto &[key, value] : catalog.metadata) {
    if (key.starts_with("tokenizer.")) {
      entries.emplace_back(key, &value);
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto &left, const auto &right) {
              return left.first < right.first;
            });
  if (entries.empty() ||
      entries.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw ConfigError("Qwen3.5 tokenizer metadata is missing or too large");
  }

  std::vector<std::uint8_t> result{'R', 'I', 'F', 'T', 'O', 'K', 0, 1};
  append_integer<std::uint32_t>(result,
                                static_cast<std::uint32_t>(entries.size()));
  for (const auto &[key, value] : entries) {
    append_bytes(result, key);
    append_value(result, *value);
  }
  return result;
}

} // namespace

class Model::Impl {
public:
  explicit Impl(const std::string &gguf_path)
      : file(gguf_path), catalog(gguf::read_catalog(file.bytes())),
        config(derive_qwen35_config(catalog)),
        manifest(validate_qwen35_manifest(catalog, config)),
        tokenizer_spec_bytes(serialize_tokenizer_spec(catalog)) {}

  std::span<const std::uint8_t>
  tensor_payload(const gguf::TensorInfo &tensor) const {
    const auto *catalog_tensor = catalog.find_tensor(tensor.name);
    if (catalog_tensor == nullptr ||
        catalog_tensor->dimensions != tensor.dimensions ||
        catalog_tensor->type != tensor.type ||
        catalog_tensor->offset != tensor.offset ||
        catalog_tensor->byte_size != tensor.byte_size) {
      throw ModelIoError("GGUF tensor descriptor does not belong to model: " +
                         tensor.name);
    }
    const auto bytes = file.bytes();
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    if (!checked_add_u64(catalog.tensor_data_offset, tensor.offset, begin) ||
        !checked_add_u64(begin, tensor.byte_size, end) || end > bytes.size()) {
      throw ModelIoError("GGUF tensor payload is outside the mapped file: " +
                         tensor.name);
    }
    if (begin > std::numeric_limits<std::size_t>::max() ||
        tensor.byte_size > std::numeric_limits<std::size_t>::max()) {
      throw ModelIoError(
          "GGUF tensor payload is too large for this platform: " + tensor.name);
    }
    return bytes.subspan(static_cast<std::size_t>(begin),
                         static_cast<std::size_t>(tensor.byte_size));
  }

  struct CudaState {
    std::shared_ptr<DeviceContext> device;
    std::shared_ptr<CudaWeightPlan> weights;
  };

  MappedFile file;
  gguf::Catalog catalog;
  Qwen35Config config;
  Qwen35Manifest manifest;
  std::vector<std::uint8_t> tokenizer_spec_bytes;
  std::unique_ptr<CudaState> cuda_state;
};

Model::Model(const std::string &gguf_path)
    : impl_(std::make_unique<Impl>(gguf_path)) {}

Model::~Model() = default;

std::span<const std::uint8_t> Model::tokenizer_spec() const noexcept {
  return impl_->tokenizer_spec_bytes;
}

const Qwen35Config &Model::qwen35_config() const noexcept {
  return impl_->config;
}

const Qwen35Manifest &Model::qwen35_manifest() const noexcept {
  return impl_->manifest;
}

std::span<const std::uint8_t>
Model::tensor_payload(const gguf::TensorInfo &tensor) const {
  return impl_->tensor_payload(tensor);
}

bool Model::cuda_ready() const noexcept {
  return impl_->cuda_state != nullptr;
}

const CudaWeightPlan *Model::cuda_weights() const noexcept {
  return impl_->cuda_state == nullptr ? nullptr
                                      : impl_->cuda_state->weights.get();
}

std::shared_ptr<const DeviceContext> Model::device_context() const noexcept {
  return impl_->cuda_state == nullptr ? std::shared_ptr<const DeviceContext>{}
                                      : impl_->cuda_state->device;
}

void Model::attach_cuda(std::shared_ptr<DeviceContext> device,
                        std::shared_ptr<CudaWeightPlan> weights) {
  if (!device || !weights) {
    throw std::invalid_argument(
        "CUDA model attachment requires device and weight ownership");
  }
  if (impl_->cuda_state != nullptr) {
    throw std::logic_error("CUDA model attachment is immutable");
  }
  auto cuda_state = std::make_unique<Impl::CudaState>(
      Impl::CudaState{std::move(device), std::move(weights)});
  impl_->cuda_state = std::move(cuda_state);
}

} // namespace raftinfer::model
