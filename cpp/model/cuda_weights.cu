#include "cuda_weights.hpp"

#include "../execution/execution_context.hpp"
#include "model.hpp"

#include <cuda/stream_ref>
#include <cuda_runtime_api.h>
#include <rmm/aligned.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brt::model {
namespace {

class BestEffortDeviceGuard {
public:
  explicit BestEffortDeviceGuard(int device_id) noexcept {
    if (cudaGetDevice(&previous_device_) != cudaSuccess)
      return;
    captured_previous_ = true;
    if (cudaSetDevice(device_id) != cudaSuccess)
      return;
    selected_target_ = true;
  }

  ~BestEffortDeviceGuard() noexcept { restore(); }

  BestEffortDeviceGuard(const BestEffortDeviceGuard &) = delete;
  BestEffortDeviceGuard &operator=(const BestEffortDeviceGuard &) = delete;

  bool selected_target() const noexcept { return selected_target_; }

  void restore() noexcept {
    if (!captured_previous_)
      return;
    (void)cudaSetDevice(previous_device_);
    captured_previous_ = false;
  }

private:
  int previous_device_{};
  bool captured_previous_{};
  bool selected_target_{};
};

[[noreturn]] void throw_cuda_weight_error(cudaError_t error,
                                          const std::string &tensor_name) {
  throw CudaWeightError("CUDA weight upload failed for " + tensor_name + ": " +
                        cudaGetErrorString(error));
}

CudaWeightType checked_type(const gguf::TensorInfo &tensor) {
  if (tensor.type == static_cast<std::uint32_t>(CudaWeightType::f16)) {
    return CudaWeightType::f16;
  }
  if (tensor.type == static_cast<std::uint32_t>(CudaWeightType::bf16)) {
    return CudaWeightType::bf16;
  }
  throw CudaWeightError("unsupported Qwen3.5 M2 primary weight type for " +
                        tensor.name);
}

enum class TensorUploadRole {
  primary,
  auxiliary,
};

struct ManifestTensor {
  const gguf::TensorInfo *descriptor{};
  TensorUploadRole role{};
};

std::vector<ManifestTensor> manifest_tensors(const Qwen35Manifest &manifest) {
  std::vector<ManifestTensor> tensors;
  tensors.reserve(3 + manifest.layers.size() * 14);
  const auto add_primary = [&tensors](const gguf::TensorInfo *tensor) {
    tensors.push_back(ManifestTensor{
        .descriptor = tensor, .role = TensorUploadRole::primary});
  };
  const auto add_auxiliary = [&tensors](const gguf::TensorInfo *tensor) {
    tensors.push_back(ManifestTensor{
        .descriptor = tensor, .role = TensorUploadRole::auxiliary});
  };

  add_primary(manifest.token_embedding);
  add_auxiliary(manifest.output_norm);
  add_primary(manifest.output);
  for (std::size_t layer_position = 0; layer_position < manifest.layers.size();
       ++layer_position) {
    const auto &layer = manifest.layers[layer_position];
    if (layer.index != layer_position) {
      throw CudaWeightError("Qwen3.5 CUDA weight layers must be contiguous");
    }
    if (layer.full_attention.has_value() ==
        layer.linear_attention.has_value()) {
      throw CudaWeightError(
          "Qwen3.5 CUDA weight layer must have exactly one attention branch");
    }
    add_auxiliary(layer.common.input_norm);
    add_auxiliary(layer.common.post_attention_norm);
    add_primary(layer.common.ffn_gate);
    add_primary(layer.common.ffn_down);
    add_primary(layer.common.ffn_up);
    if (layer.full_attention.has_value()) {
      const auto &full = *layer.full_attention;
      add_primary(full.query);
      add_primary(full.key);
      add_primary(full.value);
      add_primary(full.output);
      add_auxiliary(full.query_norm);
      add_auxiliary(full.key_norm);
    }
    if (layer.linear_attention.has_value()) {
      const auto &linear = *layer.linear_attention;
      add_primary(linear.qkv);
      add_primary(linear.gate);
      add_auxiliary(linear.convolution);
      add_auxiliary(linear.time_step_bias);
      add_auxiliary(linear.recurrent_a);
      add_primary(linear.beta);
      add_primary(linear.alpha);
      add_auxiliary(linear.output_norm);
      add_primary(linear.output);
    }
  }
  if (std::any_of(tensors.begin(), tensors.end(),
                  [](const auto &tensor) {
                    return tensor.descriptor == nullptr;
                  })) {
    throw CudaWeightError("Qwen3.5 manifest contains a null primary tensor");
  }
  return tensors;
}

std::uint32_t round_shift_right(std::uint32_t value, unsigned shift) {
  if (shift == 0) {
    return value;
  }
  if (shift >= 32) {
    return 0;
  }
  const std::uint32_t quotient = value >> shift;
  const std::uint32_t remainder_mask = (std::uint32_t{1} << shift) - 1U;
  const std::uint32_t remainder = value & remainder_mask;
  const std::uint32_t halfway = std::uint32_t{1} << (shift - 1U);
  if (remainder > halfway || (remainder == halfway && (quotient & 1U) != 0)) {
    return quotient + 1U;
  }
  return quotient;
}

std::uint16_t float_to_f16_bits(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t exponent = (bits >> 23U) & 0xffU;
  const std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    if (mantissa == 0) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    std::uint16_t payload = static_cast<std::uint16_t>(mantissa >> 13U);
    if (payload == 0) {
      payload = 1;
    }
    return static_cast<std::uint16_t>(sign | 0x7c00U | payload);
  }

  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -24) {
      return sign;
    }
    const std::uint32_t rounded =
        round_shift_right(mantissa | 0x800000U,
                          static_cast<unsigned>(14 - half_exponent));
    return static_cast<std::uint16_t>(sign | rounded);
  }

  std::uint32_t rounded_mantissa = round_shift_right(mantissa, 13);
  std::uint32_t rounded_exponent = static_cast<std::uint32_t>(half_exponent);
  if (rounded_mantissa == 0x400U) {
    rounded_mantissa = 0;
    ++rounded_exponent;
    if (rounded_exponent >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
  }
  return static_cast<std::uint16_t>(sign | (rounded_exponent << 10U) |
                                    rounded_mantissa);
}

std::uint16_t float_to_bf16_bits(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7F800000U) == 0x7F800000U) {
    const std::uint16_t sign_and_exp =
        static_cast<std::uint16_t>((bits >> 16U) & 0xFF80U);
    if ((bits & 0x007FFFFFU) == 0) {
      return sign_and_exp;
    }
    std::uint16_t payload =
        static_cast<std::uint16_t>((bits >> 16U) & 0x007FU);
    if (payload == 0) {
      payload = 1;
    }
    return static_cast<std::uint16_t>(sign_and_exp | payload);
  }
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  const std::uint32_t rounded = bits + 0x7FFFU + lsb;
  return static_cast<std::uint16_t>(rounded >> 16U);
}

std::vector<std::uint8_t>
convert_f32_payload(std::span<const std::uint8_t> payload,
                    CudaWeightType target_type,
                    const std::string &tensor_name) {
  if (payload.size() % sizeof(float) != 0) {
    throw CudaWeightError("F32 auxiliary tensor byte size is invalid: " +
                          tensor_name);
  }
  std::vector<std::uint8_t> converted(payload.size() / sizeof(float) *
                                      sizeof(std::uint16_t));
  for (std::size_t element = 0; element < payload.size() / sizeof(float);
       ++element) {
    float value = 0.0F;
    std::memcpy(&value, payload.data() + element * sizeof(float),
                sizeof(value));
    const std::uint16_t bits = target_type == CudaWeightType::f16
                                   ? float_to_f16_bits(value)
                                   : float_to_bf16_bits(value);
    converted[element * sizeof(std::uint16_t)] =
        static_cast<std::uint8_t>(bits & 0xffU);
    converted[element * sizeof(std::uint16_t) + 1] =
        static_cast<std::uint8_t>((bits >> 8U) & 0xffU);
  }
  return converted;
}

bool descriptors_equal(const gguf::TensorInfo &left,
                       const gguf::TensorInfo &right) {
  return left.name == right.name && left.dimensions == right.dimensions &&
         left.type == right.type && left.offset == right.offset &&
         left.byte_size == right.byte_size;
}

class DeviceAllocation {
public:
  DeviceAllocation(ExecutionContext &context, std::size_t bytes)
      : resource_(context.memory_resource()), stream_ref_(context.stream()),
        stream_(context.stream()), device_id_(context.device_id()),
        bytes_(bytes == 0 ? 1 : bytes),
        data_(resource_.allocate(stream_ref_, bytes_,
                                 rmm::CUDA_ALLOCATION_ALIGNMENT)) {}

  ~DeviceAllocation() noexcept { reset(); }

  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  DeviceAllocation(DeviceAllocation &&other) noexcept
      : resource_(other.resource_), stream_ref_(other.stream_ref_),
        stream_(other.stream_), device_id_(other.device_id_),
        bytes_(std::exchange(other.bytes_, 0)),
        data_(std::exchange(other.data_, nullptr)) {}

  DeviceAllocation &operator=(DeviceAllocation &&other) noexcept = delete;

  void *data() const noexcept { return data_; }

private:
  void reset() noexcept {
    if (data_ == nullptr)
      return;
    BestEffortDeviceGuard device_guard{device_id_};
    if (!device_guard.selected_target())
      return;
    (void)cudaStreamSynchronize(stream_);
    try {
      resource_.deallocate(stream_ref_, data_, bytes_,
                           rmm::CUDA_ALLOCATION_ALIGNMENT);
    } catch (...) {
    }
    data_ = nullptr;
    bytes_ = 0;
  }

  rmm::device_async_resource_ref resource_;
  cuda::stream_ref stream_ref_;
  cudaStream_t stream_{};
  int device_id_{};
  std::size_t bytes_{};
  void *data_{};
};

} // namespace

class CudaWeightPlan::Impl {
public:
  std::shared_ptr<void> lifetime_anchor;
  std::vector<DeviceAllocation> allocations;
  std::vector<CudaTensorView> views;
  std::vector<gguf::TensorInfo> descriptors;
  std::unordered_map<std::string, std::size_t> indices_by_name;
  std::vector<Qwen35CudaLayerWeights> layers;
};

CudaWeightPlan::CudaWeightPlan(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

CudaWeightPlan::~CudaWeightPlan() noexcept = default;
CudaWeightPlan::CudaWeightPlan(CudaWeightPlan &&) noexcept = default;
CudaWeightPlan &CudaWeightPlan::operator=(CudaWeightPlan &&) noexcept = default;

std::size_t CudaWeightPlan::tensor_count() const noexcept {
  return impl_->views.size();
}

const CudaTensorView &CudaWeightPlan::token_embedding() const noexcept {
  return impl_->views[0];
}

const CudaTensorView &CudaWeightPlan::output_norm() const noexcept {
  return impl_->views[1];
}

const CudaTensorView &CudaWeightPlan::output() const noexcept {
  return impl_->views[2];
}

const CudaTensorView &CudaWeightPlan::tensor(std::size_t index) const {
  return impl_->views.at(index);
}

const CudaTensorView &CudaWeightPlan::tensor(std::string_view name) const {
  const auto found = impl_->indices_by_name.find(std::string{name});
  if (found == impl_->indices_by_name.end()) {
    throw CudaWeightError("unknown Qwen3.5 CUDA weight: " + std::string{name});
  }
  return impl_->views[found->second];
}

const CudaTensorView &
CudaWeightPlan::tensor(const gguf::TensorInfo &descriptor) const {
  const auto found = impl_->indices_by_name.find(descriptor.name);
  if (found == impl_->indices_by_name.end() ||
      !descriptors_equal(impl_->descriptors[found->second], descriptor)) {
    throw CudaWeightError(
        "GGUF tensor descriptor does not belong to CUDA weight plan: " +
        descriptor.name);
  }
  return impl_->views[found->second];
}

std::size_t CudaWeightPlan::layer_count() const noexcept {
  return impl_->layers.size();
}

const Qwen35CudaLayerWeights &CudaWeightPlan::layer(std::size_t index) const {
  return impl_->layers.at(index);
}

std::unique_ptr<CudaWeightPlan>
CudaWeightPlan::upload(ExecutionContext &context, const Model &model,
                       const Qwen35Manifest &manifest,
                       std::shared_ptr<void> lifetime_anchor) {
  if (!lifetime_anchor) {
    throw CudaWeightError(
        "CUDA weight upload requires a device resource lifetime anchor");
  }
  const auto tensors = manifest_tensors(manifest);
  struct ValidatedTensor {
    const gguf::TensorInfo *descriptor;
    CudaWeightType type;
    std::span<const std::uint8_t> payload;
    std::vector<std::uint8_t> converted_payload;
  };
  std::vector<ValidatedTensor> validated;
  validated.reserve(tensors.size());
  std::unordered_map<std::string, std::size_t> indices_by_name;
  indices_by_name.reserve(tensors.size());
  try {
    const CudaWeightType primary_type = checked_type(*manifest.token_embedding);
    for (std::size_t index = 0; index < tensors.size(); ++index) {
      const auto *tensor = tensors[index].descriptor;
      const bool inserted = indices_by_name.emplace(tensor->name, index).second;
      if (!inserted) {
        throw CudaWeightError("duplicate Qwen3.5 CUDA weight name: " +
                              tensor->name);
      }
      const auto payload = model.tensor_payload(*tensor);
      if (payload.size() != tensor->byte_size) {
        throw CudaWeightError("truncated Qwen3.5 tensor payload: " +
                              tensor->name);
      }
      validated.push_back(ValidatedTensor{
          .descriptor = tensor,
          .type = primary_type,
          .payload = {},
          .converted_payload = {},
      });
      auto &validated_tensor = validated.back();
      if (tensors[index].role == TensorUploadRole::primary) {
        const CudaWeightType type = checked_type(*tensor);
        if (type != primary_type) {
          throw CudaWeightError("Qwen3.5 primary CUDA weights must share the "
                                "selected dtype: " +
                                tensor->name);
        }
        validated_tensor.payload = payload;
      } else if (tensor->type == static_cast<std::uint32_t>(primary_type)) {
        validated_tensor.payload = payload;
      } else if (tensor->type == 0) {
        validated_tensor.converted_payload =
            convert_f32_payload(payload, primary_type, tensor->name);
        validated_tensor.payload = validated_tensor.converted_payload;
      } else {
        throw CudaWeightError(
            "Qwen3.5 auxiliary CUDA weight must be F32 or match the selected "
            "dtype: " +
            tensor->name);
      }
    }
  } catch (const CudaWeightError &) {
    throw;
  } catch (const std::exception &error) {
    throw CudaWeightError(error.what());
  }

  auto impl = std::unique_ptr<CudaWeightPlan::Impl>(new CudaWeightPlan::Impl);
  impl->lifetime_anchor = std::move(lifetime_anchor);
  impl->allocations.reserve(tensors.size());
  impl->views.reserve(tensors.size());
  impl->descriptors.reserve(tensors.size());
  impl->indices_by_name = std::move(indices_by_name);

  try {
    for (const auto &tensor : validated) {
      const auto payload = tensor.payload;
      impl->allocations.emplace_back(context, payload.size());
      void *device = impl->allocations.back().data();
      const auto copy_error =
          cudaMemcpyAsync(device, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice, context.stream());
      if (copy_error != cudaSuccess) {
        throw_cuda_weight_error(copy_error, tensor.descriptor->name);
      }
      impl->views.push_back(CudaTensorView{
          .device_data = device, .bytes = payload.size(), .type = tensor.type});
      impl->descriptors.push_back(*tensor.descriptor);
    }
    const auto sync_error = cudaStreamSynchronize(context.stream());
    if (sync_error != cudaSuccess) {
      throw CudaWeightError("CUDA weight upload synchronization failed: " +
                            std::string(cudaGetErrorString(sync_error)));
    }
  } catch (const CudaWeightError &) {
    throw;
  } catch (const std::exception &error) {
    throw CudaWeightError(error.what());
  }

  if (impl->views.size() != impl->indices_by_name.size() ||
      impl->descriptors.size() != impl->views.size()) {
    throw CudaWeightError("incomplete Qwen3.5 CUDA weight view map");
  }

  const auto view_for =
      [&impl](const gguf::TensorInfo &descriptor) -> const CudaTensorView & {
    const auto found = impl->indices_by_name.find(descriptor.name);
    if (found == impl->indices_by_name.end() ||
        !descriptors_equal(impl->descriptors[found->second], descriptor)) {
      throw CudaWeightError(
          "Qwen3.5 manifest descriptor is absent from CUDA weight plan: " +
          descriptor.name);
    }
    return impl->views[found->second];
  };
  impl->layers.reserve(manifest.layers.size());
  for (const auto &layer : manifest.layers) {
    Qwen35CudaLayerWeights cuda_layer{
        .index = layer.index,
        .common =
            Qwen35CudaCommonLayerWeights{
                .input_norm = view_for(*layer.common.input_norm),
                .post_attention_norm =
                    view_for(*layer.common.post_attention_norm),
                .ffn_gate = view_for(*layer.common.ffn_gate),
                .ffn_down = view_for(*layer.common.ffn_down),
                .ffn_up = view_for(*layer.common.ffn_up),
            },
        .full_attention = std::nullopt,
        .linear_attention = std::nullopt,
    };
    if (layer.full_attention.has_value()) {
      const auto &full = *layer.full_attention;
      cuda_layer.full_attention.emplace(Qwen35CudaFullAttentionWeights{
          .query = view_for(*full.query),
          .key = view_for(*full.key),
          .value = view_for(*full.value),
          .output = view_for(*full.output),
          .query_norm = view_for(*full.query_norm),
          .key_norm = view_for(*full.key_norm),
      });
    } else {
      const auto &linear = *layer.linear_attention;
      cuda_layer.linear_attention.emplace(Qwen35CudaLinearAttentionWeights{
          .qkv = view_for(*linear.qkv),
          .gate = view_for(*linear.gate),
          .convolution = view_for(*linear.convolution),
          .time_step_bias = view_for(*linear.time_step_bias),
          .recurrent_a = view_for(*linear.recurrent_a),
          .beta = view_for(*linear.beta),
          .alpha = view_for(*linear.alpha),
          .output_norm = view_for(*linear.output_norm),
          .output = view_for(*linear.output),
      });
    }
    impl->layers.push_back(std::move(cuda_layer));
  }

  return std::unique_ptr<CudaWeightPlan>(new CudaWeightPlan(std::move(impl)));
}

} // namespace brt::model
