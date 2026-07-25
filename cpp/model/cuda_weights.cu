#include "cuda_weights.hpp"

#include "../execution/execution_context.hpp"
#include "model.hpp"

#include <rmm/aligned.hpp>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace brt::model {
namespace {

class BestEffortDeviceGuard {
 public:
  explicit BestEffortDeviceGuard(int device_id) noexcept {
    if (cudaGetDevice(&previous_device_) != cudaSuccess) return;
    captured_previous_ = true;
    if (cudaSetDevice(device_id) != cudaSuccess) return;
    selected_target_ = true;
  }

  ~BestEffortDeviceGuard() noexcept { restore(); }

  BestEffortDeviceGuard(const BestEffortDeviceGuard&) = delete;
  BestEffortDeviceGuard& operator=(const BestEffortDeviceGuard&) = delete;

  bool selected_target() const noexcept { return selected_target_; }

  void restore() noexcept {
    if (!captured_previous_) return;
    (void)cudaSetDevice(previous_device_);
    captured_previous_ = false;
  }

 private:
  int previous_device_{};
  bool captured_previous_{};
  bool selected_target_{};
};

[[noreturn]] void throw_cuda_weight_error(cudaError_t error,
                                          const std::string& tensor_name) {
  throw CudaWeightError("CUDA weight upload failed for " + tensor_name + ": " +
                        cudaGetErrorString(error));
}

CudaWeightType checked_type(const gguf::TensorInfo& tensor) {
  if (tensor.type == static_cast<std::uint32_t>(CudaWeightType::f16)) {
    return CudaWeightType::f16;
  }
  if (tensor.type == static_cast<std::uint32_t>(CudaWeightType::bf16)) {
    return CudaWeightType::bf16;
  }
  throw CudaWeightError("unsupported Qwen3.5 M2 primary weight type for " +
                        tensor.name);
}

std::vector<const gguf::TensorInfo*> primary_tensors(
    const Qwen35Manifest& manifest) {
  std::vector<const gguf::TensorInfo*> tensors;
  tensors.reserve(3 + manifest.layers.size() * 14);
  tensors.push_back(manifest.token_embedding);
  tensors.push_back(manifest.output_norm);
  tensors.push_back(manifest.output);
  for (const auto& layer : manifest.layers) {
    tensors.push_back(layer.common.input_norm);
    tensors.push_back(layer.common.post_attention_norm);
    tensors.push_back(layer.common.ffn_gate);
    tensors.push_back(layer.common.ffn_down);
    tensors.push_back(layer.common.ffn_up);
    if (layer.full_attention.has_value()) {
      const auto& full = *layer.full_attention;
      tensors.push_back(full.query);
      tensors.push_back(full.key);
      tensors.push_back(full.value);
      tensors.push_back(full.output);
      tensors.push_back(full.query_norm);
      tensors.push_back(full.key_norm);
    }
    if (layer.linear_attention.has_value()) {
      const auto& linear = *layer.linear_attention;
      tensors.push_back(linear.qkv);
      tensors.push_back(linear.gate);
      tensors.push_back(linear.convolution);
      tensors.push_back(linear.time_step_bias);
      tensors.push_back(linear.recurrent_a);
      tensors.push_back(linear.beta);
      tensors.push_back(linear.alpha);
      tensors.push_back(linear.output_norm);
      tensors.push_back(linear.output);
    }
  }
  if (std::any_of(tensors.begin(), tensors.end(),
                  [](const auto* tensor) { return tensor == nullptr; })) {
    throw CudaWeightError("Qwen3.5 manifest contains a null primary tensor");
  }
  return tensors;
}

class DeviceAllocation {
 public:
  DeviceAllocation(ExecutionContext& context, std::size_t bytes)
      : resource_(context.memory_resource()),
        stream_ref_(context.stream()),
        stream_(context.stream()),
        device_id_(context.device_id()),
        bytes_(bytes == 0 ? 1 : bytes),
        data_(resource_.allocate(stream_ref_, bytes_,
                                 rmm::CUDA_ALLOCATION_ALIGNMENT)) {}

  ~DeviceAllocation() noexcept { reset(); }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  DeviceAllocation(DeviceAllocation&& other) noexcept
      : resource_(other.resource_),
        stream_ref_(other.stream_ref_),
        stream_(other.stream_),
        device_id_(other.device_id_),
        bytes_(std::exchange(other.bytes_, 0)),
        data_(std::exchange(other.data_, nullptr)) {}

  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept = delete;

  void* data() const noexcept { return data_; }

 private:
  void reset() noexcept {
    if (data_ == nullptr) return;
    BestEffortDeviceGuard device_guard{device_id_};
    if (!device_guard.selected_target()) return;
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
  void* data_{};
};

}  // namespace

class CudaWeightPlan::Impl {
 public:
  std::shared_ptr<void> lifetime_anchor;
  std::vector<DeviceAllocation> allocations;
  std::vector<CudaTensorView> views;
};

CudaWeightPlan::CudaWeightPlan(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

CudaWeightPlan::~CudaWeightPlan() noexcept = default;
CudaWeightPlan::CudaWeightPlan(CudaWeightPlan&&) noexcept = default;
CudaWeightPlan& CudaWeightPlan::operator=(CudaWeightPlan&&) noexcept = default;

std::size_t CudaWeightPlan::tensor_count() const noexcept {
  return impl_->views.size();
}

const CudaTensorView& CudaWeightPlan::token_embedding() const noexcept {
  return impl_->views[0];
}

const CudaTensorView& CudaWeightPlan::output_norm() const noexcept {
  return impl_->views[1];
}

const CudaTensorView& CudaWeightPlan::output() const noexcept {
  return impl_->views[2];
}

const CudaTensorView& CudaWeightPlan::tensor(std::size_t index) const {
  return impl_->views.at(index);
}

std::unique_ptr<CudaWeightPlan>
CudaWeightPlan::upload(ExecutionContext& context, const Model& model,
                       const Qwen35Manifest& manifest,
                       std::shared_ptr<void> lifetime_anchor) {
  if (!lifetime_anchor) {
    throw CudaWeightError("CUDA weight upload requires a device resource lifetime anchor");
  }
  const auto tensors = primary_tensors(manifest);
  auto impl = std::unique_ptr<CudaWeightPlan::Impl>(
      new CudaWeightPlan::Impl);
  impl->lifetime_anchor = std::move(lifetime_anchor);
  impl->allocations.reserve(tensors.size());
  impl->views.reserve(tensors.size());

  try {
    for (const auto* tensor : tensors) {
      const CudaWeightType type = checked_type(*tensor);
      const auto payload = model.tensor_payload(*tensor);
      if (payload.size() != tensor->byte_size) {
        throw CudaWeightError("truncated Qwen3.5 tensor payload: " +
                              tensor->name);
      }
      impl->allocations.emplace_back(context, payload.size());
      void* device = impl->allocations.back().data();
      const auto copy_error =
          cudaMemcpyAsync(device, payload.data(), payload.size(),
                          cudaMemcpyHostToDevice, context.stream());
      if (copy_error != cudaSuccess) {
        throw_cuda_weight_error(copy_error, tensor->name);
      }
      impl->views.push_back(CudaTensorView{.device_data = device,
                                           .bytes = payload.size(),
                                           .type = type});
    }
    const auto sync_error = cudaStreamSynchronize(context.stream());
    if (sync_error != cudaSuccess) {
      throw CudaWeightError("CUDA weight upload synchronization failed: " +
                            std::string(cudaGetErrorString(sync_error)));
    }
  } catch (const CudaWeightError&) {
    throw;
  } catch (const std::exception& error) {
    throw CudaWeightError(error.what());
  }

  return std::unique_ptr<CudaWeightPlan>(
      new CudaWeightPlan(std::move(impl)));
}

}  // namespace brt::model
