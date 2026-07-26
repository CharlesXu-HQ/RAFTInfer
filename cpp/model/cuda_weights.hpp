#pragma once

#include "qwen35_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace brt {
class DeviceContext;
class ExecutionContext;
} // namespace brt

namespace brt::model {

class Model;

class CudaWeightError : public std::runtime_error {
public:
  explicit CudaWeightError(const std::string &message)
      : std::runtime_error(message) {}
};

enum class CudaWeightType : std::uint32_t {
  f32 = 0,
  f16 = 1,
  bf16 = 30,
};

struct CudaTensorView {
  const void *device_data{};
  std::size_t bytes{};
  CudaWeightType type{};
};

struct Qwen35CudaCommonLayerWeights {
  const CudaTensorView &input_norm;
  const CudaTensorView &post_attention_norm;
  const CudaTensorView &ffn_gate;
  const CudaTensorView &ffn_down;
  const CudaTensorView &ffn_up;
};

struct Qwen35CudaFullAttentionWeights {
  const CudaTensorView &query;
  const CudaTensorView &key;
  const CudaTensorView &value;
  const CudaTensorView &output;
  const CudaTensorView &query_norm;
  const CudaTensorView &key_norm;
};

struct Qwen35CudaLinearAttentionWeights {
  const CudaTensorView &qkv;
  const CudaTensorView &gate;
  const CudaTensorView &convolution;
  const CudaTensorView &time_step_bias;
  const CudaTensorView &recurrent_a;
  const CudaTensorView &beta;
  const CudaTensorView &alpha;
  const CudaTensorView &output_norm;
  const CudaTensorView &output;
};

struct Qwen35CudaLayerWeights {
  std::uint32_t index{};
  Qwen35CudaCommonLayerWeights common;
  std::optional<Qwen35CudaFullAttentionWeights> full_attention;
  std::optional<Qwen35CudaLinearAttentionWeights> linear_attention;
};

class CudaWeightPlan {
public:
  ~CudaWeightPlan() noexcept;

  CudaWeightPlan(const CudaWeightPlan &) = delete;
  CudaWeightPlan &operator=(const CudaWeightPlan &) = delete;
  CudaWeightPlan(CudaWeightPlan &&) noexcept;
  CudaWeightPlan &operator=(CudaWeightPlan &&) noexcept;

  std::size_t tensor_count() const noexcept;
  const CudaTensorView &token_embedding() const noexcept;
  const CudaTensorView &output_norm() const noexcept;
  const CudaTensorView &output() const noexcept;
  const CudaTensorView &tensor(std::size_t index) const;
  const CudaTensorView &tensor(std::string_view name) const;
  const CudaTensorView &tensor(const gguf::TensorInfo &descriptor) const;
  std::size_t layer_count() const noexcept;
  const Qwen35CudaLayerWeights &layer(std::size_t index) const;

private:
  class Impl;
  explicit CudaWeightPlan(std::unique_ptr<Impl> impl) noexcept;
  static std::unique_ptr<CudaWeightPlan>
  upload(ExecutionContext &context, const Model &model,
         const Qwen35Manifest &manifest, std::shared_ptr<void> lifetime_anchor);

  std::unique_ptr<Impl> impl_;

  friend class brt::DeviceContext;
};

} // namespace brt::model
