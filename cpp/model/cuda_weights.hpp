#pragma once

#include "qwen35_manifest.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace brt {
class DeviceContext;
class ExecutionContext;
}

namespace brt::model {

class Model;

class CudaWeightError : public std::runtime_error {
 public:
  explicit CudaWeightError(const std::string& message)
      : std::runtime_error(message) {}
};

enum class CudaWeightType : std::uint32_t {
  f16 = 1,
  bf16 = 30,
};

struct CudaTensorView {
  const void* device_data{};
  std::size_t bytes{};
  CudaWeightType type{};
};

class CudaWeightPlan {
 public:
  ~CudaWeightPlan() noexcept;

  CudaWeightPlan(const CudaWeightPlan&) = delete;
  CudaWeightPlan& operator=(const CudaWeightPlan&) = delete;
  CudaWeightPlan(CudaWeightPlan&&) noexcept;
  CudaWeightPlan& operator=(CudaWeightPlan&&) noexcept;

  std::size_t tensor_count() const noexcept;
  const CudaTensorView& token_embedding() const noexcept;
  const CudaTensorView& output_norm() const noexcept;
  const CudaTensorView& output() const noexcept;
  const CudaTensorView& tensor(std::size_t index) const;

 private:
  class Impl;
  explicit CudaWeightPlan(std::unique_ptr<Impl> impl) noexcept;
  static std::unique_ptr<CudaWeightPlan>
  upload(ExecutionContext& context, const Model& model,
         const Qwen35Manifest& manifest,
         std::shared_ptr<void> lifetime_anchor);

  std::unique_ptr<Impl> impl_;

  friend class brt::DeviceContext;
};

}  // namespace brt::model
