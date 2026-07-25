#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "gguf_types.hpp"
#include "qwen35_config.hpp"
#include "qwen35_manifest.hpp"

namespace brt {
class DeviceContext;
class Engine;
} // namespace brt

namespace brt::model {

class CudaWeightPlan;

class ModelIoError : public std::runtime_error {
public:
  explicit ModelIoError(const std::string &message)
      : std::runtime_error(message) {}
};

class Model {
public:
  explicit Model(const std::string &gguf_path);
  ~Model();

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(Model &&) = delete;
  Model &operator=(Model &&) = delete;

  std::span<const std::uint8_t> tokenizer_spec() const noexcept;
  const Qwen35Config &qwen35_config() const noexcept;
  const Qwen35Manifest &qwen35_manifest() const noexcept;
  std::span<const std::uint8_t>
  tensor_payload(const gguf::TensorInfo &tensor) const;
  bool cuda_ready() const noexcept;
  const CudaWeightPlan *cuda_weights() const noexcept;
  std::shared_ptr<const DeviceContext> device_context() const noexcept;

private:
  void attach_cuda(std::shared_ptr<DeviceContext> device,
                   std::shared_ptr<CudaWeightPlan> weights);

  class Impl;
  std::unique_ptr<Impl> impl_;

  friend class brt::Engine;
};

} // namespace brt::model
