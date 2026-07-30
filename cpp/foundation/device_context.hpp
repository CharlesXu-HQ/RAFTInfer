#pragma once

#include <raftinfer/c_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace raftinfer::model {
class CudaWeightPlan;
class Model;
struct Qwen35Manifest;
}  // namespace raftinfer::model

namespace raftinfer {

class ExecutionContext;
class DeviceContext;

class DeviceExecutionOwner {
 public:
  ~DeviceExecutionOwner() noexcept;

  DeviceExecutionOwner(const DeviceExecutionOwner&) = delete;
  DeviceExecutionOwner& operator=(const DeviceExecutionOwner&) = delete;
  DeviceExecutionOwner(DeviceExecutionOwner&&) = delete;
  DeviceExecutionOwner& operator=(DeviceExecutionOwner&&) = delete;

  ExecutionContext execution_context();
  std::size_t workspace_bytes() const noexcept;
  int device_id() const noexcept;

 private:
  class Impl;

  explicit DeviceExecutionOwner(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;

  friend class DeviceContext;
};

class DeviceContext {
 public:
  DeviceContext(int device_id, uint64_t initial_pool_bytes);
  ~DeviceContext() noexcept;
  RaftInferSmokeResult run_smoke();
  std::uint64_t peak_allocated_bytes();
  std::unique_ptr<DeviceExecutionOwner>
  create_execution_owner(std::size_t workspace_bytes) const;
  std::unique_ptr<model::CudaWeightPlan>
  upload_qwen35_weights(const model::Model& model);
  std::unique_ptr<model::CudaWeightPlan>
  upload_qwen35_weights_for_tests(const model::Model& model,
                                  const model::Qwen35Manifest& manifest);

 private:
  class Resources;

  int device_id_;
  std::shared_ptr<Resources> resources_;
};

}  // namespace raftinfer
