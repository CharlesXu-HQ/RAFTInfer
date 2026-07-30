#pragma once

#include "workspace_arena.hpp"

#include <raft/core/device_resources.hpp>
#include <rmm/resource_ref.hpp>
#include <cuda_runtime_api.h>

namespace raftinfer {

class ExecutionContext {
 public:
  ExecutionContext(
      raft::device_resources& resources,
      rmm::device_async_resource_ref memory_resource,
      cudaStream_t stream,
      WorkspaceArena& workspace,
      int device_id,
      int compute_capability_major,
      int compute_capability_minor,
      int max_shared_memory_per_block) noexcept
      : resources_(resources),
        memory_resource_(memory_resource),
        stream_(stream),
        workspace_(workspace),
        device_id_(device_id),
        compute_capability_major_(compute_capability_major),
        compute_capability_minor_(compute_capability_minor),
        max_shared_memory_per_block_(max_shared_memory_per_block) {}

  raft::device_resources& resources() const noexcept { return resources_; }
  rmm::device_async_resource_ref memory_resource() const noexcept { return memory_resource_; }
  cudaStream_t stream() const noexcept { return stream_; }
  WorkspaceArena& workspace() const noexcept { return workspace_; }
  int device_id() const noexcept { return device_id_; }
  int compute_capability_major() const noexcept { return compute_capability_major_; }
  int compute_capability_minor() const noexcept { return compute_capability_minor_; }
  int max_shared_memory_per_block() const noexcept { return max_shared_memory_per_block_; }

 private:
  raft::device_resources& resources_;
  rmm::device_async_resource_ref memory_resource_;
  cudaStream_t stream_;
  WorkspaceArena& workspace_;
  int device_id_;
  int compute_capability_major_;
  int compute_capability_minor_;
  int max_shared_memory_per_block_;
};

}  // namespace raftinfer
