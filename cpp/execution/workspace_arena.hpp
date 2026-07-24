#pragma once

#include "workspace_layout.hpp"

#include <rmm/resource_ref.hpp>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace brt {

class WorkspaceArena {
 public:
  WorkspaceArena(
      rmm::device_async_resource_ref resource,
      cuda::stream_ref stream_ref,
      cudaStream_t stream,
      std::size_t bytes);
  ~WorkspaceArena() noexcept;

  WorkspaceArena(const WorkspaceArena&) = delete;
  WorkspaceArena& operator=(const WorkspaceArena&) = delete;

  void* allocate(std::size_t bytes, std::size_t alignment);
  void reset() noexcept;
  std::size_t used() const noexcept;
  std::size_t capacity() const noexcept;

 private:
  rmm::device_async_resource_ref resource_;
  cuda::stream_ref stream_ref_;
  cudaStream_t stream_;
  std::size_t bytes_;
  std::byte* base_{};
  WorkspaceLayout layout_;
};

}  // namespace brt
