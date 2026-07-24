#include "workspace_arena.hpp"

#include <rmm/aligned.hpp>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace brt {

WorkspaceArena::WorkspaceArena(
    rmm::device_async_resource_ref resource,
    cuda::stream_ref stream_ref,
    cudaStream_t stream,
    std::size_t bytes)
    : resource_(resource),
      stream_ref_(stream_ref),
      stream_(stream),
      bytes_(std::max<std::size_t>(bytes, 1)),
      base_(static_cast<std::byte*>(
          resource_.allocate(stream_ref_, bytes_, rmm::CUDA_ALLOCATION_ALIGNMENT))),
      layout_(bytes_) {}

WorkspaceArena::~WorkspaceArena() noexcept {
  if (base_ != nullptr) {
    (void)cudaStreamSynchronize(stream_);
    try {
      std::byte* base = std::exchange(base_, nullptr);
      resource_.deallocate(stream_ref_, base, bytes_, rmm::CUDA_ALLOCATION_ALIGNMENT);
    } catch (...) {
    }
  }
}

void* WorkspaceArena::allocate(std::size_t bytes, std::size_t alignment) {
  const std::size_t offset = layout_.allocate(bytes, alignment);
  return base_ + offset;
}

void WorkspaceArena::reset() noexcept { layout_.reset(); }

std::size_t WorkspaceArena::used() const noexcept { return layout_.used(); }

std::size_t WorkspaceArena::capacity() const noexcept { return layout_.capacity(); }

}  // namespace brt
