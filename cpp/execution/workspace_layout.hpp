#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace brt {

class WorkspaceLayout {
 public:
  explicit WorkspaceLayout(std::size_t capacity) noexcept : capacity_(capacity) {}

  std::size_t allocate(std::size_t bytes, std::size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
      throw std::invalid_argument("workspace alignment must be a non-zero power of two");
    }

    const std::size_t mask = alignment - 1;
    if (used_ > std::numeric_limits<std::size_t>::max() - mask) {
      throw std::length_error("workspace offset overflow");
    }
    const std::size_t aligned = (used_ + mask) & ~mask;
    if (aligned > std::numeric_limits<std::size_t>::max() - bytes) {
      throw std::length_error("workspace allocation overflow");
    }
    const std::size_t next = aligned + bytes;
    if (next > capacity_) {
      throw std::length_error("workspace capacity exceeded");
    }
    if (bytes != 0) {
      used_ = next;
    }
    return aligned;
  }

  void reset() noexcept { used_ = 0; }
  std::size_t used() const noexcept { return used_; }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  std::size_t capacity_;
  std::size_t used_{};
};

}  // namespace brt
