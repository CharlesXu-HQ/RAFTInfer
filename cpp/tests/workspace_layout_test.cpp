#include "../execution/workspace_layout.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>

int main() {
  brt::WorkspaceLayout layout{256};
  assert(layout.allocate(3, 1) == 0);
  assert(layout.allocate(16, 16) == 16);
  assert(layout.used() == 32);
  try {
    (void)layout.allocate(1, 3);
    assert(false);
  } catch (const std::invalid_argument&) {
  }
  try {
    (void)layout.allocate(240, 16);
    assert(false);
  } catch (const std::length_error&) {
  }
  layout.reset();
  assert(layout.used() == 0);
  assert(layout.allocate(256, 256) == 0);

  constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
  brt::WorkspaceLayout alignment_overflow{max};
  assert(alignment_overflow.allocate(max - 7, 1) == 0);
  try {
    (void)alignment_overflow.allocate(1, 16);
    assert(false);
  } catch (const std::length_error&) {
  }

  brt::WorkspaceLayout allocation_overflow{max};
  assert(allocation_overflow.allocate(max - 15, 1) == 0);
  try {
    (void)allocation_overflow.allocate(16, 16);
    assert(false);
  } catch (const std::length_error&) {
  }
}
