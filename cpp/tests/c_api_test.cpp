#include <brt/status.h>

#include <cassert>
#include <type_traits>

int main() {
  static_assert(std::is_standard_layout_v<BrtStatus>);
  static_assert(std::is_trivially_copyable_v<BrtStatus>);
  BrtStatus ok{BRT_STATUS_OK, nullptr};
  assert(ok.code == BRT_STATUS_OK);
  assert(ok.message == nullptr);
  return 0;
}
