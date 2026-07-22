#include <brt/status.h>

#include <type_traits>

static_assert(std::is_standard_layout_v<BrtStatus>);
static_assert(std::is_trivially_copyable_v<BrtStatus>);
