#include <raftinfer/status.h>

#include <type_traits>

static_assert(std::is_standard_layout_v<RaftInferStatus>);
static_assert(std::is_trivially_copyable_v<RaftInferStatus>);
