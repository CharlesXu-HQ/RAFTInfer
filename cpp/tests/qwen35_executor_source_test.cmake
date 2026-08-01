function(require_match content expression description)
  string(REGEX MATCH "${expression}" match "${content}")
  if(NOT match)
    message(FATAL_ERROR "Missing ${description}")
  endif()
endfunction()

file(READ "${QWEN35_EXECUTOR_SOURCE}" source)

# Production change caught: reintroducing the legacy namespace in executor
# implementation after the RAFTInfer rename.
set(legacy_namespace "b" "rt")
string(JOIN "" legacy_namespace ${legacy_namespace})
if(source MATCHES "namespace[ \\t\\r\\n]+${legacy_namespace}" OR source MATCHES "${legacy_namespace}::")
  message(FATAL_ERROR "Qwen3.5 executor source retains the legacy ${legacy_namespace} namespace")
endif()

require_match("${source}" "cudaFreeHost\\(ptr\\)"
              "CUDA pinned-host deleter")
require_match("${source}" "CudaHostInt32 host_decode_token_;[ \n]+CudaHostInt32 host_decode_result_;"
              "owned pinned decode scalar members")
require_match("${source}" "require\\(synchronize \\|\\| fixed_host_result != nullptr,"
              "fixed host storage precondition for unsynchronized result downloads")
require_match("${source}" "\\.token = synchronize \\? \\*result : 0,"
              "unsynchronized result return avoids dereferencing pending D2H storage")
