file(READ "${C_API_SOURCE}" c_api_source)

# Production change caught: reintroducing a legacy public C++ namespace,
# include root, C ABI type, function, or macro after the RAFTInfer rename.
foreach(legacy_expression
    "#include[ \\t]*<brt/"
    "brt::"
    "Brt[A-Za-z0-9_]*"
    "brt_[A-Za-z0-9_]*"
    "BRT_[A-Za-z0-9_]*")
  if(c_api_source MATCHES "${legacy_expression}")
    message(FATAL_ERROR "C API source retains a legacy BRT public spelling: ${legacy_expression}")
  endif()
endforeach()

if(c_api_source MATCHES "make_unique[ \t\r\n]*<[ \t\r\n]*RaftInferEngineHandle[ \t\r\n]*>[ \t\r\n]*\\([ \t\r\n]*RaftInferEngineHandle[ \t\r\n]*\\{")
  message(FATAL_ERROR "raftinfer_engine_create must construct RaftInferEngineHandle directly; aggregate temporary construction requires moving the non-movable CUDA Engine")
endif()

foreach(message
    "unmapped Qwen3.5 attention implementation"
    "unmapped Qwen3.5 KV cache dtype"
    "unmapped Qwen3.5 KV cache layout")
  if(NOT c_api_source MATCHES "throw std::logic_error\\(\"${message}\"\\)")
    message(FATAL_ERROR "C API diagnostics converters must throw std::logic_error for: ${message}")
  endif()
endforeach()
