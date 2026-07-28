file(READ "${C_API_SOURCE}" c_api_source)

if(c_api_source MATCHES "make_unique[ \t\r\n]*<[ \t\r\n]*BrtEngineHandle[ \t\r\n]*>[ \t\r\n]*\\([ \t\r\n]*BrtEngineHandle[ \t\r\n]*\\{")
  message(FATAL_ERROR "brt_engine_create must construct BrtEngineHandle directly; aggregate temporary construction requires moving the non-movable CUDA Engine")
endif()

foreach(message
    "unmapped Qwen3.5 attention implementation"
    "unmapped Qwen3.5 KV cache dtype"
    "unmapped Qwen3.5 KV cache layout")
  if(NOT c_api_source MATCHES "throw std::logic_error\\(\"${message}\"\\)")
    message(FATAL_ERROR "C API diagnostics converters must throw std::logic_error for: ${message}")
  endif()
endforeach()
