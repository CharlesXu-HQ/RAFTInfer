file(READ "${C_API_SOURCE}" c_api_source)

# Production change caught: reintroducing a legacy public C++ namespace,
# include root, C ABI type, function, or macro after the RAFTInfer rename.
string(CONCAT legacy_lower "br" "t")
string(CONCAT legacy_title "Br" "t")
string(CONCAT legacy_upper "BR" "T")
string(CONCAT legacy_include "#include[ \\t]*<" "${legacy_lower}" "/")
string(CONCAT legacy_namespace "${legacy_lower}" "::")
string(CONCAT legacy_type "${legacy_title}" "[A-Za-z0-9_]*")
string(CONCAT legacy_function "${legacy_lower}" "_[A-Za-z0-9_]*")
string(CONCAT legacy_macro "${legacy_upper}" "_[A-Za-z0-9_]*")
string(CONCAT legacy_diagnostic
  "C API source retains a legacy " "${legacy_upper}" " public spelling: ")
foreach(legacy_expression
    "${legacy_include}"
    "${legacy_namespace}"
    "${legacy_type}"
    "${legacy_function}"
    "${legacy_macro}")
  if(c_api_source MATCHES "${legacy_expression}")
    message(FATAL_ERROR "${legacy_diagnostic}${legacy_expression}")
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
