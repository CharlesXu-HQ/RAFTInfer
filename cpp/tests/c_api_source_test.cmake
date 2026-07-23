file(READ "${C_API_SOURCE}" c_api_source)

if(c_api_source MATCHES "make_unique[ \t\r\n]*<[ \t\r\n]*BrtEngineHandle[ \t\r\n]*>[ \t\r\n]*\\([ \t\r\n]*BrtEngineHandle[ \t\r\n]*\\{")
  message(FATAL_ERROR "brt_engine_create must construct BrtEngineHandle directly; aggregate temporary construction requires moving the non-movable CUDA Engine")
endif()
