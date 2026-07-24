add_library(brt_cpp STATIC src/status.cpp)
target_sources(brt_cpp PRIVATE src/c_api.cpp src/engine.cpp operators/tensor_validation.cpp)
target_compile_features(brt_cpp PUBLIC cxx_std_20)
target_include_directories(brt_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(brt_cpp PRIVATE BRT_ENABLE_CUDA=$<BOOL:${BRT_ENABLE_CUDA}>)

if(BRT_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(raft 26.06 CONFIG REQUIRED)
  find_package(rmm 26.06 CONFIG REQUIRED)
  target_sources(brt_cpp PRIVATE foundation/device_context.cu kernels/smoke.cu)
  target_link_libraries(brt_cpp PUBLIC raft::raft rmm::rmm CUDA::cudart)
  target_compile_options(brt_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
endif()

include(GNUInstallDirs)
install(TARGETS brt_cpp ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_executable(brt-smoke tools/brt_smoke.cpp)
target_link_libraries(brt-smoke PRIVATE brt_cpp)

if(BRT_BUILD_TESTS)
  enable_testing()
  add_executable(brt_c_api_test tests/c_api_test.cpp)
  target_link_libraries(brt_c_api_test PRIVATE brt_cpp)
  add_test(NAME brt_c_api_test COMMAND brt_c_api_test)
  add_executable(brt_tensor_validation_test tests/tensor_validation_test.cpp)
  target_link_libraries(brt_tensor_validation_test PRIVATE brt_cpp)
  add_test(NAME brt_tensor_validation_test COMMAND brt_tensor_validation_test)
  add_test(
    NAME brt_c_api_source_test
    COMMAND ${CMAKE_COMMAND}
      -DC_API_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/src/c_api.cpp
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/c_api_source_test.cmake)
  add_test(
    NAME brt_device_context_source_test
    COMMAND ${CMAKE_COMMAND}
      -DDEVICE_CONTEXT_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/foundation/device_context.hpp
      -DDEVICE_CONTEXT_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/foundation/device_context.cu
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/device_context_source_test.cmake)
endif()
