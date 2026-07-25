add_library(brt_cpp STATIC src/status.cpp)
target_sources(brt_cpp PRIVATE
  src/c_api.cpp
  src/engine.cpp
  operators/tensor_validation.cpp
  model/gguf_reader.cpp
  model/model.cpp
  model/qwen35_config.cpp
  model/qwen35_manifest.cpp
  execution/qwen35_state.cpp)
target_sources(brt_cpp PRIVATE registry/operator_registry.cpp)
target_sources(brt_cpp PRIVATE benchmarks/benchmark_record.cpp)
target_compile_features(brt_cpp PUBLIC cxx_std_20)
target_include_directories(brt_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(brt_cpp PRIVATE BRT_ENABLE_CUDA=$<BOOL:${BRT_ENABLE_CUDA}>)

add_library(brt_reference STATIC reference/operators.cpp reference/correctness.cpp reference/qwen35.cpp)
target_compile_features(brt_reference PUBLIC cxx_std_20)
target_include_directories(brt_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/reference)

if(BRT_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(raft 26.06 CONFIG REQUIRED)
  find_package(rmm 26.06 CONFIG REQUIRED)
  target_sources(brt_cpp PRIVATE
    execution/workspace_arena.cu
    foundation/device_context.cu
    kernels/smoke.cu)
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

  function(brt_add_assert_test target)
    add_executable(${target} ${ARGN})
    target_compile_options(${target} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_test(NAME ${target} COMMAND ${target})
  endfunction()

  brt_add_assert_test(brt_c_api_test tests/c_api_test.cpp)
  target_link_libraries(brt_c_api_test PRIVATE brt_cpp)
  brt_add_assert_test(brt_tensor_validation_test tests/tensor_validation_test.cpp)
  target_link_libraries(brt_tensor_validation_test PRIVATE brt_cpp)
  brt_add_assert_test(brt_workspace_layout_test tests/workspace_layout_test.cpp)
  target_link_libraries(brt_workspace_layout_test PRIVATE brt_cpp)
  brt_add_assert_test(brt_operator_registry_test tests/operator_registry_test.cpp)
  target_link_libraries(brt_operator_registry_test PRIVATE brt_cpp)
  brt_add_assert_test(brt_reference_operators_test tests/reference_operators_test.cpp)
  target_link_libraries(brt_reference_operators_test PRIVATE brt_reference)
  brt_add_assert_test(brt_qwen35_reference_test tests/qwen35_reference_test.cpp)
  target_link_libraries(brt_qwen35_reference_test PRIVATE brt_reference brt_cpp)
  brt_add_assert_test(brt_qwen35_fixture_test tests/qwen35_fixture_test.cpp)
  target_link_libraries(brt_qwen35_fixture_test PRIVATE brt_reference brt_cpp)
  target_compile_definitions(brt_qwen35_fixture_test PRIVATE
    BRT_QWEN35_FIXTURE_PATH="${CMAKE_SOURCE_DIR}/tests/fixtures/qwen35/tiny-layer-v1.bin")
  brt_add_assert_test(brt_correctness_test tests/correctness_test.cpp)
  target_link_libraries(brt_correctness_test PRIVATE brt_reference)
  brt_add_assert_test(brt_benchmark_record_test
    tests/benchmark_record_test.cpp)
  target_link_libraries(brt_benchmark_record_test PRIVATE brt_cpp)
  brt_add_assert_test(brt_gguf_reader_test tests/gguf_reader_test.cpp)
  target_link_libraries(brt_gguf_reader_test PRIVATE brt_cpp)
  brt_add_assert_test(brt_qwen35_config_test tests/qwen35_config_test.cpp)
  target_link_libraries(brt_qwen35_config_test PRIVATE brt_cpp)
  brt_add_assert_test(brt_qwen35_manifest_test tests/qwen35_manifest_test.cpp)
  target_link_libraries(brt_qwen35_manifest_test PRIVATE brt_cpp)
  brt_add_assert_test(brt_qwen35_state_test tests/qwen35_state_test.cpp)
  target_link_libraries(brt_qwen35_state_test PRIVATE brt_cpp)
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
      -DWORKSPACE_ARENA_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/execution/workspace_arena.hpp
      -DWORKSPACE_ARENA_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/execution/workspace_arena.cu
      -DEXECUTION_CONTEXT_HEADER=${CMAKE_CURRENT_SOURCE_DIR}/execution/execution_context.hpp
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/device_context_source_test.cmake)
endif()
