option(BRT_ENABLE_CUDA "Build the RAFT/RMM CUDA backend" OFF)
option(BRT_BUILD_TESTS "Build BRT tests" ON)

if(BRT_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures" FORCE)
endif()
