cmake_minimum_required(VERSION 3.26.4)
project(raftinfer VERSION 0.1.0 LANGUAGES CXX)

include(cmake/RAFTInferOptions.cmake)
enable_testing()
add_subdirectory(cpp)
