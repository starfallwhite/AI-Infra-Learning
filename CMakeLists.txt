cmake_minimum_required(VERSION 3.20)

project(AIInfraLearning
        VERSION 0.1
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS on)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)

#单元测试
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

#新能测试
FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest benchmark)

enable_testing()

add_subdirectory(cpp-basics)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda-kernels/CMakeLists.txt")
    add_subdirectory(cuda-kernels)
else()
    message(STATUS "Skipping cuda-kernels: no CMakeLists.txt found")
endif()