find_package(CUDAToolkit REQUIRED)

add_library(cuda_vector_add
    src/vector_add.cu
)

add_library(cuda_reductions
    src/reduction.cu
    src/softmax.cu
)

target_include_directories(cuda_vector_add
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(cuda_reductions
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(cuda_vector_add PROPERTIES
    CUDA_ARCHITECTURES 75
)

set_target_properties(cuda_reductions PROPERTIES
    CUDA_ARCHITECTURES 75
)

target_link_libraries(cuda_vector_add
    PUBLIC
        CUDA::cudart
)

target_link_libraries(cuda_reductions
    PUBLIC
        CUDA::cudart
)

add_executable(vector_add_cuda_test
    tests/vector_add_cuda_test.cu
)

target_link_libraries(vector_add_cuda_test
    PRIVATE
        cuda_vector_add
        GTest::gtest_main
)

set_target_properties(vector_add_cuda_test PROPERTIES
    CUDA_ARCHITECTURES 75
)

add_executable(reduction_cuda_test
    tests/reduction_cuda_test.cu
)

target_link_libraries(reduction_cuda_test
    PRIVATE
        cuda_reductions
        GTest::gtest_main
)

set_target_properties(reduction_cuda_test PROPERTIES
    CUDA_ARCHITECTURES 75
)

add_executable(softmax_cuda_test
    tests/softmax_cuda_test.cu
)

target_link_libraries(softmax_cuda_test
    PRIVATE
        cuda_reductions
        GTest::gtest_main
)

set_target_properties(softmax_cuda_test PROPERTIES
    CUDA_ARCHITECTURES 75
)

include(GoogleTest)
gtest_discover_tests(vector_add_cuda_test)
gtest_discover_tests(reduction_cuda_test)
gtest_discover_tests(softmax_cuda_test)

add_executable(vector_add_cpu_gpu_benchmark
    benchmarks/vector_add_cpu_gpu_benchmark.cu
)

target_link_libraries(vector_add_cpu_gpu_benchmark
    PRIVATE
        cuda_vector_add
        benchmark::benchmark
)

set_target_properties(vector_add_cpu_gpu_benchmark PROPERTIES
    CUDA_ARCHITECTURES 75
)

add_executable(reduction_benchmark
    benchmarks/reduction_benchmark.cu
)

target_link_libraries(reduction_benchmark
    PRIVATE
        cuda_reductions
        benchmark::benchmark
)

set_target_properties(reduction_benchmark PROPERTIES
    CUDA_ARCHITECTURES 75
)

add_executable(softmax_benchmark
    benchmarks/softmax_benchmark.cu
)

target_link_libraries(softmax_benchmark
    PRIVATE
        cuda_reductions
        benchmark::benchmark
)

set_target_properties(softmax_benchmark PROPERTIES
    CUDA_ARCHITECTURES 75
)