add_library(vector_add
    src/vector_add.cpp
)

target_include_directories(vector_add
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(vector_add_test
    tests/vector_add_test.cpp
)

target_link_libraries(vector_add_test
    PRIVATE
        vector_add
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(vector_add_test)

add_executable(vector_add_benchmark
    benchmarks/vector_add_benchmark.cpp
)

target_link_libraries(vector_add_benchmark
    PRIVATE
        vector_add
        benchmark::benchmark
)