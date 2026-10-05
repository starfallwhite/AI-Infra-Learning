#include <gtest/gtest.h>

#include "reduction.cuh"

#include <vector>

TEST(ReductionCudaTest, MatchesCpuResult)
{
    const std::vector<float> input{1.0F, -2.0F, 3.5F, 4.0F, 8.0F};

    // 使用 CPU 实现的结果作为参考答案。
    const float expected = cpu_reduce_sum(input.data(), input.size());
    const float actual = cuda_reduce_sum(input);

    EXPECT_FLOAT_EQ(actual, expected);
}

TEST(ReductionCudaTest, HandlesEmptyInput)
{
    EXPECT_FLOAT_EQ(cuda_reduce_sum({}), 0.0F);
}