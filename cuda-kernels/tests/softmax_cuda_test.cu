#include <gtest/gtest.h>

#include "softmax.cuh"

#include <cmath>
#include <stdexcept>
#include <vector>

TEST(SoftmaxCudaTest, MatchesCpuResult)
{
    const std::vector<float> input{1.0F, 2.0F, 3.0F,
                                   -1.0F, 0.0F, 1.0F};
    // 将 GPU 输出与独立的 CPU 实现进行比较。
    std::vector<float> expected(input.size());
    cpu_softmax(input.data(), expected.data(), 2, 3);

    const std::vector<float> actual = cuda_softmax(input, 2, 3);

    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_NEAR(actual[index], expected[index], 1e-6F);
    }
}

TEST(SoftmaxCudaTest, EachRowSumsToOne)
{
    const std::vector<float> input{100.0F, 101.0F, 102.0F,
                                   -100.0F, -101.0F, -102.0F};
    const std::vector<float> actual = cuda_softmax(input, 2, 3);

    // softmax 的每一行都应当构成概率分布。
    EXPECT_NEAR(actual[0] + actual[1] + actual[2], 1.0F, 1e-6F);
    EXPECT_NEAR(actual[3] + actual[4] + actual[5], 1.0F, 1e-6F);
}

TEST(SoftmaxCudaTest, RejectsInvalidShape)
{
    EXPECT_THROW(cuda_softmax({1.0F, 2.0F}, 1, 3), std::invalid_argument);
}