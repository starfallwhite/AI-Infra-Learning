#include <gtest/gtest.h>

#include "vector_add.cuh"

#include <cmath>
#include <vector>

TEST(VectorAddCudaTest, MatchesCpuResult)
{
    const std::vector<float> left{1.0F, -2.0F, 3.5F, 0.0F, 8.0F};
    const std::vector<float> right{2.0F, 4.0F, -0.5F, 9.0F, -3.0F};
    std::vector<float> expected(left.size());
    cpu_vector_add(left.data(), right.data(), expected.data(), expected.size());

    const std::vector<float> actual = cuda_vector_add(left, right);

    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_FLOAT_EQ(actual[index], expected[index]);
    }
}

TEST(VectorAddCudaTest, HandlesEmptyVectors)
{
    EXPECT_TRUE(cuda_vector_add({}, {}).empty());
}

TEST(VectorAddCudaTest, RejectsDifferentSizes)
{
    EXPECT_THROW(cuda_vector_add({1.0F}, {1.0F, 2.0F}), std::invalid_argument);
}