#include <gtest/gtest.h>
#include "vector_add.h"

TEST(VectorAddTest, AddsElementsCorrectlu)
{
    EXPECT_EQ(vector_add(1, 2), 3);
}