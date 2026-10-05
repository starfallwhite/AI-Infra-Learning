#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

void cpu_softmax(const float *input, float *output, std::size_t rows,
                 std::size_t columns);

void cuda_softmax_device(const float *input, float *output, std::size_t rows,
                         std::size_t columns,
                         cudaStream_t stream = nullptr);

std::vector<float> cuda_softmax(const std::vector<float> &input,
                                std::size_t rows, std::size_t columns);