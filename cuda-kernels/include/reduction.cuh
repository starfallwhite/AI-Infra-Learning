#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

float cpu_reduce_sum(const float *input, std::size_t size);

void cuda_reduce_sum_device(const float *input, float *output,
                            std::size_t size,
                            cudaStream_t stream = nullptr);

float cuda_reduce_sum(const std::vector<float> &input);