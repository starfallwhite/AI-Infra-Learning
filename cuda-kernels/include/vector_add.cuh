#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

void cpu_vector_add(const float *left, const float *right, float *output,
                    std::size_t size);

void cuda_vector_add_device(const float *left, const float *right, float *output,
                            std::size_t size, cudaStream_t stream = nullptr);

std::vector<float> cuda_vector_add(const std::vector<float> &left,
                                   const std::vector<float> &right);