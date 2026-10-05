#include "softmax.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

    __device__ float warp_reduce_max(float value)
    {
        // warp shuffle 在寄存器中完成第一阶段归约。
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            value = fmaxf(value,
                          __shfl_down_sync(0xffffffff, value, offset));
        }
        return value;
    }

    __device__ float warp_reduce_sum(float value)
    {
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        return value;
    }

    __device__ float block_reduce_max(float value, float *warp_values)
    {
        // 每个 warp 写入一个结果，再由 warp 0 继续归约这些结果。
        const unsigned int lane = threadIdx.x % warpSize;
        const unsigned int warp_id = threadIdx.x / warpSize;
        value = warp_reduce_max(value);
        if (lane == 0)
        {
            warp_values[warp_id] = value;
        }
        __syncthreads();

        value = threadIdx.x < blockDim.x / warpSize
                    ? warp_values[threadIdx.x]
                    : -FLT_MAX;
        if (warp_id == 0)
        {
            value = warp_reduce_max(value);
        }
        if (threadIdx.x == 0)
        {
            warp_values[0] = value;
        }
        return value;
    }

    __device__ float block_reduce_sum(float value, float *warp_values)
    {
        // 这是 block_reduce_max 对应的求和版本。
        const unsigned int lane = threadIdx.x % warpSize;
        const unsigned int warp_id = threadIdx.x / warpSize;
        value = warp_reduce_sum(value);
        if (lane == 0)
        {
            warp_values[warp_id] = value;
        }
        __syncthreads();

        value = threadIdx.x < blockDim.x / warpSize ? warp_values[threadIdx.x]
                                                    : 0.0F;
        if (warp_id == 0)
        {
            value = warp_reduce_sum(value);
        }
        if (threadIdx.x == 0)
        {
            warp_values[0] = value;
        }
        return value;
    }

    __global__ void softmax_kernel(const float *input, float *output,
                                   std::size_t rows, std::size_t columns)
    {
        // 一个 block 负责一行，所有线程协作计算该行的最大值和归一化分母。
        __shared__ float warp_values[32];
        const std::size_t row = blockIdx.x;
        if (row >= rows)
        {
            return;
        }

        const std::size_t row_offset = row * columns;
        float row_max = -FLT_MAX;
        // 减去该行最大值，避免 expf 溢出。
        for (std::size_t column = threadIdx.x; column < columns;
             column += blockDim.x)
        {
            row_max = fmaxf(row_max, input[row_offset + column]);
        }
        block_reduce_max(row_max, warp_values);
        __syncthreads();
        row_max = warp_values[0];
        __syncthreads();

        float row_sum = 0.0F;
        for (std::size_t column = threadIdx.x; column < columns;
             column += blockDim.x)
        {
            row_sum += expf(input[row_offset + column] - row_max);
        }
        block_reduce_sum(row_sum, warp_values);
        __syncthreads();
        row_sum = warp_values[0];
        __syncthreads();

        for (std::size_t column = threadIdx.x; column < columns;
             column += blockDim.x)
        {
            output[row_offset + column] =
                expf(input[row_offset + column] - row_max) / row_sum;
        }
    }

    void check_cuda(cudaError_t error, const char *operation)
    {
        if (error != cudaSuccess)
        {
            throw std::runtime_error(std::string(operation) + ": " +
                                     cudaGetErrorString(error));
        }
    }

} // 匿名命名空间

void cpu_softmax(const float *input, float *output, std::size_t rows,
                 std::size_t columns)
{
    // CPU 基线也使用减去最大值的数值稳定算法。
    for (std::size_t row = 0; row < rows; ++row)
    {
        const std::size_t row_offset = row * columns;
        float row_max = -std::numeric_limits<float>::infinity();
        for (std::size_t column = 0; column < columns; ++column)
        {
            row_max = std::max(row_max, input[row_offset + column]);
        }

        float row_sum = 0.0F;
        for (std::size_t column = 0; column < columns; ++column)
        {
            output[row_offset + column] =
                std::exp(input[row_offset + column] - row_max);
            row_sum += output[row_offset + column];
        }
        for (std::size_t column = 0; column < columns; ++column)
        {
            output[row_offset + column] /= row_sum;
        }
    }
}

void cuda_softmax_device(const float *input, float *output, std::size_t rows,
                         std::size_t columns, cudaStream_t stream)
{
    if (rows == 0 || columns == 0)
    {
        return;
    }

    constexpr unsigned int threads_per_block = 256;
    // Grid 的 X 维直接映射到行，每个 block 计算一行。
    softmax_kernel<<<static_cast<unsigned int>(rows), threads_per_block, 0,
                     stream>>>(input, output, rows, columns);
    check_cuda(cudaGetLastError(), "launch softmax_kernel");
}

std::vector<float> cuda_softmax(const std::vector<float> &input,
                                std::size_t rows, std::size_t columns)
{
    if (rows * columns != input.size())
    {
        throw std::invalid_argument("softmax shape does not match input size");
    }
    if (input.empty())
    {
        return {};
    }

    const std::size_t bytes = input.size() * sizeof(float);
    std::vector<float> output(input.size());
    float *device_input = nullptr;
    float *device_output = nullptr;

    try
    {
        check_cuda(cudaMalloc(&device_input, bytes), "allocate softmax input");
        check_cuda(cudaMalloc(&device_output, bytes),
                   "allocate softmax output");
        check_cuda(cudaMemcpy(device_input, input.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "copy softmax input");
        cuda_softmax_device(device_input, device_output, rows, columns);
        check_cuda(cudaDeviceSynchronize(), "synchronize softmax kernel");
        check_cuda(cudaMemcpy(output.data(), device_output, bytes,
                              cudaMemcpyDeviceToHost),
                   "copy softmax output");
    }
    catch (...)
    {
        cudaFree(device_input);
        cudaFree(device_output);
        throw;
    }

    cudaFree(device_input);
    cudaFree(device_output);
    return output;
}