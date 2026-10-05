#include "reduction.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace
{

    __device__ float warp_reduce_sum(float value)
    {
        // 使用 warp shuffle 在 warp 内交换数据，无需访问 shared memory。
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
        {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        return value;
    }

    __global__ void reduce_sum_kernel(const float *input, float *output,
                                      std::size_t size)
    {
        // 每个 block 计算局部和，再用 atomicAdd 合并所有 block 的结果。
        __shared__ float warp_sums[32];
        const unsigned int thread_id = threadIdx.x;
        const unsigned int lane = thread_id % warpSize;
        const unsigned int warp_id = thread_id / warpSize;

        float sum = 0.0F;
        const std::size_t global_thread_id =
            blockIdx.x * blockDim.x + thread_id;
        const std::size_t stride = blockDim.x * gridDim.x;
        // 当输入大于线程总数时，每个线程可以处理多个元素，保持 kernel 的可扩展性。
        for (std::size_t index = global_thread_id; index < size;
             index += stride)
        {
            sum += input[index];
        }

        sum = warp_reduce_sum(sum);
        if (lane == 0)
        {
            warp_sums[warp_id] = sum;
        }
        __syncthreads();

        // 第一个 warp 继续归约当前 block 中每个 warp 的局部结果。
        sum = thread_id < blockDim.x / warpSize ? warp_sums[thread_id] : 0.0F;
        if (warp_id == 0)
        {
            sum = warp_reduce_sum(sum);
        }
        if (thread_id == 0)
        {
            atomicAdd(output, sum);
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

float cpu_reduce_sum(const float *input, std::size_t size)
{
    float sum = 0.0F;
    for (std::size_t index = 0; index < size; ++index)
    {
        sum += input[index];
    }
    return sum;
}

void cuda_reduce_sum_device(const float *input, float *output,
                            std::size_t size, cudaStream_t stream)
{
    if (size == 0)
    {
        return;
    }

    constexpr unsigned int threads_per_block = 256;
    constexpr unsigned int elements_per_block = 8;
    // 限制 grid 大小，同时为大规模输入启动足够的 block。
    const std::size_t requested_blocks =
        (size + threads_per_block * elements_per_block - 1) /
        (threads_per_block * elements_per_block);
    const unsigned int blocks = static_cast<unsigned int>(
        requested_blocks < 1024 ? requested_blocks : 1024);
    check_cuda(cudaMemsetAsync(output, 0, sizeof(float), stream),
               "clear reduction output");
    reduce_sum_kernel<<<blocks, threads_per_block, 0, stream>>>(input, output,
                                                                size);
    check_cuda(cudaGetLastError(), "launch reduce_sum_kernel");
}

float cuda_reduce_sum(const std::vector<float> &input)
{
    if (input.empty())
    {
        return 0.0F;
    }

    const std::size_t bytes = input.size() * sizeof(float);
    float *device_input = nullptr;
    float *device_output = nullptr;
    float output = 0.0F;

    try
    {
        check_cuda(cudaMalloc(&device_input, bytes), "allocate reduction input");
        check_cuda(cudaMalloc(&device_output, sizeof(float)),
                   "allocate reduction output");
        check_cuda(cudaMemcpy(device_input, input.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "copy reduction input");
        cuda_reduce_sum_device(device_input, device_output, input.size());
        check_cuda(cudaDeviceSynchronize(), "synchronize reduction kernel");
        check_cuda(cudaMemcpy(&output, device_output, sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy reduction output");
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