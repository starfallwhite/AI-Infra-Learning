#include "vector_add.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace
{

    __global__ void vector_add_kernel(const float *left, const float *right,
                                      float *output, std::size_t size)
    {
        // 一个线程负责一个输出元素；边界检查处理最后一个 block 中多余的线程。
        const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size)
        {
            output[index] = left[index] + right[index];
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

void cpu_vector_add(const float *left, const float *right, float *output,
                    std::size_t size)
{
    // 这个 CPU 标量循环作为正确性和性能基线。
    for (std::size_t index = 0; index < size; ++index)
    {
        output[index] = left[index] + right[index];
    }
}

void cuda_vector_add_device(const float *left, const float *right, float *output,
                            std::size_t size, cudaStream_t stream)
{
    if (size == 0)
    {
        return;
    }

    constexpr unsigned int threads_per_block = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (size + threads_per_block - 1) / threads_per_block);
    // stream 参数允许调用方将 kernel 启动与其他工作重叠执行。
    vector_add_kernel<<<blocks, threads_per_block, 0, stream>>>(left, right,
                                                                output, size);
    check_cuda(cudaGetLastError(), "launch vector_add_kernel");
}

std::vector<float> cuda_vector_add(const std::vector<float> &left,
                                   const std::vector<float> &right)
{
    if (left.size() != right.size())
    {
        throw std::invalid_argument("vector sizes must match");
    }

    const std::size_t bytes = left.size() * sizeof(float);
    std::vector<float> output(left.size());
    float *device_left = nullptr;
    float *device_right = nullptr;
    float *device_output = nullptr;

    try
    {
        // 这个便捷接口包含显存申请和主机/设备数据传输。
        // benchmark 另提供了只测 kernel 的底层路径。
        check_cuda(cudaMalloc(&device_left, bytes), "allocate left vector");
        check_cuda(cudaMalloc(&device_right, bytes), "allocate right vector");
        check_cuda(cudaMalloc(&device_output, bytes), "allocate output vector");
        check_cuda(cudaMemcpy(device_left, left.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "copy left vector");
        check_cuda(cudaMemcpy(device_right, right.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "copy right vector");

        cuda_vector_add_device(device_left, device_right, device_output,
                               left.size());
        check_cuda(cudaDeviceSynchronize(), "synchronize vector_add_kernel");
        check_cuda(cudaMemcpy(output.data(), device_output, bytes,
                              cudaMemcpyDeviceToHost),
                   "copy output vector");
    }
    catch (...)
    {
        cudaFree(device_left);
        cudaFree(device_right);
        cudaFree(device_output);
        throw;
    }

    cudaFree(device_left);
    cudaFree(device_right);
    cudaFree(device_output);
    return output;
}