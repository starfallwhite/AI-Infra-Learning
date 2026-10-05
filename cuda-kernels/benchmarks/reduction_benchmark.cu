#include <benchmark/benchmark.h>

#include "reduction.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace
{

    constexpr std::size_t elements = 1 << 24;

    std::vector<float> make_input()
    {
        std::vector<float> input(elements);
        for (std::size_t index = 0; index < elements; ++index)
        {
            input[index] = static_cast<float>(index % 100) * 0.01F;
        }
        return input;
    }

    static void BM_CpuReduction(benchmark::State &state)
    {
        const auto input = make_input();
        float output = 0.0F;
        for (auto _ : state)
        {
            output = cpu_reduce_sum(input.data(), elements);
            benchmark::DoNotOptimize(output);
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * sizeof(float));
    }

    static void BM_GpuReductionEndToEnd(benchmark::State &state)
    {
        const auto input = make_input();
        for (auto _ : state)
        {
            // 包含显存申请、主机到设备/设备到主机拷贝、同步和释放。
            const float output = cuda_reduce_sum(input);
            benchmark::DoNotOptimize(output);
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * sizeof(float));
    }

    static void BM_GpuReductionKernel(benchmark::State &state)
    {
        const auto input = make_input();
        const std::size_t bytes = elements * sizeof(float);
        float *device_input = nullptr;
        float *device_output = nullptr;
        cudaMalloc(&device_input, bytes);
        cudaMalloc(&device_output, sizeof(float));
        cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice);

        for (auto _ : state)
        {
            // 将内存准备放在循环外，只测量 kernel 的耗时。
            cuda_reduce_sum_device(device_input, device_output, elements);
            cudaDeviceSynchronize();
            benchmark::DoNotOptimize(device_output);
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * sizeof(float));
        cudaFree(device_input);
        cudaFree(device_output);
    }

} // 匿名命名空间

BENCHMARK(BM_CpuReduction);
BENCHMARK(BM_GpuReductionEndToEnd);
BENCHMARK(BM_GpuReductionKernel);
BENCHMARK_MAIN();