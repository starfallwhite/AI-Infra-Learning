#include <benchmark/benchmark.h>

#include "softmax.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace
{

    constexpr std::size_t rows = 4096;
    constexpr std::size_t columns = 1024;
    constexpr std::size_t elements = rows * columns;

    std::vector<float> make_input()
    {
        std::vector<float> input(elements);
        for (std::size_t index = 0; index < elements; ++index)
        {
            input[index] = static_cast<float>(index % columns) * 0.001F;
        }
        return input;
    }

    static void BM_CpuSoftmax(benchmark::State &state)
    {
        const auto input = make_input();
        std::vector<float> output(elements);
        for (auto _ : state)
        {
            cpu_softmax(input.data(), output.data(), rows, columns);
            benchmark::DoNotOptimize(output.data());
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * 2 * sizeof(float));
    }

    static void BM_GpuSoftmaxEndToEnd(benchmark::State &state)
    {
        const auto input = make_input();
        for (auto _ : state)
        {
            // 端到端计时包含显存申请和两个方向的数据传输。
            const auto output = cuda_softmax(input, rows, columns);
            benchmark::DoNotOptimize(output.data());
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * 2 * sizeof(float));
    }

    static void BM_GpuSoftmaxKernel(benchmark::State &state)
    {
        const auto input = make_input();
        const std::size_t bytes = elements * sizeof(float);
        float *device_input = nullptr;
        float *device_output = nullptr;
        cudaMalloc(&device_input, bytes);
        cudaMalloc(&device_output, bytes);
        cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice);

        for (auto _ : state)
        {
            // 复用输入和输出缓冲区，只测量 kernel 的耗时。
            cuda_softmax_device(device_input, device_output, rows, columns);
            cudaDeviceSynchronize();
            benchmark::DoNotOptimize(device_output);
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * 2 * sizeof(float));
        cudaFree(device_input);
        cudaFree(device_output);
    }

} // 匿名命名空间

BENCHMARK(BM_CpuSoftmax);
BENCHMARK(BM_GpuSoftmaxEndToEnd);
BENCHMARK(BM_GpuSoftmaxKernel);
BENCHMARK_MAIN();