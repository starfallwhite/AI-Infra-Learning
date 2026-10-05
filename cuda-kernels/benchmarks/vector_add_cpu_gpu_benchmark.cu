#include <benchmark/benchmark.h>

#include "vector_add.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace
{

    constexpr std::size_t elements = 1 << 24;

    std::vector<float> make_input(float offset)
    {
        std::vector<float> values(elements);
        for (std::size_t index = 0; index < elements; ++index)
        {
            values[index] = static_cast<float>(index) * 0.001F + offset;
        }
        return values;
    }

    static void BM_CpuVectorAdd(benchmark::State &state)
    {
        const auto left = make_input(1.0F);
        const auto right = make_input(2.0F);
        std::vector<float> output(elements);

        for (auto _ : state)
        {
            cpu_vector_add(left.data(), right.data(), output.data(), elements);
            benchmark::DoNotOptimize(output.data());
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * 3 * sizeof(float));
    }

    static void BM_GpuVectorAddEndToEnd(benchmark::State &state)
    {
        const auto left = make_input(1.0F);
        const auto right = make_input(2.0F);

        for (auto _ : state)
        {
            const auto output = cuda_vector_add(left, right);
            benchmark::DoNotOptimize(output.data());
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * 3 * sizeof(float));
    }

    static void BM_GpuVectorAddKernel(benchmark::State &state)
    {
        const auto left = make_input(1.0F);
        const auto right = make_input(2.0F);
        float *device_left = nullptr;
        float *device_right = nullptr;
        float *device_output = nullptr;
        const std::size_t bytes = elements * sizeof(float);
        cudaMalloc(&device_left, bytes);
        cudaMalloc(&device_right, bytes);
        cudaMalloc(&device_output, bytes);
        cudaMemcpy(device_left, left.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(device_right, right.data(), bytes, cudaMemcpyHostToDevice);

        for (auto _ : state)
        {
            cuda_vector_add_device(device_left, device_right, device_output,
                                   elements);
            cudaDeviceSynchronize();
            benchmark::DoNotOptimize(device_output);
        }
        state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                                elements * 3 * sizeof(float));
        cudaFree(device_left);
        cudaFree(device_right);
        cudaFree(device_output);
    }

} // namespace

BENCHMARK(BM_CpuVectorAdd);
BENCHMARK(BM_GpuVectorAddEndToEnd);
BENCHMARK(BM_GpuVectorAddKernel);
BENCHMARK_MAIN();