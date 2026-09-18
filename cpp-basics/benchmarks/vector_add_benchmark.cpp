#include <benchmark/benchmark.h>
#include "vector_add.h"

static void BM_VectorAdd(benchmark::State &state)
{
    for (auto _ : state)
    {
        benchmark::DoNotOptimize(vector_add(1, 2));
    }
}

BENCHMARK(BM_VectorAdd);
BENCHMARK_MAIN();