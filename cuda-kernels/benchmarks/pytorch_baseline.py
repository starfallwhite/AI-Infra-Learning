"""Benchmark PyTorch CPU and CUDA implementations against the CUDA kernels."""

import argparse
import time

import torch


def measure(function, warmup, iterations):
    for _ in range(warmup):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    cpu_input = torch.rand(args.rows, args.columns)
    cpu_reduction_input = torch.rand(1 << 24)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("CPU")
    print("  reduction: %.3f ms" % measure(
        lambda: torch.sum(cpu_reduction_input), 5, args.iterations))
    print("  softmax:   %.3f ms" % measure(
        lambda: torch.softmax(cpu_input, dim=-1), 5, args.iterations))

    if not torch.cuda.is_available():
        return

    gpu_input = cpu_input.cuda()
    gpu_reduction_input = cpu_reduction_input.cuda()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("  reduction: %.3f ms" % measure(
        lambda: torch.sum(gpu_reduction_input), 10, args.iterations))
    print("  softmax:   %.3f ms" % measure(
        lambda: torch.softmax(gpu_input, dim=-1), 10, args.iterations))


if __name__ == "__main__":
    main()