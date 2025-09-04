"""
Performance benchmarks for correlation sampler
"""

import time
import torch
import numpy as np
from spatial_correlation_sampler import (
    SpatialCorrelationSampler,
    OptimizedSpatialCorrelationSampler
)


def benchmark_implementation(
        sampler,
        input_shape,
        num_iterations=100,
        warmup=10,
        device='cuda'
):
    """Benchmark a correlation sampler implementation"""

    # Create random inputs
    input1 = torch.randn(*input_shape, device=device)
    input2 = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(warmup):
        _ = sampler(input1, input2)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        output = sampler(input1, input2)

        if device == 'cuda':
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # Convert to ms

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def run_benchmarks():
    """Run all benchmarks"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on {device}")
    print("=" * 60)

    configs = [
        # MM-Tracker configuration
        {
            'name': 'MM-Tracker',
            'shape': (1, 512, 19, 34),
            'kernel_size': 9,
            'patch_size': 1,
            'padding': 4
        },
        # FlowNetC configuration
        {
            'name': 'FlowNetC',
            'shape': (4, 256, 48, 64),
            'kernel_size': 1,
            'patch_size': 21,
            'padding': 0,
            'dilation_patch': 2
        },
        # Small configuration
        {
            'name': 'Small',
            'shape': (8, 128, 32, 32),
            'kernel_size': 5,
            'patch_size': 1,
            'padding': 2
        }
    ]

    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print(f"  Input shape: {config['shape']}")
        print(f"  Kernel size: {config['kernel_size']}")
        print(f"  Patch size: {config['patch_size']}")

        # Standard implementation
        sampler = SpatialCorrelationSampler(
            kernel_size=config['kernel_size'],
            patch_size=config['patch_size'],
            padding=config['padding']
        )

        results = benchmark_implementation(
            sampler, config['shape'], device=device
        )

        print(f"\n  Standard Implementation:")
        print(f"    Mean: {results['mean']:.3f} ms")
        print(f"    Std:  {results['std']:.3f} ms")
        print(f"    Min:  {results['min']:.3f} ms")
        print(f"    Max:  {results['max']:.3f} ms")

        # Optimized implementation (if applicable)
        if config['patch_size'] == 1:
            optimized = OptimizedSpatialCorrelationSampler(
                kernel_size=config['kernel_size'],
                padding=config['padding']
            )

            results_opt = benchmark_implementation(
                optimized, config['shape'], device=device
            )

            print(f"\n  Optimized Implementation:")
            print(f"    Mean: {results_opt['mean']:.3f} ms")
            print(f"    Std:  {results_opt['std']:.3f} ms")
            print(f"    Min:  {results_opt['min']:.3f} ms")
            print(f"    Max:  {results_opt['max']:.3f} ms")

            speedup = results['mean'] / results_opt['mean']
            print(f"    Speedup: {speedup:.2f}x")

        print("-" * 40)


if __name__ == "__main__":
    run_benchmarks()