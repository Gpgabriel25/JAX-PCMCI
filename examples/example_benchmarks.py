"""
Example: GPU Benchmarking and Performance Comparison
=====================================================

This example benchmarks JAX-PCMCI performance across different
configurations to help you choose optimal settings for your hardware.
"""

import jax
import jax.numpy as jnp
import time

jax.config.update("jax_enable_x64", True)

from jax_pcmci import ParCorr, CMIKnn, set_device, get_device_info
from jax_pcmci.parallel import benchmark_parallel_modes, ParallelConfig, batch_independence_tests


def benchmark_sample_sizes():
    """Benchmark how performance scales with sample size."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Scaling with Sample Size")
    print("=" * 60)
    
    test = ParCorr()
    n_tests = 1000
    sample_sizes = [100, 500, 1000, 2000, 5000]
    
    print(f"\nRunning {n_tests} independence tests...")
    print(f"{'Samples':<15} {'Time (s)':<15} {'Tests/sec':<15}")
    print("-" * 45)
    
    for n_samples in sample_sizes:
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        X = jax.random.normal(keys[0], (n_tests, n_samples))
        Y = jax.random.normal(keys[1], (n_tests, n_samples))
        
        # Warm-up
        _ = test.run_batch(X[:10], Y[:10])
        
        # Time
        start = time.perf_counter()
        stats, pvals = test.run_batch(X, Y)
        _ = float(stats[0])  # Force evaluation
        elapsed = time.perf_counter() - start
        
        throughput = n_tests / elapsed
        print(f"{n_samples:<15} {elapsed:<15.3f} {throughput:<15.0f}")


def benchmark_test_types():
    """Compare performance of different independence tests."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Independence Test Comparison")
    print("=" * 60)
    
    n_samples = 200  # Reduced for 6GB VRAM
    n_tests = 200
    
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 2)
    X = jax.random.normal(keys[0], (n_tests, n_samples))
    Y = jax.random.normal(keys[1], (n_tests, n_samples))
    
    tests = [
        ("ParCorr (analytic)", ParCorr(significance='analytic')),
        ("CMI-kNN (k=5)", CMIKnn(k=5)),  # Statistic only, no p-value
    ]
    
    print(f"\n{n_tests} tests, {n_samples} samples each")
    print(f"{'Test':<25} {'Time (s)':<15} {'Tests/sec':<15}")
    print("-" * 55)
    
    for name, test in tests:
        # Warm-up
        try:
            _ = test.run_batch(X[:5], Y[:5])
        except:
            _ = test.run(X[0], Y[0])
        
        start = time.perf_counter()
        try:
            stats, pvals = test.run_batch(X, Y)
            _ = float(stats[0])
        except:
            # Fallback to sequential
            for i in range(n_tests):
                _ = test.run(X[i], Y[i])
        elapsed = time.perf_counter() - start
        
        throughput = n_tests / elapsed
        print(f"{name:<25} {elapsed:<15.3f} {throughput:<15.0f}")


def benchmark_precision():
    """Compare float32 vs float64 performance."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Precision Comparison (float32 vs float64)")
    print("=" * 60)
    
    test = ParCorr()
    n_tests = 2000
    n_samples = 500
    
    for dtype_name, dtype in [("float32", jnp.float32), ("float64", jnp.float64)]:
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        X = jax.random.normal(keys[0], (n_tests, n_samples)).astype(dtype)
        Y = jax.random.normal(keys[1], (n_tests, n_samples)).astype(dtype)
        
        # Warm-up
        _ = test.run_batch(X[:10], Y[:10])
        
        start = time.perf_counter()
        stats, pvals = test.run_batch(X, Y)
        _ = float(stats[0])
        elapsed = time.perf_counter() - start
        
        throughput = n_tests / elapsed
        print(f"{dtype_name}: {elapsed:.3f}s ({throughput:.0f} tests/sec)")


def main():
    # Device info
    info = get_device_info()
    print("=" * 60)
    print("JAX-PCMCI Performance Benchmarks")
    print("=" * 60)
    print(f"\nDevice: {info['default_backend']}")
    print(f"GPUs available: {info['gpu_count']}")
    print(f"CPUs available: {info['cpu_count']}")
    
    # Run benchmarks
    benchmark_sample_sizes()
    benchmark_test_types()
    benchmark_precision()
    
    # Parallel mode comparison
    print("\n" + "=" * 60)
    print("BENCHMARK: Parallelization Modes")
    print("=" * 60)
    
    results = benchmark_parallel_modes(ParCorr(), n_samples=500, n_tests=1000)
    for mode, result in results.items():
        print(f"{mode}: {result.total_time_s:.3f}s ({result.tests_per_second:.0f} tests/sec)")
    
    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
