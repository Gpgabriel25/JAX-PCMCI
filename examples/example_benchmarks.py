"""
Example: GPU Benchmarking and Performance Comparison
=====================================================

This example benchmarks JAX-PCMCI performance across different
configurations to help you choose optimal settings for your hardware.
"""

from __future__ import annotations

import os
import signal
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp

# Precision/device are configurable via env for faster runs
BENCH_PRECISION = os.environ.get("PCMCI_BENCH_PRECISION", "float32").lower()
SKIP_CMI = os.environ.get("PCMCI_BENCH_SKIP_CMI", "0").lower() in {"1", "true", "yes"}
jax.config.update("jax_enable_x64", BENCH_PRECISION == "float64")

from jax_pcmci import ParCorr, CMIKnn, set_device, get_device_info
from jax_pcmci.parallel import benchmark_parallel_modes, ParallelConfig, batch_independence_tests

# Optional device override (cpu/gpu/tpu/auto)
set_device(os.environ.get("PCMCI_BENCH_DEVICE", "auto"))


@contextmanager
def bounded_timeout(seconds: int, label: str):
    """Enforce a wall-clock timeout for a benchmark section."""
    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"{label} exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


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
        _ = jax.block_until_ready(_[0])
        
        # Time
        start = time.perf_counter()
        stats, pvals = test.run_batch(X, Y)
        jax.block_until_ready(stats)
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
    
    tests = [("ParCorr (analytic)", ParCorr(significance='analytic'))]

    if not SKIP_CMI:
        # Use fast analytic p-values by default to keep runtime reasonable
        tests.append(("CMI-kNN (k=5, analytic)", CMIKnn(k=5, significance="analytic")))
    else:
        print("[skip] CMI-kNN section disabled via PCMCI_BENCH_SKIP_CMI")
    
    print(f"\n{n_tests} tests, {n_samples} samples each")
    print(f"{'Test':<25} {'Time (s)':<15} {'Tests/sec':<15}")
    print("-" * 55)
    
    for name, test in tests:
        # Warm-up
        try:
            _ = test.run_batch(X[:5], Y[:5])
            jax.block_until_ready(_[0])
        except:
            _ = test.run(X[0], Y[0])
        
        start = time.perf_counter()
        try:
            stats, pvals = test.run_batch(X, Y)
            jax.block_until_ready(stats)
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
    
    x64_enabled = jax.config.read("jax_enable_x64")

    for dtype_name, dtype in [("float32", jnp.float32), ("float64", jnp.float64)]:
        if dtype_name == "float64" and not x64_enabled:
            print("[skip] float64 benchmark: jax_enable_x64 is false on this device")
            continue
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        X = jax.random.normal(keys[0], (n_tests, n_samples)).astype(dtype)
        Y = jax.random.normal(keys[1], (n_tests, n_samples)).astype(dtype)
        
        # Warm-up
        _ = test.run_batch(X[:10], Y[:10])
        jax.block_until_ready(_[0])
        
        start = time.perf_counter()
        stats, pvals = test.run_batch(X, Y)
        jax.block_until_ready(stats)
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
    print(f"Precision: {BENCH_PRECISION}")

    timeout_s = int(os.environ.get("PCMCI_BENCH_TIMEOUT", "110"))
    
    # Run benchmarks
    benchmarks = [
        ("sample size scaling", benchmark_sample_sizes),
        ("test type comparison", benchmark_test_types),
        ("precision comparison", benchmark_precision),
    ]

    for label, fn in benchmarks:
        try:
            with bounded_timeout(timeout_s, label):
                fn()
        except TimeoutError as exc:
            print(f"[timeout] Skipped {label}: {exc}")
    
    # Parallel mode comparison
    print("\n" + "=" * 60)
    print("BENCHMARK: Parallelization Modes")
    print("=" * 60)
    try:
        with bounded_timeout(timeout_s, "parallel modes"):
            results = benchmark_parallel_modes(ParCorr(), n_samples=500, n_tests=1000)
            for mode, result in results.items():
                print(f"{mode}: {result.total_time_s:.3f}s ({result.tests_per_second:.0f} tests/sec)")
    except TimeoutError as exc:
        print(f"[timeout] Skipped parallel modes: {exc}")
    
    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
