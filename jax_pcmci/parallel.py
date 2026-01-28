"""
Parallel Utilities Module for JAX-PCMCI
=======================================

This module provides utilities for parallel computation across devices
using JAX's vmap (vectorization) and pmap (multi-device parallelism).

The utilities here enable efficient batch processing of independence
tests and algorithm iterations across multiple cores, GPUs, or TPUs.

Key Features:
    - Automatic batch size optimization
    - Memory-efficient chunked processing
    - Multi-GPU/TPU distribution with pmap
    - Progress tracking for long computations

Example:
    >>> from jax_pcmci.parallel import batch_independence_tests
    >>> 
    >>> # Run many independence tests in parallel
    >>> results = batch_independence_tests(
    ...     test=ParCorr(),
    ...     X_batch=X_data,  # (n_tests, n_samples)
    ...     Y_batch=Y_data,
    ...     chunk_size=1000
    ... )
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, pmap, jit

from .config import PCMCIConfig


@dataclass
class ParallelConfig:
    """
    Configuration for parallel execution.
    
    This configuration controls how JAX-PCMCI distributes work
    across available compute resources.
    
    Attributes
    ----------
    mode : str
        Parallelization mode:
        - 'auto': Automatically choose based on workload and hardware
        - 'vmap': Vectorize across a single device (good for GPU)
        - 'pmap': Distribute across multiple devices
        - 'sequential': No parallelization (for debugging)
    chunk_size : int
        Maximum number of operations per batch. Larger values use
        more memory but have less overhead. Default auto-tunes.
    n_devices : int or None
        Number of devices to use for pmap. None = all available.
    memory_limit_gb : float
        Approximate memory limit per device in GB. Used to auto-tune
        chunk_size to avoid OOM errors.
    progress_callback : callable or None
        Function called with (completed, total) to track progress.
        
    Examples
    --------
    >>> config = ParallelConfig(
    ...     mode='vmap',
    ...     chunk_size=5000,
    ...     memory_limit_gb=8.0
    ... )
    """
    mode: str = 'auto'
    chunk_size: int = 10000
    n_devices: Optional[int] = None
    memory_limit_gb: float = 8.0
    progress_callback: Optional[Callable[[int, int], None]] = None
    
    def __post_init__(self):
        valid_modes = {'auto', 'vmap', 'pmap', 'sequential'}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")


def get_optimal_chunk_size(
    n_samples: int,
    n_features: int,
    memory_limit_gb: float = 8.0,
    dtype: jnp.dtype = jnp.float64
) -> int:
    """
    Compute optimal chunk size based on memory constraints.
    
    This function estimates the maximum number of operations that
    can be batched together without exceeding memory limits.
    
    Parameters
    ----------
    n_samples : int
        Number of samples in each test.
    n_features : int  
        Number of features/variables.
    memory_limit_gb : float
        Available GPU/CPU memory in GB.
    dtype : jnp.dtype
        Data type (affects memory per element).
        
    Returns
    -------
    int
        Recommended chunk size.
        
    Notes
    -----
    This is a heuristic. Actual memory usage depends on JAX's
    compilation and intermediate computations.
    """
    # Bytes per element
    bytes_per_elem = jnp.dtype(dtype).itemsize
    
    # Memory available (use 70% to leave headroom)
    available_bytes = memory_limit_gb * 1e9 * 0.7
    
    # Rough estimate: each test needs ~3x the input data for intermediates
    bytes_per_test = n_samples * n_features * bytes_per_elem * 3
    
    # Compute chunk size
    chunk_size = max(1, int(available_bytes / bytes_per_test))
    
    # Cap at reasonable limits
    return min(chunk_size, 50000)


def chunked_vmap(
    fn: Callable,
    in_axes: Union[int, Tuple[Optional[int], ...]],
    chunk_size: int = 10000,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Callable:
    """
    Create a chunked version of vmap for memory-efficient batching.
    
    When the batch size is larger than chunk_size, the computation
    is split into chunks and processed sequentially, reducing
    peak memory usage.
    
    Parameters
    ----------
    fn : callable
        The function to vectorize.
    in_axes : int or tuple
        Axis specification for vmap.
    chunk_size : int
        Maximum batch size per chunk.
    progress_callback : callable, optional
        Called with (n_completed, n_total) after each chunk.
        
    Returns
    -------
    callable
        A function that processes batches in chunks.
        
    Examples
    --------
    >>> @jit
    ... def expensive_fn(x):
    ...     return jnp.sum(x ** 2)
    >>> 
    >>> batched_fn = chunked_vmap(expensive_fn, in_axes=0, chunk_size=1000)
    >>> results = batched_fn(large_batch)  # Processes in chunks of 1000
    """
    vmapped = jit(vmap(fn, in_axes=in_axes))

    def _get_batch_size(args) -> int:
        if isinstance(in_axes, int):
            return args[0].shape[in_axes]
        for i, ax in enumerate(in_axes):
            if ax is not None:
                return args[i].shape[ax]
        raise ValueError("Could not determine batch axis from in_axes")

    def _pad_arg(arg: jnp.ndarray, ax: Optional[int], pad_size: int) -> jnp.ndarray:
        if ax is None or pad_size == 0:
            return arg
        pad_width = [(0, 0)] * arg.ndim
        pad_width[ax] = (0, pad_size)
        return jnp.pad(arg, pad_width)

    def _slice_arg(arg: jnp.ndarray, ax: Optional[int], start: int) -> jnp.ndarray:
        if ax is None:
            return arg
        start_indices = [0] * arg.ndim
        start_indices[ax] = start
        slice_sizes = list(arg.shape)
        slice_sizes[ax] = chunk_size
        return jax.lax.dynamic_slice(arg, start_indices, slice_sizes)

    def chunked_fn(*args):
        batch_size = _get_batch_size(args)
        if batch_size <= chunk_size:
            return vmapped(*args)

        n_chunks = (batch_size + chunk_size - 1) // chunk_size
        padded_size = n_chunks * chunk_size
        pad_size = padded_size - batch_size

        padded_args = []
        for j, arg in enumerate(args):
            ax = in_axes if isinstance(in_axes, int) else in_axes[j]
            padded_args.append(_pad_arg(arg, ax, pad_size))

        if progress_callback is not None:
            # Python loop to allow progress callbacks
            outputs = None
            for i in range(n_chunks):
                start = i * chunk_size
                chunk_args = []
                for j, arg in enumerate(padded_args):
                    ax = in_axes if isinstance(in_axes, int) else in_axes[j]
                    chunk_args.append(_slice_arg(arg, ax, start))

                chunk_result = vmapped(*tuple(chunk_args))
                if outputs is None:
                    outputs = jax.tree_util.tree_map(
                        lambda x: jnp.zeros((padded_size,) + x.shape[1:], x.dtype),
                        chunk_result,
                    )
                outputs = jax.tree_util.tree_map(
                    lambda out, res: jax.lax.dynamic_update_slice(
                        out, res, (start,) + (0,) * (res.ndim - 1)
                    ),
                    outputs,
                    chunk_result,
                )
                progress_callback(min(start + chunk_size, batch_size), batch_size)

            return jax.tree_util.tree_map(lambda x: x[:batch_size], outputs)

        # JAX-friendly scan for memory-efficient chunking
        def scan_body(carry, idx):
            start = idx * chunk_size
            chunk_args = []
            for j, arg in enumerate(padded_args):
                ax = in_axes if isinstance(in_axes, int) else in_axes[j]
                chunk_args.append(_slice_arg(arg, ax, start))
            chunk_result = vmapped(*tuple(chunk_args))
            updated = jax.tree_util.tree_map(
                lambda out, res: jax.lax.dynamic_update_slice(
                    out, res, (start,) + (0,) * (res.ndim - 1)
                ),
                carry,
                chunk_result,
            )
            return updated, None

        # Allocate output using the first chunk's shape
        first_chunk_args = []
        for j, arg in enumerate(padded_args):
            ax = in_axes if isinstance(in_axes, int) else in_axes[j]
            first_chunk_args.append(_slice_arg(arg, ax, 0))
        first_result = vmapped(*tuple(first_chunk_args))
        output_init = jax.tree_util.tree_map(
            lambda x: jnp.zeros((padded_size,) + x.shape[1:], x.dtype),
            first_result,
        )
        output_init = jax.tree_util.tree_map(
            lambda out, res: jax.lax.dynamic_update_slice(
                out, res, (0,) + (0,) * (res.ndim - 1)
            ),
            output_init,
            first_result,
        )

        output_final, _ = jax.lax.scan(scan_body, output_init, jnp.arange(1, n_chunks))
        return jax.tree_util.tree_map(lambda x: x[:batch_size], output_final)

    return chunked_fn


def parallel_map(
    fn: Callable,
    data: jnp.ndarray,
    config: Optional[ParallelConfig] = None,
    **kwargs
) -> jnp.ndarray:
    """
    Apply a function across batched data with automatic parallelization.
    
    This is the main entry point for parallel computation in JAX-PCMCI.
    It automatically selects the best parallelization strategy based on
    hardware and workload.
    
    Parameters
    ----------
    fn : callable
        Function to apply to each element. Should take a single array.
    data : jnp.ndarray
        Batched input data with shape (batch_size, ...).
    config : ParallelConfig, optional
        Parallelization configuration.
    **kwargs : dict
        Additional arguments passed to fn.
        
    Returns
    -------
    jnp.ndarray
        Results with shape (batch_size, ...).
        
    Examples
    --------
    >>> def my_test(x):
    ...     return jnp.mean(x ** 2)
    >>> 
    >>> results = parallel_map(my_test, batch_data)
    """
    if config is None:
        config = ParallelConfig()

    if config.n_devices is not None:
        if config.n_devices < 1:
            raise ValueError(f"n_devices must be >= 1, got {config.n_devices}")
        available = jax.device_count()
        if config.n_devices > available:
            raise ValueError(
                f"n_devices ({config.n_devices}) exceeds available devices ({available})"
            )
    
    batch_size = data.shape[0]
    n_devices = jax.device_count()
    
    # Determine mode
    mode = config.mode
    if mode == 'auto':
        if n_devices > 1 and batch_size >= n_devices * 10:
            mode = 'pmap'
        elif batch_size > 1:
            mode = 'vmap'
        else:
            mode = 'sequential'
    
    # Apply selected strategy
    if mode == 'sequential':
        results = []
        for i in range(batch_size):
            result = fn(data[i], **kwargs)
            results.append(result)
            if config.progress_callback:
                config.progress_callback(i + 1, batch_size)
        return jnp.stack(results)
    
    elif mode == 'vmap':
        fn_with_kwargs = partial(fn, **kwargs) if kwargs else fn
        batched = chunked_vmap(
            fn_with_kwargs,
            in_axes=0,
            chunk_size=config.chunk_size,
            progress_callback=config.progress_callback
        )
        return batched(data)
    
    elif mode == 'pmap':
        # Reshape for pmap: (n_devices, batch_per_device, ...)
        n_dev = config.n_devices or n_devices
        
        # Pad if necessary
        batch_per_device = (batch_size + n_dev - 1) // n_dev
        padded_size = batch_per_device * n_dev
        
        if padded_size > batch_size:
            pad_size = padded_size - batch_size
            padding = jnp.zeros((pad_size,) + data.shape[1:], dtype=data.dtype)
            data = jnp.concatenate([data, padding], axis=0)
        
        # Reshape
        data = data.reshape(n_dev, batch_per_device, *data.shape[1:])
        
        # Create pmapped function
        fn_with_kwargs = partial(fn, **kwargs) if kwargs else fn
        pmapped = pmap(vmap(fn_with_kwargs))
        
        results = pmapped(data)
        
        # Reshape back
        results = results.reshape(-1, *results.shape[2:])
        
        # Remove padding
        return results[:batch_size]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def batch_independence_tests(
    test,
    X_batch: jnp.ndarray,
    Y_batch: jnp.ndarray,
    Z_batch: Optional[jnp.ndarray] = None,
    config: Optional[ParallelConfig] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run many independence tests in parallel.
    
    This function efficiently batches multiple conditional independence
    tests, leveraging JAX's vectorization for massive speedups.
    
    Parameters
    ----------
    test : CondIndTest
        The independence test to use.
    X_batch : jnp.ndarray
        Batch of X arrays, shape (n_tests, n_samples).
    Y_batch : jnp.ndarray
        Batch of Y arrays, shape (n_tests, n_samples).
    Z_batch : jnp.ndarray, optional
        Batch of conditioning sets, shape (n_tests, n_samples, n_cond).
    config : ParallelConfig, optional
        Parallelization configuration.
        
    Returns
    -------
    statistics : jnp.ndarray
        Test statistics, shape (n_tests,).
    pvalues : jnp.ndarray
        P-values, shape (n_tests,).
        
    Examples
    --------
    >>> from jax_pcmci import ParCorr
    >>> from jax_pcmci.parallel import batch_independence_tests
    >>> 
    >>> test = ParCorr()
    >>> stats, pvals = batch_independence_tests(
    ...     test,
    ...     X_batch,  # (1000, 500) - 1000 tests, 500 samples each
    ...     Y_batch,
    ... )
    """
    return test.run_batch(X_batch, Y_batch, Z_batch)


def distribute_over_targets(
    fn: Callable[[int], jnp.ndarray],
    n_targets: int,
    config: Optional[ParallelConfig] = None
) -> jnp.ndarray:
    """
    Distribute computation over target variables.
    
    This is useful for parallelizing the MCI phase where each
    target variable can be processed independently.
    
    Parameters
    ----------
    fn : callable
        Function that takes target index and returns results.
    n_targets : int
        Number of target variables.
    config : ParallelConfig, optional
        Parallelization configuration.
        
    Returns
    -------
    jnp.ndarray
        Stacked results from all targets.
    """
    # Create index array
    target_indices = jnp.arange(n_targets)
    
    # Use parallel_map (vmap internally)
    return parallel_map(fn, target_indices, config)


def tree_parallel_map(
    fn: Callable,
    tree: dict,
    config: Optional[ParallelConfig] = None
) -> dict:
    """
    Apply a function in parallel over a pytree structure.
    
    This enables parallel processing of hierarchical data structures
    like nested dictionaries of arrays.
    
    Parameters
    ----------
    fn : callable
        Function to apply to each leaf.
    tree : dict
        Pytree structure (nested dict/list with arrays as leaves).
    config : ParallelConfig, optional
        Parallelization configuration.
        
    Returns
    -------
    dict
        Pytree with same structure, fn applied to each leaf.
    """
    return jax.tree_util.tree_map(
        lambda x: parallel_map(fn, x, config) if hasattr(x, 'shape') else x,
        tree
    )


# Convenience decorators

def vectorize_test(fn: Callable) -> Callable:
    """
    Decorator to create a vectorized version of an independence test.
    
    The decorated function can accept batched inputs and will
    automatically use vmap for parallel execution.
    
    Examples
    --------
    >>> @vectorize_test
    ... def my_test_statistic(X, Y, Z=None):
    ...     # Compute test statistic
    ...     return statistic
    >>> 
    >>> # Now works with batched inputs
    >>> stats = my_test_statistic(X_batch, Y_batch)
    """
    return jit(vmap(fn, in_axes=(0, 0, 0)))


def parallelize_mci(fn: Callable) -> Callable:
    """
    Decorator for parallelizing MCI computations across targets.
    
    Wraps a function that processes one target variable to
    process all targets in parallel.
    """
    @jit
    def parallel_fn(data, parents, **kwargs):
        n_targets = len(parents)
        
        def process_target(j):
            return fn(data, j, parents[j], **kwargs)
        
        return vmap(process_target)(jnp.arange(n_targets))
    
    return parallel_fn


# Performance profiling utilities

@dataclass
class BenchmarkResult:
    """
    Results from performance benchmarking.
    
    Attributes
    ----------
    name : str
        Name of the benchmark.
    n_tests : int
        Number of tests run.
    total_time_s : float
        Total execution time in seconds.
    tests_per_second : float
        Throughput in tests per second.
    mode : str
        Parallelization mode used.
    n_devices : int
        Number of devices used.
    """
    name: str
    n_tests: int
    total_time_s: float
    tests_per_second: float
    mode: str
    n_devices: int
    
    def __repr__(self) -> str:
        return (
            f"BenchmarkResult('{self.name}': {self.n_tests} tests in "
            f"{self.total_time_s:.2f}s = {self.tests_per_second:.0f} tests/s, "
            f"mode={self.mode}, devices={self.n_devices})"
        )


def benchmark_parallel_modes(
    test,
    n_samples: int = 500,
    n_tests: int = 1000,
    seed: int = 42
) -> Dict[str, BenchmarkResult]:
    """
    Benchmark different parallelization modes.
    
    This function helps you choose the best parallelization
    strategy for your hardware and workload.
    
    Parameters
    ----------
    test : CondIndTest
        Independence test to benchmark.
    n_samples : int
        Samples per test.
    n_tests : int
        Number of tests to run.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    dict
        Dictionary mapping mode names to BenchmarkResult objects.
        
    Examples
    --------
    >>> from jax_pcmci import ParCorr
    >>> from jax_pcmci.parallel import benchmark_parallel_modes
    >>> 
    >>> results = benchmark_parallel_modes(ParCorr())
    >>> for mode, result in results.items():
    ...     print(result)
    """
    import time
    
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 2)
    
    X = jax.random.normal(keys[0], (n_tests, n_samples))
    Y = jax.random.normal(keys[1], (n_tests, n_samples))
    
    # Compile first (warm-up)
    _ = test.run_batch(X[:10], Y[:10])
    
    results = {}
    
    for mode in ['sequential', 'vmap']:
        config = ParallelConfig(mode=mode)
        
        # Time the execution
        start = time.perf_counter()
        stats, pvals = batch_independence_tests(test, X, Y, config=config)
        # Force evaluation
        _ = float(stats[0])
        end = time.perf_counter()
        
        total_time = end - start
        throughput = n_tests / total_time
        
        results[mode] = BenchmarkResult(
            name=f"{test.name}_{mode}",
            n_tests=n_tests,
            total_time_s=total_time,
            tests_per_second=throughput,
            mode=mode,
            n_devices=jax.device_count() if mode == 'pmap' else 1
        )
    
    return results


__all__ = [
    'ParallelConfig',
    'get_optimal_chunk_size',
    'chunked_vmap',
    'parallel_map',
    'batch_independence_tests',
    'distribute_over_targets',
    'tree_parallel_map',
    'vectorize_test',
    'parallelize_mci',
    'BenchmarkResult',
    'benchmark_parallel_modes',
]
