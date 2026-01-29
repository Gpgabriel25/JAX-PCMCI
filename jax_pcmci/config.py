"""
Configuration and Device Management for JAX-PCMCI
==================================================

This module provides configuration utilities and device management for
controlling JAX execution across different hardware backends (CPU, GPU, TPU).

Device Management
-----------------
JAX-PCMCI automatically detects available devices and uses the best available
backend. You can override this behavior using the configuration functions.

Example
-------
>>> from jax_pcmci.config import set_device, get_device_info, PCMCIConfig
>>>
>>> # Check available devices
>>> info = get_device_info()
>>> print(f"Available GPUs: {info['gpu_count']}")
>>>
>>> # Force CPU execution
>>> set_device('cpu')
>>>
>>> # Configure global settings
>>> config = PCMCIConfig(
...     precision='float64',
...     parallelization='auto',
...     random_seed=42
... )
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Union

import jax
import jax.numpy as jnp
from jax import config as jax_config


class Precision(Enum):
    """
    Numerical precision settings for JAX-PCMCI computations.

    Attributes
    ----------
    FLOAT32 : str
        Single precision (32-bit). Faster but less accurate.
        Recommended for large-scale exploratory analysis.
    FLOAT64 : str
        Double precision (64-bit). More accurate but slower.
        Recommended for final analysis and statistical testing.
    """

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class ParallelizationMode(Enum):
    """
    Parallelization strategies for PCMCI computations.

    Attributes
    ----------
    AUTO : str
        Automatically select the best parallelization strategy based on
        data size and available devices.
    VMAP : str
        Use JAX vmap for vectorization across a single device.
        Best for GPU with limited memory.
    PMAP : str
        Use JAX pmap for parallelization across multiple devices.
        Best for multi-GPU or TPU pod setups.
    SEQUENTIAL : str
        No parallelization. Process tests sequentially.
        Best for debugging or very small datasets.
    """

    AUTO = "auto"
    VMAP = "vmap"
    PMAP = "pmap"
    SEQUENTIAL = "sequential"


@dataclass
class PCMCIConfig:
    """
    Global configuration settings for JAX-PCMCI.

    This class holds all configurable parameters that affect the behavior
    of PCMCI algorithms across the library. Use this to customize precision,
    parallelization, and other global settings.

    Parameters
    ----------
    precision : Precision or str, default='float32'
        Numerical precision for computations. Options:
        - 'float32': Faster, less memory, slightly less accurate
        - 'float64': Slower, more memory, more accurate (recommended)
    parallelization : ParallelizationMode or str, default='auto'
        Strategy for parallelizing independence tests. Options:
        - 'auto': Automatically select based on hardware
        - 'vmap': Vectorize on single device
        - 'pmap': Parallelize across multiple devices
        - 'sequential': No parallelization
    random_seed : int or None, default=None
        Random seed for reproducibility. If None, uses system entropy.
    jit_compile : bool, default=True
        Whether to JIT compile core functions. Disable for debugging.
    enable_x64 : bool, default=False
        Enable 64-bit floating point support in JAX.
    memory_efficient : bool, default=False
        Enable memory-efficient mode for large datasets.
        Trades speed for lower memory usage.
    batch_size : int or None, default=None
        Batch size for processing independence tests.
        If None, processes all tests at once.
    progress_bar : bool, default=True
        Show progress bar during computations.
    verbosity : int, default=1
        Verbosity level (0=silent, 1=normal, 2=verbose, 3=debug).
    cache_results : bool, default=True
        Cache intermediate results to avoid redundant computations.
    cache_max_entries : int, default=4096
        Maximum number of cached variable-pair slices.
    gpu_preallocate : bool, default=True
        Whether JAX preallocates most GPU memory at startup.
    gpu_memory_fraction : float or None, default=None
        Fraction of GPU memory to allocate (e.g., 0.7). None = JAX default.
    gpu_allocator : str or None, default=None
        GPU allocator backend, e.g. 'platform' or 'bfc'. None = JAX default.

    Examples
    --------
    >>> config = PCMCIConfig(
    ...     precision='float64',
    ...     parallelization='vmap',
    ...     random_seed=42,
    ...     verbosity=2
    ... )
    >>> # Apply configuration globally
    >>> config.apply()
    >>>
    >>> # Or use as context manager
    >>> with config:
    ...     results = pcmci.run(tau_max=5)

    Notes
    -----
    Configuration changes affect all subsequent PCMCI operations.
    Use the context manager interface for temporary configuration changes.
    """

    precision: Union[Precision, str] = "float32"
    parallelization: Union[ParallelizationMode, str] = "auto"
    random_seed: Optional[int] = None
    jit_compile: bool = True
    enable_x64: bool = False
    memory_efficient: bool = False
    batch_size: Optional[int] = None
    progress_bar: bool = True
    verbosity: int = 1
    cache_results: bool = True
    cache_max_entries: int = 8192
    gpu_preallocate: bool = True
    gpu_memory_fraction: Optional[float] = None
    gpu_allocator: Optional[str] = None
    compilation_cache_dir: Optional[str] = None
    _previous_config: Optional["PCMCIConfig"] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate and normalize configuration values."""
        # Convert string values to enums
        if isinstance(self.precision, str):
            self.precision = Precision(self.precision.lower())
        if isinstance(self.parallelization, str):
            self.parallelization = ParallelizationMode(self.parallelization.lower())

        # Validate verbosity
        if not 0 <= self.verbosity <= 3:
            raise ValueError(f"verbosity must be 0-3, got {self.verbosity}")

        # Validate batch_size
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.cache_max_entries < 1:
            raise ValueError(
                f"cache_max_entries must be positive, got {self.cache_max_entries}"
            )

        if self.gpu_memory_fraction is not None:
            if not (0.0 < self.gpu_memory_fraction <= 1.0):
                raise ValueError(
                    "gpu_memory_fraction must be in (0, 1], got "
                    f"{self.gpu_memory_fraction}"
                )
        if self.gpu_allocator is not None:
            self.gpu_allocator = self.gpu_allocator.lower()
            if self.gpu_allocator not in ("platform", "bfc"):
                raise ValueError(
                    "gpu_allocator must be 'platform' or 'bfc', got "
                    f"{self.gpu_allocator}"
                )

    def apply(self) -> None:
        """
        Apply this configuration globally.

        This method modifies JAX's global configuration to match the
        settings in this PCMCIConfig instance.

        Examples
        --------
        >>> config = PCMCIConfig(precision='float64', enable_x64=True)
        >>> config.apply()  # Now all computations use float64
        """
        global _GLOBAL_CONFIG
        _GLOBAL_CONFIG = self

        # Apply JAX-specific settings
        # Keep JAX and our dtype setting consistent to avoid silent truncation.
        jax_config.update("jax_enable_x64", bool(self.enable_x64))

        # Configure GPU memory behavior (must be set before first GPU use).
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
            "true" if self.gpu_preallocate else "false"
        )
        if self.gpu_memory_fraction is not None:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
                self.gpu_memory_fraction
            )
        if self.gpu_allocator is not None:
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = self.gpu_allocator

        # Configure compilation cache
        if self.compilation_cache_dir is not None:
            cache_dir = os.path.expanduser(self.compilation_cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
            if self.verbosity >= 2:
                print(f"JAX compilation cache enabled: {cache_dir}")
        elif "JAX_COMPILATION_CACHE_DIR" not in os.environ:
            # Enable by default with a sensible location
            default_cache = os.path.expanduser("~/.cache/jax_pcmci")
            os.makedirs(default_cache, exist_ok=True)
            os.environ["JAX_COMPILATION_CACHE_DIR"] = default_cache
            if self.verbosity >= 1:
                print(f"JAX compilation cache enabled (default): {default_cache}")

        if self.verbosity >= 2:
            print(f"Applied configuration: {self}")

    def __enter__(self) -> "PCMCIConfig":
        """Context manager entry: save current config and apply this one."""
        global _GLOBAL_CONFIG
        self._previous_config = _GLOBAL_CONFIG
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: restore previous configuration."""
        global _GLOBAL_CONFIG
        if self._previous_config is not None:
            self._previous_config.apply()
        else:
            _GLOBAL_CONFIG = PCMCIConfig()  # Reset to defaults

    @property
    def dtype(self) -> jnp.dtype:
        """
        Get the JAX dtype corresponding to the precision setting.

        Returns
        -------
        jnp.dtype
            The JAX dtype (jnp.float32 or jnp.float64).
        """
        # If x64 is disabled in JAX, float64 arrays are not representable.
        # In that case we must use float32 to avoid repeated dtype truncation.
        x64_enabled = bool(jax_config.read("jax_enable_x64"))
        if self.precision == Precision.FLOAT64 and self.enable_x64 and x64_enabled:
            return jnp.float64
        return jnp.float32

    def get_rng_key(self) -> jax.random.PRNGKey:
        """
        Get a JAX random key based on the configured seed.

        Returns
        -------
        jax.random.PRNGKey
            A JAX random key for reproducible random number generation.
        """
        seed = self.random_seed if self.random_seed is not None else 0
        return jax.random.PRNGKey(seed)


# Global configuration instance
_GLOBAL_CONFIG: PCMCIConfig = PCMCIConfig()


def get_config() -> PCMCIConfig:
    """
    Get the current global configuration.

    Returns
    -------
    PCMCIConfig
        The current global configuration instance.

    Examples
    --------
    >>> config = get_config()
    >>> print(f"Current precision: {config.precision}")
    """
    return _GLOBAL_CONFIG


def set_config(config: PCMCIConfig) -> None:
    """
    Set the global configuration.

    Parameters
    ----------
    config : PCMCIConfig
        The configuration to apply globally.

    Examples
    --------
    >>> new_config = PCMCIConfig(precision='float32')
    >>> set_config(new_config)
    """
    config.apply()


def set_device(
    device: Literal["cpu", "gpu", "tpu", "auto"] = "auto",
    device_id: Optional[int] = None,
) -> None:
    """
    Set the default device for JAX-PCMCI computations.

    Parameters
    ----------
    device : {'cpu', 'gpu', 'tpu', 'auto'}, default='auto'
        The device type to use:
        - 'cpu': Force CPU execution
        - 'gpu': Use GPU (CUDA) if available
        - 'tpu': Use TPU if available
        - 'auto': Automatically select best available device
    device_id : int or None, default=None
        Specific device ID to use (for multi-GPU setups).
        If None, uses device 0 or all devices for pmap.

    Raises
    ------
    RuntimeError
        If the requested device is not available.

    Examples
    --------
    >>> # Use GPU
    >>> set_device('gpu')
    >>>
    >>> # Use specific GPU
    >>> set_device('gpu', device_id=1)
    >>>
    >>> # Force CPU (useful for debugging)
    >>> set_device('cpu')

    Notes
    -----
    This function modifies JAX's global device settings. It affects all
    subsequent JAX operations, not just JAX-PCMCI.
    """
    device = device.lower()

    if device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
        if get_config().verbosity >= 1:
            print("JAX-PCMCI: Using CPU backend")

    elif device == "gpu":
        try:
            gpus = jax.devices("gpu")
            if not gpus:
                raise RuntimeError("No GPU devices available")
            jax.config.update("jax_platform_name", "gpu")
            if device_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            if get_config().verbosity >= 1:
                print(f"JAX-PCMCI: Using GPU backend ({len(gpus)} device(s) available)")
        except RuntimeError as e:
            raise RuntimeError(f"GPU not available: {e}")

    elif device == "tpu":
        try:
            tpus = jax.devices("tpu")
            if not tpus:
                raise RuntimeError("No TPU devices available")
            jax.config.update("jax_platform_name", "tpu")
            if get_config().verbosity >= 1:
                print(f"JAX-PCMCI: Using TPU backend ({len(tpus)} device(s) available)")
        except RuntimeError as e:
            raise RuntimeError(f"TPU not available: {e}")

    elif device == "auto":
        # Try TPU first, then GPU, then fall back to CPU
        try:
            tpus = jax.devices("tpu")
            if tpus:
                jax.config.update("jax_platform_name", "tpu")
                if get_config().verbosity >= 1:
                    print(f"JAX-PCMCI: Auto-selected TPU ({len(tpus)} device(s))")
                return
        except:
            pass

        try:
            gpus = jax.devices("gpu")
            if gpus:
                jax.config.update("jax_platform_name", "gpu")
                if get_config().verbosity >= 1:
                    print(f"JAX-PCMCI: Auto-selected GPU ({len(gpus)} device(s))")
                return
        except:
            pass

        jax.config.update("jax_platform_name", "cpu")
        if get_config().verbosity >= 1:
            print("JAX-PCMCI: Auto-selected CPU (no accelerators available)")

    else:
        raise ValueError(f"Unknown device: {device}. Use 'cpu', 'gpu', 'tpu', or 'auto'")


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available JAX devices.

    Returns
    -------
    dict
        Dictionary containing device information:
        - 'default_backend': Current default backend name
        - 'cpu_count': Number of CPU devices
        - 'gpu_count': Number of GPU devices
        - 'tpu_count': Number of TPU devices
        - 'devices': List of all available devices
        - 'default_device': The default device being used

    Examples
    --------
    >>> info = get_device_info()
    >>> print(f"Default backend: {info['default_backend']}")
    >>> print(f"Available GPUs: {info['gpu_count']}")
    >>> for device in info['devices']:
    ...     print(f"  - {device}")
    """
    info = {
        "default_backend": jax.default_backend(),
        "cpu_count": 0,
        "gpu_count": 0,
        "tpu_count": 0,
        "devices": [],
        "default_device": None,
    }

    try:
        cpus = jax.devices("cpu")
        info["cpu_count"] = len(cpus)
        info["devices"].extend(cpus)
    except:
        pass

    try:
        gpus = jax.devices("gpu")
        info["gpu_count"] = len(gpus)
        info["devices"].extend(gpus)
    except:
        pass

    try:
        tpus = jax.devices("tpu")
        info["tpu_count"] = len(tpus)
        info["devices"].extend(tpus)
    except:
        pass

    try:
        info["default_device"] = jax.devices()[0]
    except:
        pass

    return info


def enable_debug_mode() -> None:
    """
    Enable debug mode for development and troubleshooting.

    This disables JIT compilation and enables various JAX debugging features.
    Useful for tracking down NaN issues or understanding computation flow.

    Warning
    -------
    Debug mode significantly slows down execution. Only use for debugging.

    Examples
    --------
    >>> enable_debug_mode()
    >>> # Now all operations are run eagerly without JIT
    >>> pcmci.run(tau_max=3)  # Much slower but easier to debug
    """
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_debug_nans", True)

    config = get_config()
    config.jit_compile = False
    config.verbosity = 3

    warnings.warn(
        "Debug mode enabled. JIT compilation is disabled. "
        "This will significantly slow down execution.",
        UserWarning,
    )


def disable_debug_mode() -> None:
    """
    Disable debug mode and restore normal operation.

    Examples
    --------
    >>> disable_debug_mode()
    >>> # JIT compilation is now enabled again
    """
    jax.config.update("jax_disable_jit", False)
    jax.config.update("jax_debug_nans", False)

    config = get_config()
    config.jit_compile = True
