"""
Pre-compilation Module for JAX-PCMCI
====================================

This module pre-compiles PCMCI kernels for common configurations to reduce
first-run compilation overhead. Compiled kernels are cached persistently.

Usage
-----
# Option 1: Auto-precompile on import (if enabled)
from jax_pcmci import precompile
precompile.warmup_common_configs()

# Option 2: Explicitly precompile before use
from jax_pcmci.precompile import precompile_pcmci
precompile_pcmci(N=5, T=250, tau_max=2)

# Option 3: Precompile multiple configurations
from jax_pcmci.precompile import warmup_common_configs
warmup_common_configs(verbose=True)
"""

import os
import time
from typing import Optional, List, Tuple

import jax
import jax.numpy as jnp
from jax_pcmci import PCMCI, ParCorr, DataHandler
from jax_pcmci.config import get_config


# Common configurations to pre-compile
DEFAULT_CONFIGS = [
    (5, 250, 1),   # Small: N=5, T=250, tau_max=1
    (5, 250, 2),   # Small: N=5, T=250, tau_max=2
    (10, 500, 1),  # Medium: N=10, T=500, tau_max=1
    (10, 500, 2),  # Medium: N=10, T=500, tau_max=2
]


def precompile_pcmci(
    N: int,
    T: int,
    tau_max: int = 1,
    pc_alpha: float = 0.05,
    verbose: bool = False
) -> float:
    """
    Pre-compile PCMCI kernels for a specific configuration.
    
    Parameters
    ----------
    N : int
        Number of variables
    T : int
        Number of time steps
    tau_max : int
        Maximum time lag
    pc_alpha : float
        Significance level for PC algorithm
    verbose : bool
        Print compilation progress
        
    Returns
    -------
    float
        Compilation time in seconds
    """
    if verbose:
        print(f"Pre-compiling PCMCI(N={N}, T={T}, tau_max={tau_max})...", end=" ", flush=True)
    
    # Generate synthetic data
    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, (T, N))
    
    # Create PCMCI instance
    handler = DataHandler(data)
    pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=0)
    
    # Trigger compilation
    t0 = time.time()
    try:
        _ = pcmci.run(tau_max=tau_max, pc_alpha=pc_alpha)
        compile_time = time.time() - t0
        
        if verbose:
            print(f"Done ({compile_time:.2f}s)")
        
        return compile_time
    except Exception as e:
        if verbose:
            print(f"Failed: {e}")
        return -1.0


def warmup_common_configs(
    configs: Optional[List[Tuple[int, int, int]]] = None,
    verbose: bool = True
) -> dict:
    """
    Pre-compile PCMCI for multiple common configurations.
    
    Parameters
    ----------
    configs : List[Tuple[int, int, int]], optional
        List of (N, T, tau_max) configurations to pre-compile.
        If None, uses DEFAULT_CONFIGS.
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Dictionary mapping config to compilation time
    """
    if configs is None:
        configs = DEFAULT_CONFIGS
    
    if verbose:
        print(f"Pre-compiling {len(configs)} PCMCI configurations...")
        print("=" * 60)
    
    results = {}
    total_time = 0.0
    
    for N, T, tau_max in configs:
        compile_time = precompile_pcmci(N, T, tau_max, verbose=verbose)
        results[(N, T, tau_max)] = compile_time
        if compile_time > 0:
            total_time += compile_time
    
    if verbose:
        print("=" * 60)
        print(f"Total pre-compilation time: {total_time:.2f}s")
        print(f"Compiled kernels cached at: {os.environ.get('JAX_COMPILATION_CACHE_DIR', 'default cache')}")
    
    return results


def clear_cache(verbose: bool = True):
    """
    Clear the JAX compilation cache.
    
    Parameters
    ----------
    verbose : bool
        Print status messages
    """
    cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR')
    if cache_dir and os.path.exists(cache_dir):
        import shutil
        try:
            shutil.rmtree(cache_dir)
            if verbose:
                print(f"Cleared compilation cache: {cache_dir}")
        except Exception as e:
            if verbose:
                print(f"Failed to clear cache: {e}")
    else:
        if verbose:
            print("No compilation cache found")


# Auto-warmup on import if PCMCI_PRECOMPILE env var is set
if os.environ.get('PCMCI_PRECOMPILE', '').lower() in ('1', 'true', 'yes'):
    config = get_config()
    if config.verbosity >= 1:
        print("JAX-PCMCI: Auto-precompiling common configurations...")
    warmup_common_configs(verbose=(config.verbosity >= 2))
