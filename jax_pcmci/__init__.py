"""
JAX-PCMCI: High-Performance Causal Discovery with JAX Acceleration
====================================================================

JAX-PCMCI is a library for causal discovery from time series data using
PCMCI algorithms, accelerated with JAX for GPU/TPU/CPU parallelization.

Main Features
-------------
- **PCMCI**: The original PCMCI algorithm with various independence tests
- **PCMCI+**: Enhanced PCMCI with contemporaneous causal discovery
- **ParCorr**: Partial correlation test for linear dependencies
- **Nonlinear Tests**: CMI-based tests using kernel methods

Performance
-----------
- JAX JIT compilation for optimized execution
- Automatic vectorization with vmap
- Multi-device parallelization with pmap
- Support for GPU, TPU, and CPU backends

Quick Start
-----------
>>> import jax.numpy as jnp
>>> from jax_pcmci import PCMCI, PCMCIPlus, ParCorr, DataHandler
>>>
>>> # Create sample data
>>> data = jnp.array(your_time_series_data)
>>> datahandler = DataHandler(data)
>>>
>>> # Run PCMCI with partial correlation
>>> pcmci = PCMCI(datahandler, cond_ind_test=ParCorr())
>>> results = pcmci.run(tau_max=3)
>>>
>>> # Visualize results
>>> results.plot_graph()

For more examples and detailed documentation, visit:
https://jax-pcmci.readthedocs.io
"""

__version__ = "1.3.0"
__author__ = "JAX-PCMCI Contributors"

# Core imports
from jax_pcmci.data import DataHandler, TimeSeriesData
from jax_pcmci.independence_tests import (
    CondIndTest,
    ParCorr,
    CMIKnn,
    CMISymbolic,
    GPDCond,
)
from jax_pcmci.algorithms import PCMCI, PCMCIPlus
from jax_pcmci.results import PCMCIResults
from jax_pcmci.config import PCMCIConfig, set_device, get_device_info
from jax_pcmci.parallel import (
    ParallelConfig,
    parallel_map,
    batch_independence_tests,
    benchmark_parallel_modes,
)
from jax_pcmci import precompile

__all__ = [
    # Version
    "__version__",
    # Data handling
    "DataHandler",
    "TimeSeriesData",
    # Independence tests
    "CondIndTest",
    "ParCorr",
    "CMIKnn",
    "CMISymbolic",
    "GPDCond",
    # Algorithms
    "PCMCI",
    "PCMCIPlus",
    # Results
    "PCMCIResults",
    # Configuration
    "PCMCIConfig",
    "set_device",
    "get_device_info",
    # Parallel utilities
    "ParallelConfig",
    "parallel_map",
    "batch_independence_tests",
    "benchmark_parallel_modes",
    # Precompilation
    "precompile",
]

