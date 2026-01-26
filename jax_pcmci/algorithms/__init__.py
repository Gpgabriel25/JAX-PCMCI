"""
PCMCI Algorithms for JAX-PCMCI
==============================

This module provides the core PCMCI and PCMCI+ algorithms for causal
discovery from time series data.

Available Algorithms
--------------------
- PCMCI: Original PCMCI algorithm (lagged causal discovery)
- PCMCIPlus: PCMCI+ for contemporaneous + lagged causal discovery

Both algorithms are optimized for parallel execution on GPU/TPU using JAX.
"""

from jax_pcmci.algorithms.pcmci import PCMCI
from jax_pcmci.algorithms.pcmci_plus import PCMCIPlus

__all__ = ["PCMCI", "PCMCIPlus"]
