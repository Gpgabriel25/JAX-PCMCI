"""
Independence Tests for JAX-PCMCI
================================

This module provides conditional independence tests for PCMCI algorithms,
all implemented in JAX for GPU/TPU acceleration.

Available Tests
---------------
- ParCorr: Partial correlation for linear dependencies
- CMIKnn: Conditional mutual information using k-NN estimation
- GPDCond: Gaussian Process-based conditional independence test

All tests follow a common interface defined by the CondIndTest base class.
"""

from jax_pcmci.independence_tests.base import CondIndTest
from jax_pcmci.independence_tests.parcorr import ParCorr
from jax_pcmci.independence_tests.cmi_knn import CMIKnn, CMISymbolic
from jax_pcmci.independence_tests.gpdc import GPDCond

__all__ = [
    "CondIndTest",
    "ParCorr",
    "CMIKnn",
    "CMISymbolic",
    "GPDCond",
]
