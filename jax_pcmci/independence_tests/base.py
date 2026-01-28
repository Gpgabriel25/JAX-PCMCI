"""
Base Class for Conditional Independence Tests
==============================================

This module defines the abstract base class for all conditional independence
tests in JAX-PCMCI. Custom tests should inherit from CondIndTest and implement
the required methods.

Example
-------
>>> from jax_pcmci.independence_tests import CondIndTest
>>>
>>> class MyCustomTest(CondIndTest):
...     def compute_statistic(self, X, Y, Z=None):
...         # Custom implementation
...         pass
...
...     def compute_pvalue(self, statistic, n_samples, n_cond):
...         # Custom implementation
...         pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from functools import partial

from jax_pcmci.config import get_config


@dataclass
class TestResult:
    """
    Result of a conditional independence test.

    This dataclass holds the complete results of a single independence test,
    including the test statistic, p-value, and optional additional information.

    Parameters
    ----------
    statistic : float
        The test statistic value (interpretation depends on the test).
    pvalue : float
        The p-value for the null hypothesis of conditional independence.
    significant : bool
        Whether the test is significant at the specified alpha level.
    alpha : float
        The significance level used for the test.
    n_samples : int
        Number of samples used in the test.
    n_conditions : int
        Number of conditioning variables.
    test_name : str
        Name of the test used.
    extra_info : dict
        Additional test-specific information.

    Examples
    --------
    >>> result = TestResult(
    ...     statistic=0.35,
    ...     pvalue=0.001,
    ...     significant=True,
    ...     alpha=0.05,
    ...     n_samples=1000,
    ...     n_conditions=3,
    ...     test_name='ParCorr'
    ... )
    >>> print(f"Significant: {result.significant} (p={result.pvalue:.4f})")
    """

    statistic: float
    pvalue: float
    significant: bool
    alpha: float
    n_samples: int
    n_conditions: int
    test_name: str
    extra_info: dict = field(default_factory=dict)

    # Prevent pytest from treating this as a test class
    __test__ = False

    def __repr__(self) -> str:
        sig_str = "***" if self.significant else ""
        return (
            f"TestResult({self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.pvalue:.4f}{sig_str}, n={self.n_samples})"
        )


class CondIndTest(ABC):
    """
    Abstract base class for conditional independence tests.

    This class defines the interface that all conditional independence tests
    must implement. It provides common functionality for running tests,
    handling parallelization, and computing p-values.

    Subclasses must implement:
    - `compute_statistic`: Calculate the test statistic
    - `compute_pvalue`: Calculate the p-value from the statistic

    Parameters
    ----------
    significance : str, default='analytic'
        Method for computing significance:
        - 'analytic': Use analytical formula (if available)
        - 'permutation': Use permutation testing
        - 'bootstrap': Use bootstrap resampling
    n_permutations : int, default=500
        Number of permutations for permutation testing.
    alpha : float, default=0.05
        Significance level for hypothesis testing.
    random_seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    name : str
        Human-readable name of the test.
    measure : str
        Type of dependence measure (e.g., 'correlation', 'mutual_information').

    Examples
    --------
    >>> from jax_pcmci.independence_tests import ParCorr
    >>>
    >>> # Create test with permutation-based significance
    >>> test = ParCorr(significance='permutation', n_permutations=1000)
    >>>
    >>> # Run single test
    >>> result = test.run(X, Y, Z, alpha=0.05)
    >>>
    >>> # Run vectorized tests (faster for multiple pairs)
    >>> results = test.run_batch(X_batch, Y_batch, Z_batch)

    Notes
    -----
    All tests are designed to be JIT-compilable. The main computation is
    performed by `compute_statistic` and `compute_pvalue`, which should be
    pure functions suitable for JAX transformations.
    """

    # Class attributes (override in subclasses)
    name: str = "CondIndTest"
    measure: str = "dependence"

    def __init__(
        self,
        significance: str = "analytic",
        n_permutations: int = 500,
        alpha: float = 0.05,
        random_seed: Optional[int] = None,
    ):
        self.significance = significance
        self.n_permutations = n_permutations
        self.alpha = alpha
        self._random_seed = random_seed

        # Validate parameters
        if significance not in ("analytic", "permutation", "bootstrap"):
            raise ValueError(
                f"significance must be 'analytic', 'permutation', or 'bootstrap', "
                f"got '{significance}'"
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be positive, got {n_permutations}")

        # Set up random key (fold-in counter ensures unique draws per call)
        seed = random_seed if random_seed is not None else 0
        self._base_key = jax.random.PRNGKey(seed)
        self._rng_counter = 0

        # Note: We don't cache JIT'd batch runners at instance level because
        # capturing `self` in a JIT'd lambda causes recompilation per instance.
        # Instead, subclasses should override run_batch with proper JIT strategy.

    @abstractmethod
    def compute_statistic(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Compute the test statistic for conditional independence.

        This method must be implemented by subclasses. It should be a pure
        function suitable for JIT compilation.

        Parameters
        ----------
        X : jax.Array
            First variable, shape (n_samples,).
        Y : jax.Array
            Second variable, shape (n_samples,).
        Z : jax.Array or None
            Conditioning variables, shape (n_samples, n_conditions).
            None if testing unconditional independence.

        Returns
        -------
        jax.Array
            Scalar test statistic value.

        Notes
        -----
        The interpretation of the statistic depends on the specific test:
        - ParCorr: Partial correlation coefficient in [-1, 1]
        - CMI: Conditional mutual information in [0, inf)
        """
        pass

    @abstractmethod
    def compute_pvalue(
        self, statistic: jax.Array, n_samples: int, n_conditions: int
    ) -> jax.Array:
        """
        Compute the p-value for the test statistic.

        This method must be implemented by subclasses for tests that support
        analytical p-value computation.

        Parameters
        ----------
        statistic : jax.Array
            The test statistic value.
        n_samples : int
            Number of samples used in the test.
        n_conditions : int
            Number of conditioning variables.

        Returns
        -------
        jax.Array
            P-value for the null hypothesis of conditional independence.
        """
        pass

    def run(
        self,
        X: jax.Array,
        Y: jax.Array,
        Z: Optional[jax.Array] = None,
        alpha: Optional[float] = None,
    ) -> TestResult:
        """
        Run a conditional independence test.

        Tests whether X is conditionally independent of Y given Z.
        H0: X ⊥ Y | Z (conditional independence)
        H1: X ⊥̸ Y | Z (conditional dependence)

        Parameters
        ----------
        X : jax.Array
            First variable, shape (n_samples,).
        Y : jax.Array
            Second variable, shape (n_samples,).
        Z : jax.Array or None, default=None
            Conditioning variables, shape (n_samples, n_conditions).
            If None, tests unconditional independence.
        alpha : float or None, default=None
            Significance level. Uses instance default if None.

        Returns
        -------
        TestResult
            Complete test result including statistic, p-value, and significance.

        Examples
        --------
        >>> test = ParCorr()
        >>> X = jnp.array([1., 2., 3., 4., 5.])
        >>> Y = jnp.array([2., 4., 5., 8., 10.])
        >>> Z = jnp.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        >>> result = test.run(X, Y, Z)
        >>> print(f"p-value: {result.pvalue:.4f}")

        Raises
        ------
        ValueError
            If input shapes are inconsistent.
        """
        alpha = alpha if alpha is not None else self.alpha

        # Validate inputs
        X = jnp.atleast_1d(X)
        Y = jnp.atleast_1d(Y)

        if X.ndim != 1 or Y.ndim != 1:
            raise ValueError(f"X and Y must be 1D, got shapes {X.shape} and {Y.shape}")
        if len(X) != len(Y):
            raise ValueError(
                f"X and Y must have same length, got {len(X)} and {len(Y)}"
            )

        n_samples = len(X)
        n_conditions = 0

        if Z is not None:
            Z = jnp.atleast_2d(Z)
            if Z.shape[0] != n_samples:
                raise ValueError(
                    f"Z must have {n_samples} samples, got {Z.shape[0]}"
                )
            n_conditions = Z.shape[1]

        # Compute test statistic
        statistic = self.compute_statistic(X, Y, Z)

        # Compute p-value based on significance method
        if self.significance == "analytic":
            pvalue = self.compute_pvalue(statistic, n_samples, n_conditions)
        elif self.significance == "permutation":
            pvalue = self._permutation_pvalue(X, Y, Z, statistic)
        else:  # bootstrap
            pvalue = self._bootstrap_pvalue(X, Y, Z, statistic)

        # Convert to Python floats
        stat_val = float(statistic)
        pval = float(pvalue)
        significant = pval < alpha

        return TestResult(
            statistic=stat_val,
            pvalue=pval,
            significant=significant,
            alpha=alpha,
            n_samples=n_samples,
            n_conditions=n_conditions,
            test_name=self.name,
        )

    def run_batch(
        self,
        X_batch: jax.Array,
        Y_batch: jax.Array,
        Z_batch: Optional[jax.Array] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Run multiple independence tests in parallel using vmap.

        This is significantly faster than running tests sequentially,
        especially on GPU/TPU.

        Parameters
        ----------
        X_batch : jax.Array
            Batch of first variables, shape (n_tests, n_samples).
        Y_batch : jax.Array
            Batch of second variables, shape (n_tests, n_samples).
        Z_batch : jax.Array or None
            Batch of conditioning variables, shape (n_tests, n_samples, n_cond).
        alpha : float or None
        """
        # Call implementations directly - subclasses like ParCorr override
        # run_batch for better JIT strategy
        if Z_batch is None:
            return self._run_batch_no_z_impl(X_batch, Y_batch)
        return self._run_batch_with_z_impl(X_batch, Y_batch, Z_batch)

    # ----- JITed batch helpers -------------------------------------------------

    def _run_batch_no_z_impl(
        self, X_batch: jax.Array, Y_batch: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Fast path for batches without conditioning variables."""
        n_samples = X_batch.shape[1]

        statistics = jax.vmap(lambda x, y: self.compute_statistic(x, y, None))(
            X_batch, Y_batch
        )

        if self.significance == "analytic":
            pvalues = jax.vmap(lambda s: self.compute_pvalue(s, n_samples, 0))(statistics)
        else:
            pvalues = self._batch_permutation_pvalues(
                X_batch, Y_batch, None, statistics
            )

        return statistics, pvalues

    def _run_batch_with_z_impl(
        self, X_batch: jax.Array, Y_batch: jax.Array, Z_batch: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Fast path for batches with conditioning variables."""
        n_samples = X_batch.shape[1]
        n_conditions = Z_batch.shape[2] if Z_batch.ndim == 3 else 1

        statistics = jax.vmap(self.compute_statistic)(X_batch, Y_batch, Z_batch)

        if self.significance == "analytic":
            pvalues = jax.vmap(
                lambda s: self.compute_pvalue(s, n_samples, n_conditions)
            )(statistics)
        else:
            pvalues = self._batch_permutation_pvalues(
                X_batch, Y_batch, Z_batch, statistics
            )

        return statistics, pvalues

    # ----- Shared helpers ----------------------------------------------------

    def _next_key(self) -> jax.Array:
        """Return a new RNG key derived from the base seed."""
        self._rng_counter += 1
        return jax.random.fold_in(self._base_key, self._rng_counter)

    def _prepare_inputs(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array]
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """Optional preprocessing for permutation/bootstrap (override in subclasses)."""
        return X, Y, Z

    def _statistic_from_prepared(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array]
    ) -> jax.Array:
        """Compute statistic from already-prepared inputs (override in subclasses)."""
        return self.compute_statistic(X, Y, Z)

    def _permutation_pvalue(
        self,
        X: jax.Array,
        Y: jax.Array,
        Z: Optional[jax.Array],
        observed_stat: jax.Array,
    ) -> jax.Array:
        """Compute p-value using permutation testing (fully vectorized)."""
        key = self._next_key()
        if Z is None:
            return self._permutation_pvalue_no_z(X, Y, observed_stat, key)
        return self._permutation_pvalue_with_z(X, Y, Z, observed_stat, key)

    @partial(jax.jit, static_argnums=(0,))
    def _permutation_pvalue_no_z(
        self, X: jax.Array, Y: jax.Array, observed_stat: jax.Array, key: jax.Array
    ) -> jax.Array:
        X_prep, Y_prep, _ = self._prepare_inputs(X, Y, None)
        n_samples = X_prep.shape[0]
        perm_keys = jax.random.split(key, self.n_permutations)

        def perm_stat(k):
            perm = jax.random.permutation(k, n_samples)
            return self._statistic_from_prepared(X_prep[perm], Y_prep, None)

        null_stats = jax.vmap(perm_stat)(perm_keys)
        return jnp.mean(jnp.abs(null_stats) >= jnp.abs(observed_stat))

    @partial(jax.jit, static_argnums=(0,))
    def _permutation_pvalue_with_z(
        self,
        X: jax.Array,
        Y: jax.Array,
        Z: jax.Array,
        observed_stat: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        X_prep, Y_prep, Z_prep = self._prepare_inputs(X, Y, Z)
        n_samples = X_prep.shape[0]
        perm_keys = jax.random.split(key, self.n_permutations)

        def perm_stat(k):
            perm = jax.random.permutation(k, n_samples)
            return self._statistic_from_prepared(X_prep[perm], Y_prep, Z_prep)

        null_stats = jax.vmap(perm_stat)(perm_keys)
        return jnp.mean(jnp.abs(null_stats) >= jnp.abs(observed_stat))

    def _bootstrap_pvalue(
        self,
        X: jax.Array,
        Y: jax.Array,
        Z: Optional[jax.Array],
        observed_stat: jax.Array,
    ) -> jax.Array:
        """Compute p-value using bootstrap resampling (vectorized)."""
        key = self._next_key()
        if Z is None:
            return self._bootstrap_pvalue_no_z(X, Y, observed_stat, key)
        return self._bootstrap_pvalue_with_z(X, Y, Z, observed_stat, key)

    @partial(jax.jit, static_argnums=(0,))
    def _bootstrap_pvalue_no_z(
        self, X: jax.Array, Y: jax.Array, observed_stat: jax.Array, key: jax.Array
    ) -> jax.Array:
        X_prep, Y_prep, _ = self._prepare_inputs(X, Y, None)
        n_samples = X_prep.shape[0]
        boot_keys = jax.random.split(key, self.n_permutations)

        def single_bootstrap(k):
            idx = jax.random.choice(k, n_samples, shape=(n_samples,), replace=True)
            return self._statistic_from_prepared(X_prep[idx], Y_prep[idx], None)

        boot_stats = jax.vmap(single_bootstrap)(boot_keys)
        return jnp.mean(jnp.abs(boot_stats) >= jnp.abs(observed_stat))

    @partial(jax.jit, static_argnums=(0,))
    def _bootstrap_pvalue_with_z(
        self,
        X: jax.Array,
        Y: jax.Array,
        Z: jax.Array,
        observed_stat: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        X_prep, Y_prep, Z_prep = self._prepare_inputs(X, Y, Z)
        n_samples = X_prep.shape[0]
        boot_keys = jax.random.split(key, self.n_permutations)

        def single_bootstrap(k):
            idx = jax.random.choice(k, n_samples, shape=(n_samples,), replace=True)
            return self._statistic_from_prepared(
                X_prep[idx], Y_prep[idx], Z_prep[idx]
            )

        boot_stats = jax.vmap(single_bootstrap)(boot_keys)
        return jnp.mean(jnp.abs(boot_stats) >= jnp.abs(observed_stat))

    def _batch_permutation_pvalues(
        self,
        X_batch: jax.Array,
        Y_batch: jax.Array,
        Z_batch: Optional[jax.Array],
        observed_stats: jax.Array,
    ) -> jax.Array:
        """Compute permutation p-values for a batch of tests (vectorized)."""
        key = self._next_key()
        if Z_batch is None:
            return self._batch_permutation_pvalues_no_z(
                X_batch, Y_batch, observed_stats, key
            )
        return self._batch_permutation_pvalues_with_z(
            X_batch, Y_batch, Z_batch, observed_stats, key
        )

    @partial(jax.jit, static_argnums=(0,))
    def _batch_permutation_pvalues_no_z(
        self,
        X_batch: jax.Array,
        Y_batch: jax.Array,
        observed_stats: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        n_tests = X_batch.shape[0]
        keys = jax.random.split(key, n_tests)

        def single(x, y, obs, k):
            return self._permutation_pvalue_no_z(x, y, obs, k)

        return jax.vmap(single)(X_batch, Y_batch, observed_stats, keys)

    @partial(jax.jit, static_argnums=(0,))
    def _batch_permutation_pvalues_with_z(
        self,
        X_batch: jax.Array,
        Y_batch: jax.Array,
        Z_batch: jax.Array,
        observed_stats: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        n_tests = X_batch.shape[0]
        keys = jax.random.split(key, n_tests)

        def single(x, y, z, obs, k):
            return self._permutation_pvalue_with_z(x, y, z, obs, k)

        return jax.vmap(single)(X_batch, Y_batch, Z_batch, observed_stats, keys)

    def get_dependence_measure(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array] = None
    ) -> float:
        """
        Get the dependence measure (test statistic) without computing p-value.

        This is useful when you only need the strength of dependence,
        not statistical significance.

        Parameters
        ----------
        X : jax.Array
            First variable.
        Y : jax.Array
            Second variable.
        Z : jax.Array or None
            Conditioning variables.

        Returns
        -------
        float
            The dependence measure value.
        """
        return float(self.compute_statistic(X, Y, Z))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(significance='{self.significance}', "
            f"alpha={self.alpha})"
        )
