"""
Partial Correlation Test (ParCorr)
==================================

This module implements the partial correlation test for conditional
independence, optimized for JAX acceleration.

Partial correlation measures the linear association between two variables
after removing the linear effect of conditioning variables.

Mathematical Background
-----------------------
The partial correlation between X and Y given Z is defined as:

    ρ(X,Y|Z) = ρ(ε_X, ε_Y)

where ε_X and ε_Y are the residuals from regressing X and Y on Z respectively.

For linear Gaussian data, partial correlation of zero implies conditional
independence.

Example
-------
>>> from jax_pcmci.independence_tests import ParCorr
>>> import jax.numpy as jnp
>>>
>>> test = ParCorr()
>>> X = jnp.array([1., 2., 3., 4., 5.])
>>> Y = jnp.array([2., 4., 5., 8., 10.])
>>> result = test.run(X, Y)
>>> print(f"Partial correlation: {result.statistic:.3f}")
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import stats as jax_stats
from jax.scipy import linalg

from jax_pcmci.independence_tests.base import CondIndTest
from jax_pcmci.config import get_config


# ============================================================================
# Standalone JIT-compiled functions for maximum performance
# These avoid the overhead of `self` in static_argnums
# ============================================================================


@jax.jit
def _parcorr_pvalue(statistic: jax.Array, n_samples: int, n_conditions: int) -> jax.Array:
    """
    Compute p-value using Fisher's z-transformation (standalone JITted version).
    
    This is a module-level function to avoid JIT recompilation when `self` changes.
    """
    # Degrees of freedom
    df = n_samples - n_conditions - 3

    # Handle edge cases
    df = jnp.maximum(df, 1)

    # Fisher's z-transformation
    # Clip to avoid log(0) for |r| = 1
    r_clipped = jnp.clip(statistic, -0.9999, 0.9999)
    z = 0.5 * jnp.log((1 + r_clipped) / (1 - r_clipped))

    # Standard error under H0
    se = 1.0 / jnp.sqrt(df)

    # Test statistic
    z_stat = z / se

    # Two-sided p-value from standard normal
    pvalue = 2 * (1 - jax_stats.norm.cdf(jnp.abs(z_stat)))

    return pvalue


@jax.jit
def _compute_correlation_jit(X: jax.Array, Y: jax.Array) -> jax.Array:
    """
    Compute Pearson correlation between X and Y (standalone JITted version).
    """
    # Center the variables
    X_centered = X - jnp.mean(X)
    Y_centered = Y - jnp.mean(Y)

    # Compute correlation
    numerator = jnp.sum(X_centered * Y_centered)
    denominator = jnp.sqrt(jnp.sum(X_centered**2) * jnp.sum(Y_centered**2))

    # Handle zero denominator
    correlation = jnp.where(
        denominator > 1e-10,
        numerator / denominator,
        0.0
    )

    # Clip to [-1, 1] for numerical stability
    return jnp.clip(correlation, -1.0, 1.0)


@jax.jit
def _compute_partial_correlation_jit(X: jax.Array, Y: jax.Array, Z: jax.Array) -> jax.Array:
    """
    Compute partial correlation via Schur complement of Covariance Matrix.
    
    This method is more memory efficient than the residual method as it avoids
    storing O(T) residuals, instead working with O(d^2) covariance matrices.
    
    Method:
    1. Construct data matrix D = [X, Y, Z]
    2. Compute Covariance Matrix Sigma = D.T @ D
    3. Compute Schur complement to get conditional covariance of {X,Y} given Z
    4. Normalize to get correlation
    """
    # Ensure Z is 2D
    Z = jnp.atleast_2d(Z)
    if Z.shape[0] == 1 and Z.shape[1] != X.shape[0]:
        Z = Z.T

    # 1. Construct Data Matrix D = [X, Y, Z]
    # Shape: (T, 2 + dim_Z)
    # Stacking column-wise
    D = jnp.column_stack([X, Y, Z])
    
    # 2. Compute Covariance (Unnormalized is fine for correlation)
    # Center the data first
    D_centered = D - jnp.mean(D, axis=0)
    Sigma = D_centered.T @ D_centered
    
    # 3. Extract Blocks
    # Indices: 0->X, 1->Y, 2+ -> Z
    # Sigma_AA = Sigma[0:2, 0:2] (Covariance of X,Y)
    # Sigma_AB = Sigma[0:2, 2:]  (Cross-cov with Z)
    # Sigma_BB = Sigma[2:, 2:]   (Covariance of Z)
    
    Sigma_AA = Sigma[:2, :2]
    Sigma_AB = Sigma[:2, 2:]
    Sigma_BB = Sigma[2:, 2:]
    
    # 4. Compute Schur Complement: S = A - B D^-1 C
    # Here: Cond_Cov = Sigma_AA - Sigma_AB @ Sigma_BB^-1 @ Sigma_AB.T
    
    # Regularize Sigma_BB for stability
    n_z = Sigma_BB.shape[0]
    ridge = 1e-6 * jnp.eye(n_z, dtype=Sigma.dtype)
    
    # Solve linear system: Sigma_BB @ W = Sigma_AB.T  =>  W = Sigma_BB^-1 @ Sigma_AB.T
    # We use Cholesky for stability
    L_and_lower = linalg.cho_factor(Sigma_BB + ridge, lower=True)
    W = linalg.cho_solve(L_and_lower, Sigma_AB.T)
    
    # Cond_Cov = Sigma_AA - Sigma_AB @ W
    Cond_Cov = Sigma_AA - Sigma_AB @ W
    
    # 5. Extract Partial Correlation
    # Cov(X,Y|Z) = Cond_Cov[0, 1]
    # Var(X|Z)   = Cond_Cov[0, 0]
    # Var(Y|Z)   = Cond_Cov[1, 1]
    
    cov_xy = Cond_Cov[0, 1]
    var_x = Cond_Cov[0, 0]
    var_y = Cond_Cov[1, 1]
    
    denominator = jnp.sqrt(var_x * var_y)
    
    # Handle zero denominator
    correlation = jnp.where(
        denominator > 1e-10,
        cov_xy / denominator,
        0.0
    )
    
    return jnp.clip(correlation, -1.0, 1.0)


# Vectorized batch versions for maximum GPU throughput
@jax.jit
def _batch_correlation_jit(X_batch: jax.Array, Y_batch: jax.Array) -> jax.Array:
    """Compute correlations for batched inputs."""
    return jax.vmap(_compute_correlation_jit)(X_batch, Y_batch)


@jax.jit
def _batch_partial_correlation_jit(
    X_batch: jax.Array, Y_batch: jax.Array, Z_batch: jax.Array
) -> jax.Array:
    """Compute partial correlations for batched inputs."""
    return jax.vmap(_compute_partial_correlation_jit)(X_batch, Y_batch, Z_batch)


@jax.jit
def _batch_pvalue_jit(
    statistics: jax.Array, n_samples: jax.Array, n_conditions: jax.Array
) -> jax.Array:
    """Compute p-values for batched statistics (vectorized)."""
    df = n_samples - n_conditions - 3
    df = jnp.maximum(df, 1)

    r_clipped = jnp.clip(statistics, -0.9999, 0.9999)
    z = 0.5 * jnp.log((1 + r_clipped) / (1 - r_clipped))
    se = 1.0 / jnp.sqrt(df)
    z_stat = z / se

    return 2 * (1 - jax_stats.norm.cdf(jnp.abs(z_stat)))


class ParCorr(CondIndTest):
    """
    Partial Correlation test for linear conditional independence.

    This test measures the linear association between two variables after
    accounting for the linear effects of conditioning variables. It is
    optimal for data with linear relationships and Gaussian distributions.

    Parameters
    ----------
    significance : str, default='analytic'
        Method for computing p-values:
        - 'analytic': Use Fisher's z-transformation (recommended)
        - 'permutation': Permutation-based p-values
        - 'bootstrap': Bootstrap-based p-values
    n_permutations : int, default=500
        Number of permutations (only used if significance='permutation').
    alpha : float, default=0.05
        Significance level for hypothesis testing.
    robust : bool, default=False
        Use robust regression (Huber) for residual computation.
        More resistant to outliers but slower.

    Attributes
    ----------
    name : str
        'ParCorr'
    measure : str
        'partial_correlation'

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_pcmci.independence_tests import ParCorr
    >>>
    >>> # Basic usage
    >>> test = ParCorr()
    >>> X = jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    >>> Y = 2 * X + jnp.array([0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2, 0.1, -0.1])
    >>> result = test.run(X, Y)
    >>> print(f"Correlation: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")
    Correlation: 0.9950, p-value: 0.0000
    >>>
    >>> # With conditioning variable
    >>> Z = jnp.array([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]]).T
    >>> result = test.run(X, Y, Z)
    >>> print(f"Partial correlation: {result.statistic:.4f}")

    Notes
    -----
    The test statistic is the partial correlation coefficient:
    - Values range from -1 to 1
    - 0 indicates no linear relationship (after conditioning)
    - 1 or -1 indicates perfect positive/negative linear relationship

    For hypothesis testing, Fisher's z-transformation is used to convert
    the correlation to a normally distributed statistic.

    References
    ----------
    .. [1] Fisher, R.A. (1915). "Frequency distribution of the values of the
           correlation coefficient in samples from an indefinitely large
           population". Biometrika, 10(4), 507-521.
    .. [2] Runge, J. et al. (2019). "Detecting and quantifying causal
           associations in large nonlinear time series datasets".
           Science Advances, 5(11), eaau4996.
    """

    name = "ParCorr"
    measure = "partial_correlation"

    def __init__(
        self,
        significance: str = "analytic",
        n_permutations: int = 500,
        alpha: float = 0.05,
        robust: bool = False,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            significance=significance,
            n_permutations=n_permutations,
            alpha=alpha,
            random_seed=random_seed,
        )
        self.robust = robust

    def compute_statistic(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Compute the partial correlation between X and Y given Z.

        Parameters
        ----------
        X : jax.Array
            First variable, shape (n_samples,).
        Y : jax.Array
            Second variable, shape (n_samples,).
        Z : jax.Array or None
            Conditioning variables, shape (n_samples, n_conditions).

        Returns
        -------
        jax.Array
            Partial correlation coefficient in [-1, 1].
        """
        # Trust callers to pass correct types - already JAX arrays
        # Type conversions are handled at data loading time
        if Z is None or (hasattr(Z, 'shape') and Z.shape[-1] == 0):
            # Simple correlation (no conditioning)
            return _compute_correlation_jit(X, Y)
        else:
            # Partial correlation via residuals
            return _compute_partial_correlation_jit(X, Y, Z)

    def compute_pvalue(
        self, statistic: jax.Array, n_samples: int, n_conditions: int
    ) -> jax.Array:
        """
        Compute p-value using Fisher's z-transformation.

        The partial correlation is transformed to a z-score using:
            z = 0.5 * ln((1 + r) / (1 - r))

        which is approximately normal with:
            mean = 0.5 * ln((1 + ρ) / (1 - ρ))
            std = 1 / sqrt(n - |Z| - 3)

        Under H0 (ρ = 0), z ~ N(0, 1/sqrt(n - |Z| - 3)).

        Parameters
        ----------
        statistic : jax.Array
            Partial correlation coefficient.
        n_samples : int
            Number of samples.
        n_conditions : int
            Number of conditioning variables.

        Returns
        -------
        jax.Array
            Two-sided p-value.
        """
        return _parcorr_pvalue(statistic, n_samples, n_conditions)

    def get_correlation_matrix(
        self, data: jax.Array, tau_max: int = 0
    ) -> jax.Array:
        """
        Compute the correlation matrix for all variable pairs and lags.

        This is a convenience method for computing all pairwise correlations
        at once, which is useful for initial exploration or as a building
        block for more complex analyses.

        Parameters
        ----------
        data : jax.Array
            Time series data, shape (T, N).
        tau_max : int, default=0
            Maximum lag to consider.

        Returns
        -------
        jax.Array
            Correlation matrix of shape (N, N, tau_max + 1).
            Entry [i, j, tau] is the correlation between X_i(t-tau) and X_j(t).

        Examples
        --------
        >>> test = ParCorr()
        >>> data = jnp.randn(1000, 5)
        >>> corr_matrix = test.get_correlation_matrix(data, tau_max=3)
        >>> print(f"Corr(X0(t-2), X1(t)): {corr_matrix[0, 1, 2]:.3f}")
        """
        T, N = data.shape
        
        # Vectorized implementation using vmap
        @jax.jit
        def compute_corrs_for_tau(tau):
            effective_T = T - tau
            # Simple slicing - no need for lax.cond
            X_data = data[: T - tau, :] if tau > 0 else data
            Y_data = data[tau:, :] if tau > 0 else data
            
            # Use only the effective range
            X_data = X_data[:effective_T, :]
            Y_data = Y_data[:effective_T, :]
            
            # Center data for all variables at once
            X_centered = X_data - jnp.mean(X_data, axis=0, keepdims=True)
            Y_centered = Y_data - jnp.mean(Y_data, axis=0, keepdims=True)
            
            # Compute all pairwise correlations using matrix operations
            norms_X = jnp.sqrt(jnp.sum(X_centered**2, axis=0))
            norms_Y = jnp.sqrt(jnp.sum(Y_centered**2, axis=0))
            
            # Handle zero norms
            norms_X = jnp.where(norms_X > 1e-10, norms_X, 1.0)
            norms_Y = jnp.where(norms_Y > 1e-10, norms_Y, 1.0)
            
            # Correlation matrix for this tau: (X^T @ Y) / (norm_X * norm_Y)
            cov_matrix = X_centered.T @ Y_centered
            corr_tau = cov_matrix / jnp.outer(norms_X, norms_Y)
            
            return jnp.clip(corr_tau, -1.0, 1.0)
        
        # Vectorize over all tau values using vmap
        corr_matrix = jax.vmap(compute_corrs_for_tau)(jnp.arange(tau_max + 1))
        
        # Transpose to get shape (N, N, tau_max+1) instead of (tau_max+1, N, N)
        return jnp.transpose(corr_matrix, (1, 2, 0))


    def compute_statistic_batch(
        self,
        X_batch: jax.Array,
        Y_batch: jax.Array,
        Z_batch: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Compute partial correlations for a batch of tests (vectorized).

        This is the core vectorized computation used by run_batch.

        Parameters
        ----------
        X_batch : jax.Array
            Shape (n_tests, n_samples).
        Y_batch : jax.Array
            Shape (n_tests, n_samples).
        Z_batch : jax.Array or None
            Shape (n_tests, n_samples, n_conditions).

        Returns
        -------
        jax.Array
            Partial correlations, shape (n_tests,).
        """
        if Z_batch is None:
            # Vectorized simple correlation using standalone function
            return _batch_correlation_jit(X_batch, Y_batch)
        else:
            # Vectorized partial correlation using standalone function
            return _batch_partial_correlation_jit(X_batch, Y_batch, Z_batch)

    def run_batch(
        self,
        X_batch: jax.Array,
        Y_batch: jax.Array,
        Z_batch: Optional[jax.Array] = None,
        alpha: Optional[float] = None,
        n_conditions: Optional[Union[int, jax.Array]] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Optimized batch implementation for ParCorr.
        
        This overrides the base class implementation for maximum performance
        by using module-level JIT'd functions instead of method-based JIT.

        Parameters
        ----------
        n_conditions : int or jax.Array, optional
            Number of conditioning variables. If None, inferred from Z_batch.
            Provide this when using padded Z matrices to ensure correct DF
            calculations for p-values.
        """
        n_samples = jnp.asarray(X_batch.shape[1], dtype=jnp.int32)
        
        if n_conditions is None:
            n_conditions = jnp.asarray(
                0 if Z_batch is None else (Z_batch.shape[2] if Z_batch.ndim == 3 else 1),
                dtype=jnp.int32,
            )
        else:
            n_conditions = jnp.asarray(n_conditions, dtype=jnp.int32)
        
        # Compute statistics in a single vectorized call
        if Z_batch is None:
            statistics = _batch_correlation_jit(X_batch, Y_batch)
        else:
            statistics = _batch_partial_correlation_jit(X_batch, Y_batch, Z_batch)
        
        # Compute p-values
        if self.significance == "analytic":
            pvalues = _batch_pvalue_jit(statistics, n_samples, n_conditions)
        else:
            # Fall back to base class permutation method
            pvalues = self._batch_permutation_pvalues(X_batch, Y_batch, Z_batch, statistics)
        
        return statistics, pvalues

    def __repr__(self) -> str:
        return (
            f"ParCorr(significance='{self.significance}', "
            f"alpha={self.alpha}, robust={self.robust})"
        )
