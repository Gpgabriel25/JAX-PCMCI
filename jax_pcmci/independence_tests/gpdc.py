"""
Gaussian Process Distance Correlation (GPDC) Test
==================================================

This module implements a Gaussian Process-based conditional independence
test, which uses GP regression residuals with distance correlation.

Mathematical Background
-----------------------
GPDC works by:
1. Fitting Gaussian Process regressions of X and Y on Z
2. Computing the distance correlation between the GP residuals
3. Testing whether this residual correlation is significantly non-zero

This captures nonlinear dependencies while properly conditioning on Z.

Example
-------
>>> from jax_pcmci.independence_tests import GPDCond
>>> import jax.numpy as jnp
>>>
>>> test = GPDCond(kernel='rbf')
>>> X = jnp.sin(jnp.linspace(0, 10, 100))
>>> Y = jnp.cos(jnp.linspace(0, 10, 100))
>>> Z = jnp.linspace(0, 10, 100).reshape(-1, 1)
>>> result = test.run(X, Y, Z)
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from jax_pcmci.independence_tests.base import CondIndTest
from jax_pcmci.config import get_config


class GPDCond(CondIndTest):
    """
    Gaussian Process Distance Correlation conditional independence test.

    This test combines Gaussian Process regression with distance correlation
    to detect nonlinear conditional dependencies. It regresses X and Y on Z
    using GP, then tests whether the residuals are independent using distance
    correlation.

    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel function for GP regression:
        - 'rbf': Radial Basis Function (Gaussian) kernel
        - 'matern32': Matérn 3/2 kernel
        - 'matern52': Matérn 5/2 kernel
        - callable: Custom kernel function(X1, X2, params) -> K
    length_scale : float, default=1.0
        Length scale parameter for the kernel.
    noise_var : float, default=0.1
        Observation noise variance for GP.
    significance : str, default='permutation'
        Method for p-value computation.
    n_permutations : int, default=200
        Number of permutations for significance testing.
    alpha : float, default=0.05
        Significance level.

    Attributes
    ----------
    name : str
        'GPDCond'
    measure : str
        'gp_distance_correlation'

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_pcmci.independence_tests import GPDCond
    >>>
    >>> # Nonlinear conditional dependence
    >>> test = GPDCond(kernel='rbf', length_scale=0.5)
    >>> n = 200
    >>> Z = jax.random.normal(jax.random.PRNGKey(0), (n, 1))
    >>> X = jnp.sin(Z[:, 0]) + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n,))
    >>> Y = jnp.cos(Z[:, 0]) + 0.1 * jax.random.normal(jax.random.PRNGKey(2), (n,))
    >>> result = test.run(X, Y, Z)
    >>> print(f"GPDC: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")

    Notes
    -----
    GPDC is computationally more expensive than ParCorr or CMI-kNN due to
    the GP regression step, which scales as O(n³) in the number of samples.
    For large datasets, consider subsampling or using sparse GP approximations.

    References
    ----------
    .. [1] Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
           "Measuring and testing dependence by correlation of distances".
           The Annals of Statistics, 35(6), 2769-2794.
    """

    name = "GPDCond"
    measure = "gp_distance_correlation"

    def __init__(
        self,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        noise_var: float = 0.1,
        significance: str = "permutation",
        n_permutations: int = 200,
        alpha: float = 0.05,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            significance=significance,
            n_permutations=n_permutations,
            alpha=alpha,
            random_seed=random_seed,
        )
        self.kernel_name = kernel if isinstance(kernel, str) else "custom"
        self.length_scale = length_scale
        self.noise_var = noise_var

        # Set up kernel function
        if isinstance(kernel, str):
            self.kernel = self._get_kernel(kernel)
        else:
            self.kernel = kernel

    def _get_kernel(self, name: str) -> Callable:
        """
        Get kernel function by name.
        """
        kernels = {
            "rbf": self._rbf_kernel,
            "matern32": self._matern32_kernel,
            "matern52": self._matern52_kernel,
        }
        if name.lower() not in kernels:
            raise ValueError(
                f"Unknown kernel: {name}. Available: {list(kernels.keys())}"
            )
        return kernels[name.lower()]

    @partial(jax.jit, static_argnums=(0,))
    def _rbf_kernel(
        self, X1: jax.Array, X2: jax.Array
    ) -> jax.Array:
        """
        Radial Basis Function (Gaussian) kernel.

        k(x1, x2) = exp(-||x1 - x2||² / (2 * l²))
        """
        # Squared distances
        sq_dists = self._squared_distances(X1, X2)
        return jnp.exp(-sq_dists / (2 * self.length_scale ** 2))

    @partial(jax.jit, static_argnums=(0,))
    def _matern32_kernel(
        self, X1: jax.Array, X2: jax.Array
    ) -> jax.Array:
        """
        Matérn 3/2 kernel.

        k(x1, x2) = (1 + √3 * r / l) * exp(-√3 * r / l)
        where r = ||x1 - x2||
        """
        dists = jnp.sqrt(self._squared_distances(X1, X2) + 1e-10)
        scaled = jnp.sqrt(3.0) * dists / self.length_scale
        return (1 + scaled) * jnp.exp(-scaled)

    @partial(jax.jit, static_argnums=(0,))
    def _matern52_kernel(
        self, X1: jax.Array, X2: jax.Array
    ) -> jax.Array:
        """
        Matérn 5/2 kernel.

        k(x1, x2) = (1 + √5 * r / l + 5 * r² / (3 * l²)) * exp(-√5 * r / l)
        """
        sq_dists = self._squared_distances(X1, X2)
        dists = jnp.sqrt(sq_dists + 1e-10)
        scaled = jnp.sqrt(5.0) * dists / self.length_scale
        return (1 + scaled + sq_dists * 5 / (3 * self.length_scale ** 2)) * jnp.exp(-scaled)

    @staticmethod
    @jax.jit
    def _squared_distances(X1: jax.Array, X2: jax.Array) -> jax.Array:
        """
        Compute squared Euclidean distances between all pairs.
        """
        # Ensure 2D
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        XX1 = jnp.sum(X1 * X1, axis=1)
        XX2 = jnp.sum(X2 * X2, axis=1)
        XY = X1 @ X2.T
        return jnp.maximum(XX1[:, None] + XX2[None, :] - 2 * XY, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def compute_statistic(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Compute the GPDC statistic.

        If Z is None, computes distance correlation between X and Y.
        If Z is provided, computes distance correlation between GP residuals.

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
            GPDC statistic (distance correlation of residuals).
        """
        dtype = get_config().dtype
        X = jnp.asarray(X, dtype=dtype)
        Y = jnp.asarray(Y, dtype=dtype)

        if Z is None:
            # Unconditional: just distance correlation
            return self._distance_correlation(X, Y)
        else:
            Z = jnp.asarray(Z, dtype=dtype)
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)

            # Get GP residuals
            X_residual = self._gp_residual(X, Z)
            Y_residual = self._gp_residual(Y, Z)

            # Distance correlation of residuals
            return self._distance_correlation(X_residual, Y_residual)

    def _gp_residual(self, y: jax.Array, X: jax.Array) -> jax.Array:
        """
        Compute residuals from GP regression of y on X.

        Uses the GP posterior mean as the prediction.
        """
        n = len(y)

        # Compute kernel matrix
        K = self.kernel(X, X)

        # Add noise variance to diagonal
        K_noisy = K + self.noise_var * jnp.eye(n)

        # Solve K⁻¹ y using Cholesky
        L = jnp.linalg.cholesky(K_noisy + 1e-6 * jnp.eye(n))
        alpha = jax.scipy.linalg.solve_triangular(
            L.T,
            jax.scipy.linalg.solve_triangular(L, y, lower=True),
            lower=False,
        )

        # Posterior mean at training points
        y_pred = K @ alpha

        # Residuals
        return y - y_pred

    @staticmethod
    @jax.jit
    def _distance_correlation(X: jax.Array, Y: jax.Array) -> jax.Array:
        """
        Compute distance correlation between X and Y.

        Distance correlation is zero if and only if X and Y are independent
        (for sufficiently rich distributions).
        """
        n = len(X)

        # Ensure 2D for distance computation
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # Distance matrices
        D_X = jnp.sqrt(GPDCond._squared_distances(X, X) + 1e-10)
        D_Y = jnp.sqrt(GPDCond._squared_distances(Y, Y) + 1e-10)

        # Double-centered distance matrices (Abar, Bbar in Székely notation)
        A = GPDCond._double_center(D_X)
        B = GPDCond._double_center(D_Y)

        # Distance covariance
        dcov_XY = jnp.sqrt(jnp.maximum(jnp.sum(A * B) / (n * n), 0.0))
        dcov_XX = jnp.sqrt(jnp.maximum(jnp.sum(A * A) / (n * n), 0.0))
        dcov_YY = jnp.sqrt(jnp.maximum(jnp.sum(B * B) / (n * n), 0.0))

        # Distance correlation
        denom = jnp.sqrt(dcov_XX * dcov_YY)
        dcor = jnp.where(denom > 1e-10, dcov_XY / denom, 0.0)

        return dcor

    @staticmethod
    @jax.jit
    def _double_center(D: jax.Array) -> jax.Array:
        """
        Double-center a distance matrix.

        A_ij = D_ij - mean(D_i.) - mean(D_.j) + mean(D)
        """
        n = D.shape[0]
        row_mean = jnp.mean(D, axis=1, keepdims=True)
        col_mean = jnp.mean(D, axis=0, keepdims=True)
        grand_mean = jnp.mean(D)

        return D - row_mean - col_mean + grand_mean

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def compute_pvalue(
        self, statistic: jax.Array, n_samples: int, n_conditions: int
    ) -> jax.Array:
        """
        Compute p-value for GPDC statistic.

        Uses an approximate null distribution based on the t-distribution.

        Parameters
        ----------
        statistic : jax.Array
            Distance correlation value.
        n_samples : int
            Number of samples.
        n_conditions : int
            Number of conditioning variables.

        Returns
        -------
        jax.Array
            Approximate p-value.
        """
        # Approximate t-statistic for distance correlation
        # Under H0, T = sqrt((n-2)/(1-R²)) * R ~ t(n-2) approximately
        n = n_samples
        R = statistic

        # Avoid numerical issues
        R_sq = jnp.clip(R ** 2, 0.0, 0.9999)
        T = jnp.sqrt((n - 2) / (1 - R_sq + 1e-10)) * R

        # Two-sided p-value from t-distribution
        from jax.scipy.stats import t as t_dist
        pvalue = 2 * (1 - t_dist.cdf(jnp.abs(T), df=n - 2))

        return jnp.clip(pvalue, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"GPDCond(kernel='{self.kernel_name}', length_scale={self.length_scale}, "
            f"significance='{self.significance}', alpha={self.alpha})"
        )
