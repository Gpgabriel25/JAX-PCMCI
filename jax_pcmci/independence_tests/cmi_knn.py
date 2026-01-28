"""
Conditional Mutual Information (CMI) using k-Nearest Neighbors
===============================================================

This module implements the CMI-kNN estimator for nonlinear conditional
independence testing, based on the Kraskov-Stögbauer-Grassberger (KSG)
estimator.

Mathematical Background
-----------------------
Conditional Mutual Information is defined as:

    I(X; Y | Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)

where H denotes entropy. For continuous variables, this is estimated
using k-NN distances in the joint and marginal spaces.

Under the null hypothesis of conditional independence (X ⊥ Y | Z),
I(X; Y | Z) = 0.

Example
-------
>>> from jax_pcmci.independence_tests import CMIKnn
>>> import jax.numpy as jnp
>>>
>>> test = CMIKnn(k=5)
>>> X = jnp.array([1., 2., 3., 4., 5.])
>>> Y = jnp.sin(X)  # Nonlinear relationship
>>> result = test.run(X, Y)
>>> print(f"CMI: {result.statistic:.3f}")
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import digamma
from jax.scipy.stats import chi2

from jax_pcmci.independence_tests.base import CondIndTest
from jax_pcmci.config import get_config


class CMIKnn(CondIndTest):
    """
    Conditional Mutual Information test using k-Nearest Neighbors.

    This nonlinear independence test estimates conditional mutual information
    using the KSG estimator with k-nearest neighbors. It can detect nonlinear
    dependencies that linear tests like ParCorr would miss.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbors for CMI estimation.
        Higher k gives lower variance but higher bias.
    significance : str, default='analytic'
        Method for computing p-values:
        - 'analytic': Fast chi-squared approximation (default)
        - 'permutation': Recommended for highest accuracy
    n_permutations : int, default=200
        Number of permutations for significance testing (only used when
        significance='permutation').
    alpha : float, default=0.05
        Significance level.
    metric : str, default='chebyshev'
        Distance metric for k-NN:
        - 'chebyshev': Maximum norm (recommended, matches KSG paper)
        - 'euclidean': L2 norm
    standardize : bool, default=True
        Whether to standardize variables before computing distances.
        Recommended for variables on different scales.

    Attributes
    ----------
    name : str
        'CMIKnn'
    measure : str
        'conditional_mutual_information'

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_pcmci.independence_tests import CMIKnn
    >>>
    >>> # Basic nonlinear relationship
    >>> test = CMIKnn(k=10, n_permutations=200)
    >>> key = jax.random.PRNGKey(0)
    >>> X = jax.random.normal(key, shape=(500,))
    >>> Y = jnp.sin(2 * X) + 0.1 * jax.random.normal(key, shape=(500,))
    >>> result = test.run(X, Y)
    >>> print(f"CMI: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")
    >>>
    >>> # With conditioning variables
    >>> Z = jax.random.normal(key, shape=(500, 2))
    >>> result = test.run(X, Y, Z)

    Notes
    -----
    CMI-kNN is computationally more expensive than ParCorr due to the
    k-NN search. For large datasets, consider:

    1. Reducing k (trades off accuracy for speed)
    2. Subsampling the data
    3. Using GPU acceleration

    The test statistic (CMI) is always non-negative. Values close to 0
    indicate independence, while higher values indicate stronger dependence.

    References
    ----------
    .. [1] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
           "Estimating mutual information". Physical Review E, 69(6).
    .. [2] Frenzel, S., & Pompe, B. (2007). "Partial mutual information
           for coupling analysis of multivariate time series".
           Physical Review Letters, 99(20).
    """

    name = "CMIKnn"
    measure = "conditional_mutual_information"

    def __init__(
        self,
        k: int = 10,
        significance: str = "analytic",
        n_permutations: int = 200,
        alpha: float = 0.05,
        metric: str = "chebyshev",
        standardize: bool = True,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            significance=significance,
            n_permutations=n_permutations,
            alpha=alpha,
            random_seed=random_seed,
        )
        self.k = k
        self.metric = metric.lower()
        self.standardize = standardize

        if self.metric not in ("chebyshev", "euclidean"):
            raise ValueError(f"metric must be 'chebyshev' or 'euclidean', got '{metric}'")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

    @partial(jax.jit, static_argnums=(0,))
    def compute_statistic(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Compute Conditional Mutual Information I(X; Y | Z).

        Uses the KSG estimator based on k-nearest neighbor distances.

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
            CMI estimate (non-negative scalar).
        """
        # Use _prepare_inputs which handles dtype conversion, reshaping, and standardization
        if Z is None:
            X_prep, Y_prep, _ = self._prepare_inputs(X, Y, None)
            return self._compute_mi_standardized(X_prep, Y_prep)

        X_prep, Y_prep, Z_prep = self._prepare_inputs(X, Y, Z)
        return self._compute_cmi_standardized(X_prep, Y_prep, Z_prep)

    def _compute_mi_standardized(self, X: jax.Array, Y: jax.Array) -> jax.Array:
        """
        Compute Mutual Information I(X; Y) using KSG estimator.

        Inputs are assumed to be pre-processed (dtype/shape/optional standardization).
        """
        n = X.shape[0]
        k = min(self.k, n - 1)

        # Joint space XY
        XY = jnp.concatenate([X, Y], axis=1)

        # Compute distances once for joint space and reuse for counting
        dist_XY = self._compute_distances(XY)
        eps = self._kth_neighbor_from_dist(dist_XY, k)

        # Count neighbors using precomputed marginal distances
        dist_X = self._compute_distances(X)
        dist_Y = self._compute_distances(Y)
        n_X = self._count_neighbors_from_dist(dist_X, eps)
        n_Y = self._count_neighbors_from_dist(dist_Y, eps)

        # KSG estimator
        mi = digamma(k) - jnp.mean(digamma(n_X + 1) + digamma(n_Y + 1)) + digamma(n)

        return jnp.maximum(mi, 0.0)

    def _compute_cmi_standardized(
        self, X: jax.Array, Y: jax.Array, Z: jax.Array
    ) -> jax.Array:
        """
        Compute Conditional Mutual Information I(X; Y | Z).

        Inputs are assumed to be pre-processed (dtype/shape/optional standardization).
        Uses the Frenzel-Pompe estimator:
            I(X; Y | Z) = ψ(k) - <ψ(n_XZ + 1) + ψ(n_YZ + 1) - ψ(n_Z + 1)>
        
        Optimized: For Chebyshev metric, joint distances are max of component distances.
        We compute X, Y, Z distances once and combine them efficiently.
        """
        n = X.shape[0]
        k = min(self.k, n - 1)

        # For Chebyshev metric, d(XYZ) = max(d(X), d(Y), d(Z))
        # This allows us to compute component distances once and reuse
        if self.metric == "chebyshev":
            # Compute component distances once
            dist_X = self._chebyshev_distances(X, X)
            dist_Y = self._chebyshev_distances(Y, Y)
            dist_Z = self._chebyshev_distances(Z, Z)
            
            # Joint distances via max (Chebyshev property)
            dist_XZ = jnp.maximum(dist_X, dist_Z)
            dist_YZ = jnp.maximum(dist_Y, dist_Z)
            dist_XYZ = jnp.maximum(dist_XZ, dist_Y)  # = max(X, Y, Z)
        else:
            # Euclidean: need to compute full joint spaces
            XZ = jnp.concatenate([X, Z], axis=1)
            YZ = jnp.concatenate([Y, Z], axis=1)
            XYZ = jnp.concatenate([X, Y, Z], axis=1)
            dist_XZ = self._euclidean_distances(XZ, XZ)
            dist_YZ = self._euclidean_distances(YZ, YZ)
            dist_XYZ = self._euclidean_distances(XYZ, XYZ)
            dist_Z = self._euclidean_distances(Z, Z)

        # Find eps from joint space XYZ
        eps = self._kth_neighbor_from_dist(dist_XYZ, k)

        # Count neighbors in subspaces
        n_XZ = self._count_neighbors_from_dist(dist_XZ, eps)
        n_YZ = self._count_neighbors_from_dist(dist_YZ, eps)
        n_Z = self._count_neighbors_from_dist(dist_Z, eps)

        # Frenzel-Pompe CMI estimator
        cmi = (
            digamma(k)
            - jnp.mean(digamma(n_XZ + 1) + digamma(n_YZ + 1) - digamma(n_Z + 1))
        )

        return jnp.maximum(cmi, 0.0)

    def _compute_distances(self, data: jax.Array) -> jax.Array:
        """
        Compute pairwise distances for a dataset.
        """
        if self.metric == "chebyshev":
            return self._chebyshev_distances(data, data)
        else:
            return self._euclidean_distances(data, data)

    def _kth_neighbor_from_dist(self, distances: jax.Array, k: int) -> jax.Array:
        """
        Find the k-th nearest neighbor distance from precomputed distance matrix.
        """
        n = distances.shape[0]
        # Set self-distance to infinity
        distances_masked = distances + jnp.eye(n) * jnp.inf
        # Get k-th neighbor without a full sort (much faster than sort)
        kth_distances = jnp.partition(distances_masked, k - 1, axis=1)[:, k - 1]
        return kth_distances

    def _count_neighbors_from_dist(self, distances: jax.Array, eps: jax.Array) -> jax.Array:
        """
        Count neighbors within eps distance from precomputed distance matrix.
        """
        # Count points strictly within eps (excluding self where distance == 0)
        within_eps = (distances < eps.reshape(-1, 1)) & (distances > 0)
        return jnp.sum(within_eps, axis=1)

    def _kth_neighbor_distance(self, data: jax.Array, k: int) -> jax.Array:
        """
        Find the distance to the k-th nearest neighbor for each point.
        (Legacy method - kept for compatibility)
        """
        distances = self._compute_distances(data)
        return self._kth_neighbor_from_dist(distances, k)

    def _count_neighbors(self, data: jax.Array, eps: jax.Array) -> jax.Array:
        """
        Count number of points within distance eps for each point.
        (Legacy method - kept for compatibility)
        """
        distances = self._compute_distances(data)
        return self._count_neighbors_from_dist(distances, eps)

    @staticmethod
    @jax.jit
    def _chebyshev_distances(X: jax.Array, Y: jax.Array) -> jax.Array:
        """
        Compute Chebyshev (max-norm) distances between all pairs.
        """
        # X: (n, d), Y: (m, d) -> output: (n, m)
        return jnp.max(jnp.abs(X[:, None, :] - Y[None, :, :]), axis=2)

    @staticmethod
    @jax.jit
    def _euclidean_distances(X: jax.Array, Y: jax.Array) -> jax.Array:
        """
        Compute Euclidean distances between all pairs.
        """
        # Using the identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        XX = jnp.sum(X * X, axis=1)
        YY = jnp.sum(Y * Y, axis=1)
        XY = X @ Y.T
        distances_sq = XX[:, None] + YY[None, :] - 2 * XY
        return jnp.sqrt(jnp.maximum(distances_sq, 0.0))

    @staticmethod
    def _chebyshev_distances_chunked(
        X: jax.Array, Y: jax.Array, chunk_size: int = 256
    ) -> jax.Array:
        """
        Compute Chebyshev distances in chunks for memory efficiency.
        
        For large n, the full n×n distance matrix can exceed GPU memory.
        This computes distances in chunks to reduce peak memory usage.
        """
        n = X.shape[0]
        m = Y.shape[0]
        
        # Initialize output
        distances = jnp.zeros((n, m), dtype=X.dtype)
        
        # Process in chunks
        for i_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            X_chunk = X[i_start:i_end]
            
            for j_start in range(0, m, chunk_size):
                j_end = min(j_start + chunk_size, m)
                Y_chunk = Y[j_start:j_end]
                
                # Compute chunk distances
                chunk_dist = jnp.max(
                    jnp.abs(X_chunk[:, None, :] - Y_chunk[None, :, :]), axis=2
                )
                distances = distances.at[i_start:i_end, j_start:j_end].set(chunk_dist)
        
        return distances

    @partial(jax.jit, static_argnums=(0,))
    def compute_pvalue(
        self, statistic: jax.Array, n_samples: int, n_conditions: int
    ) -> jax.Array:
        """
        Compute p-value for CMI.

        For CMI, there's no simple analytical null distribution, so this
        returns a placeholder. Use significance='permutation' for accurate
        p-values.

        Parameters
        ----------
        statistic : jax.Array
            CMI value.
        n_samples : int
            Number of samples.
        n_conditions : int
            Number of conditioning variables.

        Returns
        -------
        jax.Array
            Approximate p-value (use permutation for accuracy).
        """
        # For CMI, we typically need permutation testing
        # This is a rough approximation based on chi-squared null
        # Under H0, 2*n*CMI ~ chi^2(1) approximately
        chi2_stat = 2 * n_samples * statistic
        pvalue = 1.0 - chi2.cdf(chi2_stat, df=1)

        return jnp.clip(pvalue, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"CMIKnn(k={self.k}, significance='{self.significance}', "
            f"alpha={self.alpha}, metric='{self.metric}')"
        )

    # ----- Overrides for permutation/bootstrap hooks ------------------------

    def _prepare_inputs(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array]
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        dtype = get_config().dtype
        X_arr = jnp.asarray(X, dtype=dtype).reshape(-1, 1)
        Y_arr = jnp.asarray(Y, dtype=dtype).reshape(-1, 1)

        if Z is None:
            if self.standardize:
                X_arr, Y_arr = self._standardize_pair(X_arr, Y_arr)
            return X_arr, Y_arr, None

        Z_arr = jnp.asarray(Z, dtype=dtype)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)

        if self.standardize:
            X_arr, Y_arr, Z_arr = self._standardize_triplet(X_arr, Y_arr, Z_arr)

        return X_arr, Y_arr, Z_arr

    def _statistic_from_prepared(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array]
    ) -> jax.Array:
        if Z is None:
            return self._compute_mi_standardized(X, Y)
        return self._compute_cmi_standardized(X, Y, Z)

    # ----- Standardization helpers ------------------------------------------

    def _standardize_array(self, arr: jax.Array) -> jax.Array:
        mean = jnp.mean(arr, axis=0)
        std = jnp.std(arr, axis=0) + 1e-10
        return (arr - mean) / std

    def _standardize_pair(
        self, X: jax.Array, Y: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        return self._standardize_array(X), self._standardize_array(Y)

    def _standardize_triplet(
        self, X: jax.Array, Y: jax.Array, Z: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        return (
            self._standardize_array(X),
            self._standardize_array(Y),
            self._standardize_array(Z),
        )


class CMISymbolic(CondIndTest):
    """
    Symbolic CMI for fast nonlinear independence testing.

    This is a faster alternative to CMI-kNN that discretizes continuous
    variables into symbols and uses discrete entropy estimation.

    Parameters
    ----------
    n_symbols : int, default=6
        Number of symbols for discretization.
    significance : str, default='analytic'
        Method for computing p-values.
    alpha : float, default=0.05
        Significance level.

    Notes
    -----
    Symbolic CMI is much faster than CMI-kNN but may miss subtle
    nonlinear dependencies due to discretization.
    """

    name = "CMISymbolic"
    measure = "symbolic_cmi"

    def __init__(
        self,
        n_symbols: int = 6,
        significance: str = "analytic",
        n_permutations: int = 500,
        alpha: float = 0.05,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            significance=significance,
            n_permutations=n_permutations,
            alpha=alpha,
            random_seed=random_seed,
        )
        self.n_symbols = n_symbols

    @partial(jax.jit, static_argnums=(0,))
    def compute_statistic(
        self, X: jax.Array, Y: jax.Array, Z: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Compute symbolic CMI by discretizing variables.
        """
        # Discretize variables
        X_sym = self._symbolize(X)
        Y_sym = self._symbolize(Y)

        if Z is None:
            # Mutual information
            return self._discrete_mi(X_sym, Y_sym)
        else:
            Z_sym = jax.vmap(self._symbolize, in_axes=1, out_axes=1)(Z)
            return self._discrete_cmi(X_sym, Y_sym, Z_sym)

    def _symbolize(self, x: jax.Array) -> jax.Array:
        """
        Convert continuous variable to discrete symbols using equal-frequency binning.
        """
        n = len(x)
        # Sort and assign symbols based on rank
        sorted_indices = jnp.argsort(x)
        ranks = jnp.zeros(n, dtype=jnp.int32)
        ranks = ranks.at[sorted_indices].set(jnp.arange(n))

        # Map ranks to symbols
        symbols = (ranks * self.n_symbols) // n

        return symbols

    def _discrete_mi(self, X: jax.Array, Y: jax.Array) -> jax.Array:
        """
        Compute discrete mutual information.
        """
        n = len(X)

        # Joint distribution
        joint_idx = X * self.n_symbols + Y
        joint_counts = jnp.bincount(joint_idx, length=self.n_symbols**2)
        joint_prob = joint_counts / n

        # Marginal distributions
        X_counts = jnp.bincount(X, length=self.n_symbols)
        Y_counts = jnp.bincount(Y, length=self.n_symbols)
        X_prob = X_counts / n
        Y_prob = Y_counts / n

        # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
        outer_prob = jnp.outer(X_prob, Y_prob).flatten()

        # Avoid log(0)
        valid = (joint_prob > 0) & (outer_prob > 0)
        mi_terms = jnp.where(
            valid,
            joint_prob * jnp.log(joint_prob / (outer_prob + 1e-10) + 1e-10),
            0.0
        )

        return jnp.sum(mi_terms)

    def _discrete_cmi(
        self, X: jax.Array, Y: jax.Array, Z: jax.Array
    ) -> jax.Array:
        """
        Compute discrete conditional mutual information.

        I(X; Y | Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
        """
        # This is a simplified implementation
        # For full implementation, would need to iterate over Z values
        n = len(X)

        # Encode Z as single variable
        if Z.ndim == 1:
            Z_enc = Z
        else:
            # Simple encoding: treat as base-n_symbols number using vectorized reduction
            # Z_enc = Z[:, 0] * n_symbols^(d-1) + Z[:, 1] * n_symbols^(d-2) + ... + Z[:, d-1]
            d = Z.shape[1]
            powers = jnp.power(self.n_symbols, jnp.arange(d - 1, -1, -1))
            Z_enc = jnp.sum(Z.astype(jnp.int32) * powers, axis=1)

        # Compute conditional entropies
        H_X_given_Z = self._conditional_entropy(X, Z_enc)
        H_Y_given_Z = self._conditional_entropy(Y, Z_enc)
        H_XY_given_Z = self._joint_conditional_entropy(X, Y, Z_enc)

        cmi = H_X_given_Z + H_Y_given_Z - H_XY_given_Z

        return jnp.maximum(cmi, 0.0)

    def _conditional_entropy(self, X: jax.Array, Z: jax.Array) -> jax.Array:
        """
        Compute H(X|Z) = H(X,Z) - H(Z).
        """
        n = len(X)
        n_Z = int(jnp.max(Z)) + 1

        # Joint entropy H(X, Z)
        joint_idx = X * n_Z + Z
        n_joint = self.n_symbols * n_Z
        joint_counts = jnp.bincount(joint_idx, length=n_joint)
        joint_prob = joint_counts / n
        H_XZ = -jnp.sum(jnp.where(joint_prob > 0, joint_prob * jnp.log(joint_prob + 1e-10), 0.0))

        # Marginal entropy H(Z)
        Z_counts = jnp.bincount(Z, length=n_Z)
        Z_prob = Z_counts / n
        H_Z = -jnp.sum(jnp.where(Z_prob > 0, Z_prob * jnp.log(Z_prob + 1e-10), 0.0))

        return H_XZ - H_Z

    def _joint_conditional_entropy(
        self, X: jax.Array, Y: jax.Array, Z: jax.Array
    ) -> jax.Array:
        """
        Compute H(X,Y|Z) = H(X,Y,Z) - H(Z).
        """
        n = len(X)
        n_Z = int(jnp.max(Z)) + 1

        # Joint entropy H(X, Y, Z)
        joint_idx = (X * self.n_symbols + Y) * n_Z + Z
        n_joint = self.n_symbols * self.n_symbols * n_Z
        joint_counts = jnp.bincount(joint_idx, length=n_joint)
        joint_prob = joint_counts / n
        H_XYZ = -jnp.sum(jnp.where(joint_prob > 0, joint_prob * jnp.log(joint_prob + 1e-10), 0.0))

        # Marginal entropy H(Z)
        Z_counts = jnp.bincount(Z, length=n_Z)
        Z_prob = Z_counts / n
        H_Z = -jnp.sum(jnp.where(Z_prob > 0, Z_prob * jnp.log(Z_prob + 1e-10), 0.0))

        return H_XYZ - H_Z

    @partial(jax.jit, static_argnums=(0,))
    def compute_pvalue(
        self, statistic: jax.Array, n_samples: int, n_conditions: int
    ) -> jax.Array:
        """
        Compute p-value using chi-squared approximation.
        """
        # Under H0, 2*n*CMI ~ chi^2(df) where df depends on dimensions
        df = (self.n_symbols - 1) ** 2
        chi2_stat = 2 * n_samples * statistic
        pvalue = 1.0 - chi2.cdf(chi2_stat, df=df)

        return jnp.clip(pvalue, 0.0, 1.0)
