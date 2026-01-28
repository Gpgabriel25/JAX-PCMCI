"""
PCMCI Algorithm Implementation
==============================

This module implements the PCMCI algorithm for causal discovery from
time series data. PCMCI combines the PC algorithm (for conditioning set
selection) with Momentary Conditional Independence (MCI) testing.

Algorithm Overview
------------------
PCMCI proceeds in two phases:

1. **PC1 Phase (Condition Selection)**:
   For each target variable, iteratively remove spurious parents by
   testing conditional independence with increasing conditioning sets.

2. **MCI Phase (Causal Link Identification)**:
   Test each remaining potential link using momentary conditional
   independence, conditioning on the parents of both source and target.

Example
-------
>>> from jax_pcmci import PCMCI, ParCorr, DataHandler
>>> import jax.numpy as jnp
>>>
>>> # Generate sample data
>>> data = jnp.randn(1000, 5)
>>> handler = DataHandler(data)
>>>
>>> # Run PCMCI
>>> pcmci = PCMCI(handler, cond_ind_test=ParCorr())
>>> results = pcmci.run(tau_max=3, pc_alpha=0.05)
>>>
>>> # Examine results
>>> print(results.summary())
>>> results.plot_graph()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import math
from itertools import combinations
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from tqdm import tqdm

from jax_pcmci.data import DataHandler
from jax_pcmci.independence_tests.base import CondIndTest, TestResult
from jax_pcmci.independence_tests.parcorr import ParCorr
from jax_pcmci.results import PCMCIResults
from jax_pcmci.config import get_config


class PCMCI:
    """
    PCMCI algorithm for time series causal discovery.

    PCMCI is a constraint-based causal discovery method that identifies
    causal relationships in time series data. It combines efficient
    condition selection (PC algorithm) with robust statistical testing
    (Momentary Conditional Independence).

    Parameters
    ----------
    datahandler : DataHandler
        Data handler containing the time series data.
    cond_ind_test : CondIndTest, optional
        Conditional independence test to use. Default is ParCorr.
    verbosity : int, default=1
        Level of output detail:
        - 0: Silent
        - 1: Basic progress info
        - 2: Detailed progress
        - 3: Debug output
    selected_variables : list of int, optional
        Indices of target variables to analyze. Default is all variables.

    Attributes
    ----------
    T : int
        Number of time points.
    N : int
        Number of variables.
    var_names : list of str
        Variable names.
    test : CondIndTest
        The conditional independence test being used.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_pcmci import PCMCI, ParCorr, DataHandler
    >>>
    >>> # Simple linear VAR process
    >>> T, N = 500, 3
    >>> data = jnp.zeros((T, N))
    >>> key = jax.random.PRNGKey(42)
    >>> noise = jax.random.normal(key, (T, N)) * 0.5
    >>>
    >>> # Create data handler
    >>> handler = DataHandler(data + noise)
    >>>
    >>> # Run PCMCI with different tests
    >>> pcmci_linear = PCMCI(handler, cond_ind_test=ParCorr())
    >>> results = pcmci_linear.run(tau_max=2)
    >>>
    >>> # Access results
    >>> print(f"Number of significant links: {results.n_significant_links}")

    Notes
    -----
    PCMCI is designed for:
    - Time series with potential contemporaneous confounders
    - Linear (ParCorr) or nonlinear (CMI) dependencies
    - Stationary processes (non-stationarity may require preprocessing)

    The algorithm complexity is O(N² * tau_max * 2^p_max) where p_max is the
    maximum conditioning set size. GPU parallelization significantly speeds
    up the independence tests.

    References
    ----------
    .. [1] Runge, J. et al. (2019). "Detecting and quantifying causal
           associations in large nonlinear time series datasets".
           Science Advances, 5(11), eaau4996.
    .. [2] Runge, J. (2018). "Causal network reconstruction from time series:
           From theoretical assumptions to practical estimation".
           Chaos, 28(7), 075310.

    See Also
    --------
    PCMCIPlus : Extended algorithm for contemporaneous causal discovery.
    ParCorr : Partial correlation test for linear dependencies.
    CMIKnn : CMI test for nonlinear dependencies.
    """

    def __init__(
        self,
        datahandler: DataHandler,
        cond_ind_test: Optional[CondIndTest] = None,
        verbosity: int = 1,
        selected_variables: Optional[List[int]] = None,
    ):
        self.datahandler = datahandler
        self.test = cond_ind_test if cond_ind_test is not None else ParCorr()
        self.verbosity = verbosity

        # Set selected variables
        self.N = datahandler.N
        self.T = datahandler.T
        self.var_names = datahandler.var_names

        if selected_variables is None:
            self.selected_variables = list(range(self.N))
        else:
            self.selected_variables = selected_variables

        # Internal state for algorithm
        self._parents: Dict[int, Set[Tuple[int, int]]] = {}
        self._pval_matrix: Optional[jax.Array] = None
        self._val_matrix: Optional[jax.Array] = None
        self._batch_size_cache: Dict[Tuple[int, int], Optional[int]] = {}

    def _sample_condition_subsets(
        self,
        items: List[Tuple[int, int]],
        k: int,
        max_subsets: int,
        seed: int,
    ) -> List[Tuple[Tuple[int, int], ...]]:
        """
        Reservoir-sample at most max_subsets k-sized combinations from items.

        Falls back to full enumeration only when the total count is small
        enough, avoiding large intermediate lists.
        """
        if k == 0:
            return [tuple()]
        if len(items) < k:
            return []

        total = math.comb(len(items), k)
        if total <= max_subsets:
            return list(combinations(items, k))

        # Use NumPy for faster random sampling
        rng = np.random.RandomState(seed)
        reservoir: List[Tuple[Tuple[int, int], ...]] = []
        
        # Pre-sample indices for faster reservoir sampling
        # Sample up front to reduce overhead
        comb_iter = combinations(items, k)
        for idx in range(max_subsets):
            reservoir.append(next(comb_iter))
        
        # Continue with reservoir sampling for remaining
        for idx in range(max_subsets, min(total, max_subsets * 10)):
            try:
                combo = next(comb_iter)
                j = rng.randint(0, idx + 1)
                if j < max_subsets:
                    reservoir[j] = combo
            except StopIteration:
                break

        return reservoir

    def _get_effective_batch_size(self, n_samples: Optional[int] = None, n_conditions: int = 0) -> Optional[int]:
        """
        Resolve effective batch size for memory-aware batching.
        
        Dynamically computes batch size based on available GPU memory
        if not explicitly configured.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples per test (for memory estimation).
        n_conditions : int, default=0
            Number of conditioning variables (for memory estimation).
            
        Returns
        -------
        int or None
            Batch size, or None for unlimited batching.
        """
        config = get_config()
        if config.batch_size is not None:
            return config.batch_size
        if config.memory_efficient:
            return 256
            
        # Auto-compute batch size based on GPU memory
        if n_samples is not None:
            cache_key = (n_samples, n_conditions)
            if cache_key in self._batch_size_cache:
                return self._batch_size_cache[cache_key]
            try:
                device = jax.devices()[0]
                if hasattr(device, 'memory_stats'):
                    mem_stats = device.memory_stats() or {}
                    # Use bytes_limit if available, otherwise estimate 8GB
                    total_mem = mem_stats.get('bytes_limit', 8 * 1024**3)
                    in_use = mem_stats.get('bytes_in_use', 0)
                    available = (total_mem - in_use) * 0.7  # Use 70% of available
                    
                    # Estimate bytes per test: (X + Y + Z) * dtype_size * 3 (intermediates)
                    dtype_size = jnp.dtype(config.dtype).itemsize
                    bytes_per_test = n_samples * (2 + n_conditions) * dtype_size * 3
                    
                    if bytes_per_test > 0:
                        computed_batch = max(64, int(available / bytes_per_test))
                        batch = min(computed_batch, 8192)  # Cap at reasonable max
                        self._batch_size_cache[cache_key] = batch
                        return batch
            except Exception:
                pass  # Fall through to default

            # Default safe batch size if memory stats unavailable or failed
            self._batch_size_cache[cache_key] = 4096
            return 4096
        
        return 4096

    def run(
        self,
        tau_max: int = 1,
        tau_min: int = 1,
        pc_alpha: Optional[float] = 0.05,
        max_conds_dim: Optional[int] = None,
        max_conds_py: Optional[int] = None,
        max_conds_px: Optional[int] = None,
        max_subsets: int = 100,
        alpha_level: float = 0.05,
        fdr_method: Optional[str] = None,
    ) -> PCMCIResults:
        """
        Run the PCMCI algorithm.

        This is the main entry point for running causal discovery.
        It executes the PC condition selection phase followed by the
        MCI phase for final causal link identification.

        Parameters
        ----------
        tau_max : int, default=1
            Maximum time lag to test.
        tau_min : int, default=1
            Minimum time lag to test. Use tau_min=0 for contemporaneous
            effects (but consider PCMCIPlus for this case).
        pc_alpha : float or None, default=0.05
            Significance level for PC condition selection phase.
            If None, all lagged parents are used (no selection).
        max_conds_dim : int or None
            Maximum conditioning set size. None means no limit.
        max_conds_py : int or None
            Maximum conditions from target's parents. None means no limit.
        max_conds_px : int or None
            Maximum conditions from source's parents. None means no limit.
        max_subsets : int, default=100
            Maximum conditioning subsets to test per parent in PC phase.
            Limits combinatorial explosion for high-dimensional data.
        alpha_level : float, default=0.05
            Significance level for final MCI tests.
        fdr_method : str or None
            Method for false discovery rate correction:
            - None: No correction
            - 'fdr_bh': Benjamini-Hochberg FDR
            - 'bonferroni': Bonferroni correction

        Returns
        -------
        PCMCIResults
            Results object containing:
            - val_matrix: Test statistic values
            - pval_matrix: P-values
            - graph: Significant causal links
            - parents: Identified parent sets

        Examples
        --------
        >>> results = pcmci.run(
        ...     tau_max=3,
        ...     pc_alpha=0.05,
        ...     alpha_level=0.01,
        ...     fdr_method='fdr_bh'
        ... )
        >>> print(results.summary())

        Raises
        ------
        ValueError
            If tau_max < tau_min or other invalid parameters.

        Notes
        -----
        The algorithm has two phases:

        1. **PC1 Phase**: For each target j, test potential parents
           (i, -tau) and remove those conditionally independent of j
           given an increasing conditioning set.

        2. **MCI Phase**: For each remaining link (i, -tau) -> j,
           test X_i(t-tau) ⊥ X_j(t) | Parents(X_i) ∪ Parents(X_j).

        See Also
        --------
        run_pc_stable : Run only the PC condition selection phase.
        run_mci : Run only the MCI phase (requires prior PC results).
        """
        if tau_max < tau_min:
            raise ValueError(f"tau_max ({tau_max}) must be >= tau_min ({tau_min})")
        if tau_min < 0:
            raise ValueError(f"tau_min must be non-negative, got {tau_min}")

        # Precompute lagged data to avoid repeated construction
        self.datahandler.precompute_lagged_data(tau_max)

        # Initialize result matrices
        self._val_matrix = jnp.zeros((self.N, self.N, tau_max + 1))
        self._pval_matrix = jnp.ones((self.N, self.N, tau_max + 1))

        # Phase 1: PC condition selection
        if self.verbosity >= 1:
            print(f"\n{'='*60}")
            print("PCMCI: Phase 1 - PC Condition Selection")
            print(f"{'='*60}")

        self._parents = self.run_pc_stable(
            tau_max=tau_max,
            tau_min=tau_min,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
            max_subsets=max_subsets,
        )

        # Clear cache between phases to free memory
        self.datahandler.clear_cache()

        # Phase 2: MCI tests
        if self.verbosity >= 1:
            print(f"\n{'='*60}")
            print("PCMCI: Phase 2 - MCI Tests")
            print(f"{'='*60}")

        # Use batch MCI by default for GPU/TPU acceleration
        # Only falls back to sequential if batch test not available
        if hasattr(self.test, 'run_batch'):
            val_matrix, pval_matrix = self.run_batch_mci(
                tau_max=tau_max,
                tau_min=tau_min,
                parents=self._parents,
                max_conds_py=max_conds_py,
                max_conds_px=max_conds_px,
            )
        else:
            val_matrix, pval_matrix = self.run_mci(
                tau_max=tau_max,
                tau_min=tau_min,
                parents=self._parents,
                max_conds_py=max_conds_py,
                max_conds_px=max_conds_px,
            )

        # Store results
        self._val_matrix = val_matrix
        self._pval_matrix = pval_matrix

        # Create results object
        results = PCMCIResults(
            val_matrix=val_matrix,
            pval_matrix=pval_matrix,
            var_names=self.var_names,
            alpha_level=alpha_level,
            fdr_method=fdr_method,
            test_name=self.test.name,
            tau_max=tau_max,
            tau_min=tau_min,
        )

        if self.verbosity >= 1:
            print(f"\n{results.summary()}")

        return results

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def _get_active_tests_vectorized(
        self,
        snapshot_mask: jax.Array,
        active_links: jax.Array,
        valid_mask: jax.Array,
        cond_dim: int,
        max_subsets: int,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        Generate test specifications for all active links in parallel.
        
        Returns
        -------
        Tuple of (i_arr, j_arr, tau_arr, subsets_arr)
        """
        return jnp.empty((0, )), jnp.empty((0, )), jnp.empty((0, )), jnp.empty((0, ))

    def run_pc_stable(
        self,
        tau_max: int = 1,
        tau_min: int = 1,
        pc_alpha: Optional[float] = 0.05,
        max_conds_dim: Optional[int] = None,
        max_subsets: int = 100,
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """
        Run the PC-stable condition selection algorithm (JIT-compiled internals).
        """
        # Initialization
        parents_mask = jnp.ones((self.N, self.N, tau_max + 1), dtype=bool)
        # Remove self-loops at lag 0
        parents_mask = parents_mask.at[jnp.arange(self.N), jnp.arange(self.N), 0].set(False)
        # Apply tau_min
        if tau_min > 0:
            parents_mask = parents_mask.at[:, :, :tau_min].set(False)
        
        if pc_alpha is None:
            return self._mask_to_dict(parents_mask)

        max_dim = max_conds_dim if max_conds_dim is not None else self.N * tau_max
        max_cond_dim_limit = max_dim
        iterator = range(max_dim + 1)
        if self.verbosity >= 1:
            iterator = tqdm(iterator, desc="PC iterations", leave=False)

        # Extract data values for JIT
        data_values = self.datahandler.values
        
        # Check T_eff
        T_full = data_values.shape[0]
        if T_full <= tau_max:
             raise ValueError("Data length must be greater than tau_max")

        key = jax.random.PRNGKey(42)
        
        # Pre-compute grid indices
        # Pre-compute grid indices
        i_idx, j_idx, tau_idx = jnp.meshgrid(
            jnp.arange(self.N), jnp.arange(self.N), jnp.arange(tau_max + 1), indexing='ij'
        )
        i_flat = i_idx.reshape(-1)
        j_flat = j_idx.reshape(-1)
        tau_flat = tau_idx.reshape(-1)
        
        # Prepare batched inputs
        n_links = i_flat.shape[0]
        # Use conservative batch size to avoid OOM - moderate improvement from 16 to 64
        batch_size = 64
        n_batches = (n_links + batch_size - 1) // batch_size
        n_padded = n_batches * batch_size
        
        # Pad indices
        pad_len = n_padded - n_links
        i_padded = jnp.pad(i_flat, (0, pad_len), constant_values=0)
        j_padded = jnp.pad(j_flat, (0, pad_len), constant_values=0)
        tau_padded = jnp.pad(tau_flat, (0, pad_len), constant_values=0)
        
        # Reshape for scan/map: (n_batches, batch_size)
        i_batched = i_padded.reshape(n_batches, batch_size)
        j_batched = j_padded.reshape(n_batches, batch_size)
        tau_batched = tau_padded.reshape(n_batches, batch_size)
        
        # Use a single key per iteration, split inside the JIT function
        key, base_key = jax.random.split(key)

        for cond_dim in iterator:
            # Check convergence
            degrees = jnp.sum(parents_mask, axis=(0, 2))
            max_degree = jnp.max(degrees)
            if cond_dim > 0 and max_degree < cond_dim:
                break
            
            # Run scanned kernel
            pvals_batched = self._run_pc_scanned(
                data_values, parents_mask, cond_dim,
                i_batched, j_batched, tau_batched, base_key,
                max_subsets, pc_alpha, tau_max, max_cond_dim_limit
            )
            
            pvals_flat = pvals_batched.reshape(-1)[:n_links]
            pvals_grid = pvals_flat.reshape(self.N, self.N, tau_max + 1)
            
            should_remove = pvals_grid > pc_alpha
            parents_mask = jnp.logical_and(parents_mask, jnp.logical_not(should_remove))

        if self.verbosity >= 1:
            total_parents = jnp.sum(parents_mask).item()
            print(f"JIT PC phase complete: {total_parents} total parent links")

        return self._mask_to_dict(parents_mask)

    @partial(jax.jit, static_argnums=(0, 8, 9, 10, 11))
    def _run_pc_scanned(
        self,
        data_values: jax.Array,
        mask: jax.Array,
        cond_dim: Union[int, jax.Array],
        i_batched: jax.Array,
        j_batched: jax.Array,
        tau_batched: jax.Array,
        base_key: jax.Array,
        max_subsets: int,
        pc_alpha: float,
        tau_max: int,
        max_cond_dim_limit: int,
    ) -> jax.Array:
        """
        Run batches of PC tests using lax.map to avoid OOM.
        """
        # Generate keys for each batch
        n_batches = i_batched.shape[0]
        batch_size = i_batched.shape[1]
        batch_keys = jax.random.split(base_key, n_batches)
        
        def body_fun(args):
            b_i, b_j, b_tau, b_key = args
            # Split the batch key into per-element keys
            elem_keys = jax.random.split(b_key, batch_size)
            return self._pc_batch_kernel(
                data_values, mask, cond_dim,
                b_i, b_j, b_tau, elem_keys,
                max_subsets, pc_alpha, tau_max, max_cond_dim_limit
            )
        
        # lax.map over the batch dimension (axis 0 of inputs)
        xs = (i_batched, j_batched, tau_batched, batch_keys)
        return jax.lax.map(body_fun, xs)

    def _pc_batch_kernel(
        self,
        data_values: jax.Array,
        mask: jax.Array,
        cond_dim: Union[int, jax.Array],
        i_flat: jax.Array,
        j_flat: jax.Array,
        tau_flat: jax.Array,
        keys: jax.Array,
        max_subsets: int,
        pc_alpha: float,
        tau_max: int,
        max_cond_dim_limit: int,
    ) -> jax.Array:
        """Run PC tests for a batch of links with optimized data access."""
        
        # Pre-compute commonly used values
        eff_T = data_values.shape[0] - tau_max
        
        # Define the per-link test function
        def test_one_link(i, j, tau, key_in):
            is_active = mask[i, j, tau]
            
            # Parents of target j
            target_parents = mask[:, j, :] 
            tp_flat = target_parents.reshape(-1)
            
            # Remove (i, tau) itself
            curr_flat_idx = i * (tau_max + 1) + tau
            tp_flat = tp_flat.at[curr_flat_idx].set(False)
            
            n_parents = jnp.sum(tp_flat)
            
            can_test = jnp.logical_and(is_active, n_parents >= cond_dim)
            
            def perform_test(key_in):
                p = tp_flat.astype(jnp.float32)
                sub_keys = jax.random.split(key_in, max_subsets)
                
                # Optimized: Use static limit for top_k
                def get_subset(sk):
                    g = -jnp.log(-jnp.log(jax.random.uniform(sk, p.shape) + 1e-20))
                    sc = jnp.where(tp_flat, g, -1e9)
                    # Use static limit for top_k
                    _, idxs = jax.lax.top_k(sc, max_cond_dim_limit)
                    c_lags = idxs % (tau_max + 1)
                    c_vars = idxs // (tau_max + 1)
                    return c_vars, c_lags

                c_vars_all, c_lags_all = jax.vmap(get_subset)(sub_keys)
                
                # Pre-compute X and Y once
                start_x = tau_max - tau
                X_vals = jax.lax.dynamic_slice(data_values, (start_x, i), (eff_T, 1)).squeeze(1)
                X_rep = jnp.broadcast_to(X_vals, (max_subsets, eff_T))
                
                start_y = tau_max 
                Y_vals = jax.lax.dynamic_slice(data_values, (start_y, j), (eff_T, 1)).squeeze(1)
                Y_rep = jnp.broadcast_to(Y_vals, (max_subsets, eff_T))
                
                def get_Z_matrix(c_vars, c_lags):
                    def fetch_col(v, l):
                        s = tau_max - l
                        return jax.lax.dynamic_slice(data_values, (s, v), (eff_T, 1)).squeeze(1)
                    z_mat = jax.vmap(fetch_col)(c_vars, c_lags).T 
                    return z_mat
                
                Z_rep = jax.vmap(get_Z_matrix)(c_vars_all, c_lags_all)
                
                # Use masking (cond_dim is traced, can't use in slice shapes)
                limit_indices = jnp.arange(max_cond_dim_limit)
                col_mask = limit_indices < cond_dim
                Z_masked = Z_rep * col_mask.reshape(1, 1, -1)
                
                _, p_vals = self.test.run_batch(
                    X_rep, Y_rep, Z_masked, 
                    alpha=pc_alpha, 
                    n_conditions=cond_dim
                )
                
                return jnp.max(p_vals)

            return jax.lax.cond(
                can_test,
                perform_test,
                lambda k: jnp.float32(0.0) if get_config().dtype == jnp.float32 else 0.0,
                key_in
            )

        # Use keys directly - they are already (batch_size, 2) from split
        all_pvals = jax.vmap(test_one_link)(i_flat, j_flat, tau_flat, keys)
        return all_pvals

    def _mask_to_dict(self, mask: jax.Array) -> Dict[int, Set[Tuple[int, int]]]:
        """Convert boolean mask to dictionary of parent sets."""
        parents = {}
        # Ensure mask is on CPU for efficient iteration
        mask_np = np.array(mask)
        
        # Iterate over targets
        for j in self.selected_variables:
            parents[j] = set()
            # Find True entries for this target
            # mask[:, j, :]
            src_indices, lag_indices = np.where(mask_np[:, j, :])
            
            for src, lag in zip(src_indices, lag_indices):
                # Parents are stored as (i, -tau)
                parents[j].add((int(src), -int(lag)))
                
        return parents

    def _test_with_conditioning_subsets(
        self,
        i: int,
        j: int,
        tau: int,
        other_parents: List[Tuple[int, int]],
        cond_dim: int,
        pc_alpha: float,
        max_subsets: int = 100,
    ) -> bool:
        """
        Test if (i, -tau) is independent of j given subsets of other parents.

        Returns True if ANY subset leads to independence (conservative).
        
        Note: other_parents contains tuples of (var, -lag) where lag is stored
        as negative. We convert to positive lag for get_variable_pair_data.
        
        Parameters
        ----------
        max_subsets : int
            Maximum number of conditioning subsets to test. For large parent
            sets, randomly samples subsets instead of testing all C(n,k).
        """
        if cond_dim == 0:
            # Unconditional test
            X, Y, Z = self.datahandler.get_variable_pair_data(i, j, tau, None)
            result = self.test.run(X, Y, None, alpha=pc_alpha)
            return not result.significant

        subsets_to_test = self._sample_condition_subsets(
            other_parents,
            cond_dim,
            max_subsets,
            seed=i * 1000 + j * 100 + tau,
        )

        # Group subsets by the max lag in their conditioning set for batching
        # (same max lag = same data length = can batch)
        if hasattr(self.test, 'run_batch') and len(subsets_to_test) > 1:
            subsets_by_max_lag: Dict[int, List] = {}
            for subset in subsets_to_test:
                max_lag_in_subset = max(-neg_lag for _, neg_lag in subset)
                effective_max_lag = max(tau, max_lag_in_subset)
                if effective_max_lag not in subsets_by_max_lag:
                    subsets_by_max_lag[effective_max_lag] = []
                subsets_by_max_lag[effective_max_lag].append(subset)

            # Process batches
            for effective_max_lag, subsets_group in subsets_by_max_lag.items():
                batch_size = len(subsets_group)
                
                # Prepare batch data
                X, Y, Z_list = [], [], []
                
                for subset in subsets_group:
                    # Convert to required format
                    cond_vars, cond_lags = [], []
                    if len(subset) > 0:
                         cond_vars = [s[0] for s in subset]
                         cond_lags = [-s[1] for s in subset]
                    
                    data_pair = self.datahandler.get_variable_pair_data(
                        i, j, tau, cond_vars, cond_lags, max_lag=tau_max
                    )
                    X.append(data_pair[0])
                    Y.append(data_pair[1])
                    Z_list.append(data_pair[2])
                    
                # Stack
                X_batch = jnp.array(X)
                Y_batch = jnp.array(Y) 
                
                # Z might be irregular if subset sizes differed, but here they are fixed size k
                # If Z is None (cond_dim=0), handled separately above.
                Z_batch = jnp.array(Z_list)
                
                results = self.test.run_batch(X_batch, Y_batch, Z_batch, alpha=pc_alpha)
                # results.significant is a boolean array
                if jnp.any(jnp.logical_not(results.significant)):
                     return True

            return False

        # Sequential fallback
        for subset in subsets_to_test:
            cond_vars = [s[0] for s in subset]
            cond_lags = [-s[1] for s in subset]
            
            X, Y, Z = self.datahandler.get_variable_pair_data(
                i, j, tau, cond_vars, cond_lags, max_lag=tau_max
            )
            
            result = self.test.run(X, Y, Z, alpha=pc_alpha)
            if not result.significant:
                return True
                
        return False

    def run_mci(
        self,
        tau_max: int,
        tau_min: int,
        parents: Dict[int, Set[Tuple[int, int]]],
        max_conds_py: Optional[int] = None,
        max_conds_px: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Run the MCI phase (Momentary Conditional Independence).
        """
        p_val_matrix = jnp.ones((self.N, self.N, tau_max + 1))
        val_matrix = jnp.zeros((self.N, self.N, tau_max + 1))

        # Iterate over all potential links (i, -tau -> j)
        for j in self.selected_variables:
            parents_j = list(parents[j])
            
            for i in range(self.N):
                for tau in range(tau_min, tau_max + 1):
                    if i == j and tau == 0:
                        continue
                        
                    # MCI condition: Parents(j) U Parents(i, lagged)
                    # Exclude the link itself (i, -tau) from parents of j if present
                    cond_set = set(parents_j)
                    if (i, -tau) in cond_set:
                        cond_set.remove((i, -tau))
                        
                    # Add parents of i, lagged by tau
                    parents_i = list(parents[i])
                    for p_var, p_lag in parents_i:
                        new_lag = p_lag - tau
                        # Filter out future parents (optional, depends on definition)
                        if new_lag <= 0:
                            cond_set.add((p_var, new_lag))
                            
                    # Limit conditioning set size if requested
                    # prioritizing parents of j (Y) or i (X)? 
                    # Standard PCMCI uses heuristics or just takes all.
                    cond_list = list(cond_set)
                    
                    # Run test
                    # Needs conversion of lags to positive
                    cond_vars = [c[0] for c in cond_list]
                    cond_lags = [-c[1] for c in cond_list]
                    
                    X, Y, Z = self.datahandler.get_variable_pair_data(
                        i, j, tau, cond_vars, cond_lags, max_lag=tau_max
                    )
                    
                    result = self.test.run(X, Y, Z)
                    
                    val_matrix = val_matrix.at[i, j, tau].set(result.statistic)
                    p_val_matrix = p_val_matrix.at[i, j, tau].set(result.pvalue)


        return val_matrix, p_val_matrix
        
    def _get_mci_conditions(
        self,
        i: int,
        j: int,
        tau: int,
        parents: Dict[int, Set[Tuple[int, int]]],
        max_conds_py: Optional[int],
        max_conds_px: Optional[int],
    ) -> List[Tuple[int, int]]:
        """
        Get the conditioning set for MCI test of (i, -tau) -> j.

        Includes:
        1. Parents of j (excluding the link being tested)
        2. Parents of i (shifted by tau time steps)
        
        Returns list of (variable_index, positive_lag) tuples.
        """
        conditions = []

        # Parents of Y (target j), excluding (i, -tau)
        if j in parents and (max_conds_py is None or max_conds_py > 0):
            for var, neg_lag in parents[j]:
                # Skip the link being tested
                if var == i and neg_lag == -tau:
                    continue
                # Convert negative lag to positive
                pos_lag = -neg_lag
                conditions.append((var, pos_lag))
                # Early exit if we've reached limit
                if max_conds_py is not None and len(conditions) >= max_conds_py:
                    break

        # Parents of X (source i), shifted by tau
        n_from_py = len(conditions)
        if i in parents and (max_conds_px is None or max_conds_px > 0):
            for var, neg_lag in parents[i]:
                # Shift: if (k, -tau') is parent of i at time t,
                # then (k, -(tau' + tau)) is parent of i at time t - tau
                # Convert to positive lag for data handler
                pos_lag = -neg_lag + tau
                conditions.append((var, pos_lag))
                # Early exit if we've reached limit
                if max_conds_px is not None and len(conditions) >= n_from_py + max_conds_px:
                    break

        # Remove duplicates and ensure target at lag 0 is not included
        unique_conditions = []
        seen = set()
        for var, lag in conditions:
            if (var, lag) not in seen and not (var == j and lag == 0):
                seen.add((var, lag))
                unique_conditions.append((var, lag))

        return unique_conditions

    def run_batch_mci(
        self,
        tau_max: int,
        tau_min: int,
        parents: Dict[int, Set[Tuple[int, int]]],
        max_conds_py: Optional[int] = None,
        max_conds_px: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Run MCI phase using batched operations for efficiency.
        """
        # We collect all tests to be run
        test_specs = []
        
        for j in self.selected_variables:
            parents_j = list(parents.get(j, set()))
            
            for i in range(self.N):
                for tau in range(tau_min, tau_max + 1):
                    if i == j and tau == 0:
                        continue
                        
                    # Construct conditioning set
                    cond_set = set(parents_j)
                    if (i, -tau) in cond_set:
                        cond_set.remove((i, -tau))
                        
                    for p_var, p_lag in parents.get(i, set()):
                        new_lag = p_lag - tau
                        if new_lag <= 0:
                            cond_set.add((p_var, new_lag))
                            
                    cond_list = sorted(list(cond_set)) # Sort for stability
                    
                    # Apply max_conds limits
                    if max_conds_py is not None or max_conds_px is not None:
                        # Separate into parents of Y and parents of X
                        py_conds = [(v, l) for v, l in cond_list if (v, l) in parents_j or (v, l-tau) in parents_j]
                        px_conds = [(v, l) for v, l in cond_list if (v, l) not in py_conds]
                        
                        # Apply limits
                        if max_conds_py is not None:
                            py_conds = py_conds[:max_conds_py]
                        if max_conds_px is not None:
                            px_conds = px_conds[:max_conds_px]
                            
                        cond_list = sorted(py_conds + px_conds)
                    
                    test_specs.append({
                        'i': i, 'j': j, 'tau': tau,
                        'conds': cond_list
                    })

                    
        if not test_specs:
            return jnp.zeros((self.N, self.N, tau_max + 1)), jnp.ones((self.N, self.N, tau_max + 1))
            
        # Group by conditioning set size for efficient batching
        specs_by_dim = {}
        for spec in test_specs:
            dim = len(spec['conds'])
            if dim not in specs_by_dim:
                specs_by_dim[dim] = []
            specs_by_dim[dim].append(spec)
            
        final_val = jnp.zeros((self.N, self.N, tau_max + 1))
        # Initialize p-values to 1.0 (not significant)
        final_pval = jnp.ones((self.N, self.N, tau_max + 1))
        
        # Process each group
        if self.verbosity >= 1:
            pbar = tqdm(total=len(test_specs), desc="MCI Tests")
            
        for dim, specs in specs_by_dim.items():
            # Create batches - use conservative batch size to avoid OOM
            batch_size = 2048  # Reasonable for most GPUs
            
            for k in range(0, len(specs), batch_size):
                batch = specs[k:k+batch_size]
                
                i_s =  [b['i'] for b in batch]
                j_s =  [b['j'] for b in batch]
                tau_s = [b['tau'] for b in batch]
                
                cond_vars = []
                cond_lags = []
                
                if dim > 0:
                    for b in batch:
                        cv = [c[0] for c in b['conds']]
                        cl = [-c[1] for c in b['conds']]
                        cond_vars.append(cv)
                        cond_lags.append(cl)
                
                # Get batch data
                X_b, Y_b, Z_b = self.datahandler.get_variable_pair_batch(
                    jnp.array(i_s), jnp.array(j_s), jnp.array(tau_s),
                    jnp.array(cond_vars) if dim > 0 else None,
                    jnp.array(cond_lags) if dim > 0 else None,
                    max_lag=tau_max
                )
                
                statistics, pvals = self.test.run_batch(X_b, Y_b, Z_b)
                
                # Vectorized scatter back to matrix using advanced indexing
                # Convert lists to arrays for indexing
                i_arr = jnp.array(i_s)
                j_arr = jnp.array(j_s)
                tau_arr = jnp.array(tau_s)
                
                # Use vectorized .at[] operation with tuple indexing
                final_val = final_val.at[i_arr, j_arr, tau_arr].set(statistics)
                final_pval = final_pval.at[i_arr, j_arr, tau_arr].set(pvals)
                    
                if self.verbosity >= 1:
                    pbar.update(len(batch))
                    
        if self.verbosity >= 1:
            pbar.close()
            
        return final_val, final_pval

