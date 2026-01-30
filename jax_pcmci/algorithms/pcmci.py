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
>>> import time
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
import time
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
        
        # Profiling timers
        self.timers: Dict[str, float] = {}

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
        max_subsets: int = 10,
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

        # Initialize timers
        self.timers = {}
        
        # Phase 1: PC condition selection
        if self.verbosity >= 1:
            print(f"\n{'='*60}")
            print("PCMCI: Phase 1 - PC Condition Selection")
            print(f"{'='*60}")
        
        t0 = time.perf_counter()
        self._parents = self.run_pc_stable(
            tau_max=tau_max,
            tau_min=tau_min,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
            max_subsets=max_subsets,
        )
        self.timers['skeleton_discovery'] = time.perf_counter() - t0

        # Clear cache between phases to free memory
        self.datahandler.clear_cache()

        # Phase 2: MCI tests
        if self.verbosity >= 1:
            print(f"\n{'='*60}")
            print("PCMCI: Phase 2 - MCI Tests")
            print(f"{'='*60}")

        t0 = time.perf_counter()
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
        self.timers['mci_test'] = time.perf_counter() - t0

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

    @partial(jax.jit, static_argnums=(0, 8, 10, 11), donate_argnums=(1, 2))
    def _run_pc_loop(
        self,
        parents_mask: jax.Array,
        sepsets_mask: jax.Array,
        data_values: jax.Array,
        i_batched: jax.Array,
        j_batched: jax.Array,
        tau_batched: jax.Array,
        key: jax.Array,
        max_subsets: int,
        pc_alpha: float,
        tau_max: int,
        max_cond_dim_limit: int,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Execute the PC algorithm loop using jax.lax.while_loop for end-to-end JIT.
        Returns (final_parents_mask, final_sepsets_mask).
        """
        
        # State: (parents_mask, sepsets_mask, cond_dim, key)
        # Note: sepsets_mask is dense (N, N, tau_max+1, N, tau_max+1)
        init_state = (parents_mask, sepsets_mask, 0, key)

        def cond_fun(state):
            mask, _, cond_dim, _ = state
            
            # Stop if cond_dim exceeds limit
            is_within_limit = cond_dim <= max_cond_dim_limit
            
            # Convergence check: stop if max_degree < cond_dim
            degrees = jnp.sum(mask, axis=(0, 2))
            max_degree = jnp.max(degrees)
            has_enough_neighbors = jnp.logical_or(cond_dim == 0, max_degree >= cond_dim)
            
            return jnp.logical_and(is_within_limit, has_enough_neighbors)

        def body_fun(state):
            mask, sepsets, cond_dim, k = state
            
            # Split key for this iteration
            step_key, next_key = jax.random.split(k)
            
            # Run the batch kernel
            pvals_batched, cond_masks_batched = self._run_pc_scanned(
                data_values, mask, cond_dim,
                i_batched, j_batched, tau_batched, step_key,
                max_subsets, pc_alpha, tau_max, max_cond_dim_limit
            )
            
            # Reshape results
            full_pvals = pvals_batched.reshape(-1) # Padded
            # reshape cond_masks: (n_batches, batch_size, N, tau_max+1) -> (n_padded, N, tau_max+1)
            full_cond_masks = cond_masks_batched.reshape(-1, self.N, tau_max + 1)
            
            full_i = i_batched.reshape(-1)
            full_j = j_batched.reshape(-1)
            full_tau = tau_batched.reshape(-1)
            
            should_remove = full_pvals > pc_alpha
            
            # Update Parents Mask (REMOVE edges)
            # mask[i, j, tau] &= !should_remove
            mask = mask.at[full_i, full_j, full_tau].min(jnp.logical_not(should_remove))
            
            # Update Sepsets Mask (ADD sepsets for removed edges)
            # For removed edges, we store the full_cond_masks.
            # Use max(logical_or) to accumulate.
            # Only where should_remove is True.
            
            # Broadcast should_remove for masking: (n_padded,) -> (n_padded, 1, 1)
            update_mask = should_remove[:, None, None]
            masked_conds = full_cond_masks & update_mask
            
            # sepsets[i, j, tau] = masked_conds
            # Note: since edge is removed, it won't be tested again, so we won't overwrite with a larger set.
            sepsets = sepsets.at[full_i, full_j, full_tau].max(masked_conds)
            
            return (mask, sepsets, cond_dim + 1, next_key)

        final_mask, final_sepsets, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return final_mask, final_sepsets

    def run_pc_stable(
        self,
        tau_max: int = 1,
        tau_min: int = 1,
        pc_alpha: Optional[float] = 0.05,
        max_conds_dim: Optional[int] = None,
        max_subsets: int = 10,
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

        # Extract data values for JIT
        data_values = self.datahandler.values
        
        # Check T_eff
        T_full = data_values.shape[0]
        if T_full <= tau_max:
             raise ValueError("Data length must be greater than tau_max")

        key = jax.random.PRNGKey(42)
        
        # Pre-compute grid indices
        i_idx, j_idx, tau_idx = jnp.meshgrid(
            jnp.arange(self.N), jnp.arange(self.N), jnp.arange(tau_max + 1), indexing='ij'
        )
        i_flat = i_idx.reshape(-1)
        j_flat = j_idx.reshape(-1)
        tau_flat = tau_idx.reshape(-1)
        
        # Prepare batched inputs
        n_links = i_flat.shape[0]
        # Use dynamic batch size based on available memory
        mem_batch_size = self._get_effective_batch_size(n_samples=self.T, n_conditions=self.N * tau_max)
        if mem_batch_size is None or mem_batch_size <= 0:
            mem_batch_size = 64
            
        # Avoid excessive padding for small problems
        if n_links < mem_batch_size:
            batch_size = max(1, n_links)
        else:
            batch_size = mem_batch_size

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
        
        key, loop_key = jax.random.split(key)

        if self.verbosity >= 1:
            print("Starting JIT-compiled PC Phase...")

        # Initialize separate sets mask (N, N, tau_max+1, N, tau_max+1)
        # But here we just return (parents_mask, sepsets_mask)
        # Actually initializing such a huge tensor might be costly? (10,10,6,10,6) is small.
        # But if N is large (e.g. 100) -> 100^2*6*100*6 is big.
        # For now assume N is small as per benchmarks (N=10).
        sepsets_mask = jnp.zeros(
            (self.N, self.N, tau_max + 1, self.N, tau_max + 1), 
            dtype=jnp.bool_
        )

        parents_mask, sepsets_mask = self._run_pc_loop(
            parents_mask,
            sepsets_mask,
            data_values,
            i_batched,
            j_batched,
            tau_batched,
            loop_key,
            max_subsets,
            pc_alpha,
            tau_max,
            max_cond_dim_limit
        )
        # Block until ready to ensure timing captures computation
        parents_mask.block_until_ready()

        if self.verbosity >= 1:
            # Need to pull mask to CPU for sum
            total_parents = jnp.sum(parents_mask).item()
            print(f"JIT PC phase complete: {total_parents} total parent links")

        return self._mask_to_dict(parents_mask)

    @partial(jax.jit, static_argnums=(0, 8, 10, 11))
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
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Run batches of PC tests using lax.map to avoid OOM.
        Uses bucketed kernels (lax.switch) to optimize for cond_dim.
        Returns (all_pvals, all_cond_masks).
        """
        # Generate keys for each batch
        n_batches = i_batched.shape[0]
        batch_size = i_batched.shape[1]
        batch_keys = jax.random.split(base_key, n_batches)

        # Define buckets for max_cond_dim_limit optimization
        # Reduced from 7 buckets to 3 for faster compilation while maintaining performance
        possible_limits = [0, 4, 32]
        # Filter to only use buckets <= max_cond_dim_limit
        buckets = [l for l in possible_limits if l < max_cond_dim_limit]
        buckets.append(max_cond_dim_limit)
        buckets = sorted(list(set(buckets))) # unique and sorted
        
        # Create branches for switch
        branches = []
        for limit in buckets:
            # Capture limit in closure (partial-like)
            def make_branch(l_val):
                def branch_impl(args):
                    b_i, b_j, b_tau, b_key = args
                    elem_keys = jax.random.split(b_key, batch_size)
                    # Returns (pvals, cond_masks)
                    return self._pc_batch_kernel(
                        data_values, mask, cond_dim,
                        b_i, b_j, b_tau, elem_keys,
                        max_subsets, pc_alpha, tau_max, l_val
                    )
                return branch_impl
            branches.append(make_branch(limit))
            
        # For 3 buckets, use static switch is still efficient
        # Could use jnp.where cascade but switch is clearer for 3-4 branches
        bucket_arr = jnp.array(buckets)
        idx = jnp.argmax(bucket_arr >= cond_dim)
        
        def body_fun(args):
            # Static dispatch to appropriate bucket kernel
            return jax.lax.switch(idx, branches, args)
    
        # lax.map over batch dimension with buffer donation for efficiency
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
    ) -> Tuple[jax.Array, jax.Array]:
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
                
                # Find best p-value and corresponding sepset
                max_idx = jnp.argmax(p_vals)
                max_pval = p_vals[max_idx]
                
                # Extract winning sepset info
                winning_c_vars = c_vars_all[max_idx]
                winning_c_lags = c_lags_all[max_idx]
                
                # Create mask (N, tau_max+1)
                cond_mask = jnp.zeros((self.N, tau_max + 1), dtype=jnp.bool_)
                # Use col_mask to only set valid entries to True
                cond_mask = cond_mask.at[winning_c_vars, winning_c_lags].set(col_mask)
                
                return max_pval, cond_mask

            empty_mask = jnp.zeros((self.N, tau_max + 1), dtype=jnp.bool_)
            return jax.lax.cond(
                can_test,
                perform_test,
                lambda k: (
                    jnp.float32(0.0) if get_config().dtype == jnp.float32 else 0.0,
                    empty_mask
                ),
                key_in
            )

        # Use keys directly - they are already (batch_size, 2) from split
        all_pvals, all_cond_masks = jax.vmap(test_one_link)(i_flat, j_flat, tau_flat, keys)
        return all_pvals, all_cond_masks

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
        max_subsets: int = 10,
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
        """
        Run MCI phase using batched operations for efficiency.
        """
        # Collect test specifications
        # Convert parents to a more accessible structure or just iterate once to build arrays
        
        # Lists to build arrays
        i_list, j_list, tau_list = [], [], []
        cond_vars_list, cond_lags_list = [], []
        n_conds_list = []
        
        # First pass: collect all tests and find max conditioning set size
        max_cond_size = 0
        
        # We can iterate over tests in Python since it's just building lists (fast enough compared to testing)
        # But for truly large graphs we might want to JIT this too. detailed in plan: we stick to Python for list construction
        # as the bottleneck is the `run_batch` calls inside the loop, not the list building itself.
        
        if self.verbosity >= 1:
            pbar = tqdm(total=self.N * self.N * (tau_max - tau_min + 1), desc="MCI Prep")

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
                            
                    cond_list = sorted(list(cond_set))
                    
                    # Apply limits if needed (same logic as before)
                    if max_conds_py is not None or max_conds_px is not None:
                        py_conds = [(v, l) for v, l in cond_list if (v, l) in parents_j or (v, l-tau) in parents_j]
                        px_conds = [(v, l) for v, l in cond_list if (v, l) not in py_conds]
                        if max_conds_py is not None: py_conds = py_conds[:max_conds_py]
                        if max_conds_px is not None: px_conds = px_conds[:max_conds_px]
                        cond_list = sorted(py_conds + px_conds)
                    
                    n_c = len(cond_list)
                    max_cond_size = max(max_cond_size, n_c)
                    
                    i_list.append(i)
                    j_list.append(j)
                    tau_list.append(tau)
                    
                    # Store conditions temporarily
                    c_vars = [c[0] for c in cond_list]
                    c_lags = [-c[1] for c in cond_list]
                    cond_vars_list.append(c_vars)
                    cond_lags_list.append(c_lags)
                    n_conds_list.append(n_c)
                    
        if self.verbosity >= 1:
            pbar.close()
            
        if not i_list:
             return jnp.zeros((self.N, self.N, tau_max + 1)), jnp.ones((self.N, self.N, tau_max + 1))

        if self.verbosity >= 1:
            pbar.close()
            
        if not i_list:
             return jnp.zeros((self.N, self.N, tau_max + 1)), jnp.ones((self.N, self.N, tau_max + 1))

        # Group by condition size (Bucketing) to avoid padding/masking issues
        # ParCorr is sensitive to zero-padding (singular matrix with ridge regression)
        # Optimized: Use fixed bucket sizes to minimize JIT compilations
        BUCKET_SIZES = [0, 1, 2, 4, 8, 16, 32, 64]
        
        # Organize tests by exact n_c first
        tests_by_nc = {}
        for idx in range(len(i_list)):
            n_c = n_conds_list[idx]
            if n_c not in tests_by_nc:
                tests_by_nc[n_c] = {
                    'i': [], 'j': [], 'tau': [], 
                    'c_vars': [], 'c_lags': []
                }
            tests_by_nc[n_c]['i'].append(i_list[idx])
            tests_by_nc[n_c]['j'].append(j_list[idx])
            tests_by_nc[n_c]['tau'].append(tau_list[idx])
            tests_by_nc[n_c]['c_vars'].append(cond_vars_list[idx])
            tests_by_nc[n_c]['c_lags'].append(cond_lags_list[idx])

        # Assign each n_c to a bucket
        # processing_queue[bucket_size] = [list of (n_c, test_dict)]
        processing_queue = {b: [] for b in BUCKET_SIZES}
        # Also need a catch-all for huge max_conds if any (N > 64)
        MAX_BUCKET = 128
        
        for n_c, tests in tests_by_nc.items():
            # Find smallest bucket >= n_c
            chosen_bucket = None
            for b in BUCKET_SIZES:
                if b >= n_c:
                    chosen_bucket = b
                    break
            
            if chosen_bucket is None:
                # Should fit in MAX_BUCKET or largest available
                chosen_bucket = max(BUCKET_SIZES) if n_c <= max(BUCKET_SIZES) else n_c
                # If n_c is huge, we might just JIT for it (or pad to it)
                # Let's dynamically add it if needed, or use n_c itself as bucket
                # Using n_c itself is fallback behavior
                if n_c > max(BUCKET_SIZES):
                    chosen_bucket = n_c
                    if chosen_bucket not in processing_queue:
                         processing_queue[chosen_bucket] = []

            processing_queue[chosen_bucket].append((n_c, tests))

        # Initialize result matrices
        val_matrix = jnp.zeros((self.N, self.N, tau_max + 1))
        pval_matrix = jnp.ones((self.N, self.N, tau_max + 1))

        # Process each bucket

        if self.verbosity >= 1:
            print(f"Processing buckets: {[b for b, items in processing_queue.items() if items]}")

        for bucket_size, subgroups in processing_queue.items():
            if not subgroups:
                continue
                
            X_bucket = []
            Y_bucket = []
            Z_bucket = []
            i_bucket = []
            j_bucket = []
            tau_bucket = []
            
            # Collect data for all subgroups in this bucket
            for n_c, tests in subgroups:
                # Prepare indices
                i_arr = jnp.array(tests['i'], dtype=jnp.int32)
                j_arr = jnp.array(tests['j'], dtype=jnp.int32)
                tau_arr = jnp.array(tests['tau'], dtype=jnp.int32)
                
                # Fetch data
                if n_c > 0:
                    cv_arr = jnp.array(tests['c_vars'], dtype=jnp.int32)
                    cl_arr = jnp.array(tests['c_lags'], dtype=jnp.int32)
                else:
                    cv_arr = None
                    cl_arr = None
                
                X_b, Y_b, Z_b = self.datahandler.get_variable_pair_batch(
                    i_arr, j_arr, tau_arr, cv_arr, cl_arr, max_lag=tau_max
                )
                
                # Pad Z_b to bucket_size
                # Z_b is (Batch, T, n_c) -> (Batch, T, bucket_size)
                pad_width = bucket_size - n_c
                if pad_width > 0:
                    # Pad last dimension with zeros
                    # ((0,0), (0,0), (0, pad_width))
                    padding = ((0, 0), (0, 0), (0, pad_width))
                    Z_b_padded = jnp.pad(Z_b, padding, constant_values=0.0)
                elif pad_width == 0:
                    Z_b_padded = Z_b
                else:
                    # Should not happen if chosen_bucket >= n_c
                    Z_b_padded = Z_b # Fail safe
                
                # If n_c=0, Z_b might be None or specialized. 
                # get_variable_pair_batch returns None for Z if n_c=0?
                # Let's check or handle it.
                # Usually it returns array(0) or similar.
                # If Z_b is None, we need to create zeros (Batch, T, bucket_size)
                if Z_b is None and bucket_size > 0:
                     # Create zeros
                     T_eff = X_b.shape[1]
                     batch_n = X_b.shape[0]
                     Z_b_padded = jnp.zeros((batch_n, T_eff, bucket_size))
                elif Z_b is None: # bucket_size == 0
                     # Z_b is None, which stands for empty
                     pass # handled by run_batch logic for None?
                
                # Handling bucket_size=0: call run_batch with Z=None
                # But here we are collecting lists.
                # If bucket_size=0, Z_list will contain Nones? 
                # Or run_batch handles Z=None.
                
                X_bucket.append(X_b)
                Y_bucket.append(Y_b)
                if bucket_size > 0:
                    Z_bucket.append(Z_b_padded)
                
                i_bucket.append(i_arr)
                j_bucket.append(j_arr)
                tau_bucket.append(tau_arr)

            # Concatenate all subgroups
            if not X_bucket:
                continue

            X_all = jnp.concatenate(X_bucket, axis=0)
            Y_all = jnp.concatenate(Y_bucket, axis=0)
            
            if bucket_size > 0:
                Z_all = jnp.concatenate(Z_bucket, axis=0)
            else:
                Z_all = None

            i_all = jnp.concatenate(i_bucket, axis=0)
            j_all = jnp.concatenate(j_bucket, axis=0)
            tau_all = jnp.concatenate(tau_bucket, axis=0)
            
            # Run Batched Test
            # Only JIT compiled for 'bucket_size'
            statistics, pvals = self.test.run_batch(
                X_all, Y_all, Z_all, 
                n_conditions=bucket_size
            )
            
            # Scatter results
            val_matrix = val_matrix.at[i_all, j_all, tau_all].set(statistics)
            pval_matrix = pval_matrix.at[i_all, j_all, tau_all].set(pvals)

        if self.verbosity >= 1:
            pbar.close()

        return val_matrix, pval_matrix
