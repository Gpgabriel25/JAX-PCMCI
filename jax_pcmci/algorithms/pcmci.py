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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from functools import partial
from tqdm import tqdm
from itertools import combinations
import math

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
        # active_links is padded, valid_mask indicates real links
        n_links = active_links.shape[0]
        
        # Unpack active links
        i_arr = active_links[:, 0]
        j_arr = active_links[:, 1]
        tau_arr = active_links[:, 2]
        
        if cond_dim == 0:
            return i_arr, j_arr, tau_arr, jnp.zeros((n_links, 0, 2), dtype=jnp.int32)
            
        # For each link, we need to sample subsets from OTHER parents of j
        # parents_mask shape: (N, N, tau_max+1)
        
        def sample_for_link(idx):
            # Check validity
            is_valid_link = valid_mask[idx]
            
            # We must run the logic to maintain shape, but can short-circuit or mask result
            # However, JAX vmap requires same control flow. 
            # We use lax.cond to handle invalid links by returning dummy 
            # but we must ensure we don't access out of bounds or error.
            # active_links should be padded with valid indices (e.g. 0,0,0)
            
            i, j, tau = i_arr[idx], j_arr[idx], tau_arr[idx]
            
            # Get potential parents indices for target j
            # slice shape: (N, tau_max+1)
            target_parents_mask = snapshot_mask[:, j, :]
            
            # Flatten mask to sample indices
            flat_mask = target_parents_mask.reshape(-1) # Size N*(tau_max+1)
            
            # Set current parent (i, tau) to False
            current_flat_idx = i * (snapshot_mask.shape[2]) + tau
            flat_mask = flat_mask.at[current_flat_idx].set(False)
            
            # Get indices where mask is True
            potential_indices = jnp.arange(flat_mask.shape[0])
            
            # Count valid parents
            n_potential = jnp.sum(flat_mask)
            
            # If not valid link OR not enough parents, return invalid
            def get_subsets():
                # We need to sample 'max_subsets' of size 'cond_dim'
                # Strategy: Use Gumbel-Top-K or iterative sampling if exact enumeration isn't needed.
                # Here we use a simplified random choice with replacement (checking uniqueness loop is hard in pure JAX)
                # or just accept collisions for performance in this randomized approx.
                
                # To get unique subsets in JAX is tricky. 
                # We will restart the RNG seeder based on link ID to get deterministic behavior per link
                key = jax.random.PRNGKey(idx * 12345 + cond_dim)
                
                # We want to sample 'cond_dim' elements 'max_subsets' times.
                # p = flat_mask / sum(flat_mask)
                p = flat_mask.astype(jnp.float32)
                p = p / (jnp.sum(p) + 1e-10)
                
                # Sample (max_subsets, cond_dim) indices
                # Note: choice with replace=False is hard for batching if populations differ.
                # We use replace=True and maybe filter? Or just replace=False if supported?
                # jax.random.choice only supports replace=False for 1D.
                
                # Workaround: For the randomized phase, we might just pick random parents.
                # Implementing a fully vectorized unique-subset sampler is advanced.
                # Fallback: We proceed with a slightly simplified logic where we assume 
                # we can sample independent indices.
                
                # Let's map flattened indices back to (var, lag)
                keys = jax.random.split(key, max_subsets)
                
                def sample_one_subset(k):
                     return jax.random.choice(k, potential_indices, shape=(cond_dim,), p=p, replace=False)
                     
                sampled_flat_indices = jax.vmap(sample_one_subset)(keys)
                
                # Convert back to (var, lag)
                # lag = idx % (tau_max+1)
                # var = idx // (tau_max+1)
                n_lags = snapshot_mask.shape[2]
                subset_lags = sampled_flat_indices % n_lags
                subset_vars = sampled_flat_indices // n_lags
                
                # Stack to (max_subsets, cond_dim, 2)
                return jnp.stack([subset_vars, subset_lags], axis=-1).astype(jnp.int32)

            # Conditional: if valid link AND enough parents, get subsets, else zeros
            # Note: The logic "if len(potential_parents) < cond_dim" needs to be checked dynamically
            return jax.lax.cond(
                jnp.logical_and(is_valid_link, n_potential >= cond_dim),
                get_subsets,
                lambda: jnp.zeros((max_subsets, cond_dim, 2), dtype=jnp.int32) - 1 # Mark invalid
            )

        # Vmap over all active links
        subsets_all = jax.vmap(sample_for_link)(jnp.arange(n_links))
        
        return i_arr, j_arr, tau_arr, subsets_all

    def run_pc_stable(
        self,
        tau_max: int = 1,
        tau_min: int = 1,
        pc_alpha: Optional[float] = 0.05,
        max_conds_dim: Optional[int] = None,
        max_subsets: int = 100,
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """
        Run the PC-stable condition selection algorithm (Vectorized Phase 2).
        """
        # Initialization
        parents_mask = jnp.ones((self.N, self.N, tau_max + 1), dtype=bool)
        parents_mask = parents_mask.at[jnp.arange(self.N), jnp.arange(self.N), 0].set(False)
        if tau_min > 0:
            parents_mask = parents_mask.at[:, :, :tau_min].set(False)
        
        if pc_alpha is None:
            return self._mask_to_dict(parents_mask)

        max_dim = max_conds_dim if max_conds_dim is not None else self.N * tau_max
        iterator = range(max_dim + 1)
        if self.verbosity >= 1:
            iterator = tqdm(iterator, desc="PC iterations", leave=False)

        # Pre-calculate fixed size for padding
        # Max possible links is roughly len(selected) * N * (tau_max+1)
        # We use a conservative upper bound
        n_selected = len(self.selected_variables)
        max_links_fixed = n_selected * self.N * (tau_max + 1)

        for cond_dim in iterator:
            snapshot_mask = parents_mask # JAX array, no need to copy as it's immutable
            
            # Identify active links
            # (i, j, tau) indices
            active_links_indices = jnp.argwhere(snapshot_mask)
            
            # Filter for selected targets (still need this check as we iterate active links)
            # Efficiently filtering in JAX:
            # Create mask of selected targets
            target_mask = jnp.zeros(self.N, dtype=bool)
            target_mask = target_mask.at[jnp.array(list(self.selected_variables))].set(True)
            
            # active_links_indices: (K, 3) where column 1 is j
            # Keep rows where target_mask[j] is True
            j_indices = active_links_indices[:, 1]
            rows_to_keep = target_mask[j_indices]
            
            active_links_dynamic = active_links_indices[rows_to_keep]
            n_active = active_links_dynamic.shape[0]
            
            if n_active == 0:
                break
                
            # Pad active_links to fixed size to avoid recompilation
            # We act on max_links_fixed or slightly larger if needed
            # Ensure we don't exceed - handled by dynamic shape in non-padded approach,
            # but here we force padding.
            
            if n_active > max_links_fixed:
                # Should not happen given the bound logic, but for safety:
                max_links_fixed = n_active 
                
            padding_len = max_links_fixed - n_active
            
            # Create padded arrays
            # Pad with 0 (valid index) but mask out via valid_mask
            if padding_len > 0:
                active_links_padded = jnp.pad(active_links_dynamic, ((0, padding_len), (0, 0)), mode='constant', constant_values=0)
                valid_mask = jnp.concatenate([jnp.ones(n_active, dtype=bool), jnp.zeros(padding_len, dtype=bool)])
            else:
                active_links_padded = active_links_dynamic
                valid_mask = jnp.ones(n_active, dtype=bool)

            # If cond_dim > 0, check if we can stop early (no node has enough parents)
            # Calculate degree per node
            degrees = jnp.sum(snapshot_mask, axis=(0, 2)) # Shape (N,)
            max_degree = jnp.max(degrees)
            if cond_dim > 0 and max_degree < cond_dim:
                # We can technically stop here if strictly following PC, 
                # but let's just continue to be safe or break
                pass 

            # Generate tests fully vectorized
            if cond_dim == 0:
                # Simple case: 1 test per link
                # We can just process dynamic links directly here as this step is usually fast or matches padded
                # But to rely on padding consistency, let's use the padded arrays but filter before run_batch?
                # Actually run_batch for cond_dim=0 is best done on dynamic shape or masked?
                # DataHandler handles dynamic shapes fine.
                # Let's use dynamic shape for cond_dim=0 as it is only 1 iteration, no loop recompilation problem usually 
                # (cond_dim=0 is always first step)
                
                i_arr = active_links_dynamic[:, 0]
                j_arr = active_links_dynamic[:, 1]
                tau_arr = active_links_dynamic[:, 2]
                
                # Run batch
                X_b, Y_b, Z_b = self.datahandler.get_variable_pair_batch(
                    i_arr, j_arr, tau_arr, None, None, max_lag=tau_max
                )
                
                stats, pvals = self.test.run_batch(X_b, Y_b, Z_b, alpha=pc_alpha)
                
                # Check significant
                is_indep = pvals > pc_alpha
                
                # Remove links
                # indices to remove: active_links[is_indep]
                links_to_remove = active_links_dynamic[is_indep]
                if links_to_remove.shape[0] > 0:
                     parents_mask = parents_mask.at[links_to_remove[:, 0], links_to_remove[:, 1], links_to_remove[:, 2]].set(False)

            else:
                # Conditional case - Use padding to stabilize shapes
                i_arr, j_arr, tau_arr, subsets_all = self._get_active_tests_vectorized(
                    snapshot_mask, active_links_padded, valid_mask, cond_dim, max_subsets
                )
                
                # subsets_all: (max_links_fixed, max_subsets, cond_dim, 2)
                
                n_links_p, n_subs, _, _ = subsets_all.shape
                
                # Reshape for batch run
                i_flat = jnp.repeat(i_arr, n_subs)
                j_flat = jnp.repeat(j_arr, n_subs)
                tau_flat = jnp.repeat(tau_arr, n_subs)
                
                subsets_flat = subsets_all.reshape(-1, cond_dim, 2)
                
                # Filter out invalid subsets 
                # (Both masked from padding AND invalid returns from subset sampling)
                valid_tests_mask = subsets_flat[:, 0, 0] != -1
                
                if jnp.sum(valid_tests_mask) == 0:
                    continue
                    
                i_run = i_flat[valid_tests_mask]
                j_run = j_flat[valid_tests_mask]
                tau_run = tau_flat[valid_tests_mask]
                subsets_run = subsets_flat[valid_tests_mask]
                
                # Now we have a massive batch of tests.
                total_tests = i_run.shape[0]
                chunk_size = 50000 
                
                is_independent_link_padded = jnp.zeros(n_links_p, dtype=bool)
                
                # Map back to padded index
                original_indices = jnp.arange(n_links_p).repeat(n_subs)[valid_tests_mask]
                
                for k in range(0, total_tests, chunk_size):
                    end = min(k + chunk_size, total_tests)
                    
                    i_c = i_run[k:end]
                    j_c = j_run[k:end]
                    tau_c = tau_run[k:end]
                    subs_c = subsets_run[k:end]
                    
                    cond_vars = subs_c[:, :, 0]
                    cond_lags = subs_c[:, :, 1]
                    
                    # Ensure integer types
                    i_c = i_c.astype(jnp.int32)
                    j_c = j_c.astype(jnp.int32)
                    tau_c = tau_c.astype(jnp.int32)
                    cond_vars = cond_vars.astype(jnp.int32)
                    cond_lags = cond_lags.astype(jnp.int32)
                    
                    X_b, Y_b, Z_b = self.datahandler.get_variable_pair_batch(
                        i_c, j_c, tau_c, cond_vars, cond_lags, max_lag=tau_max
                    )
                    
                    # Run test
                    _, pvals = self.test.run_batch(X_b, Y_b, Z_b, alpha=pc_alpha)
                    
                    # Check independence
                    indep_c = pvals > pc_alpha
                    
                    # Update status
                    batch_orig_indices = original_indices[k:end]
                    found_indep_indices = batch_orig_indices[indep_c]
                    
                    if found_indep_indices.shape[0] > 0:
                        is_independent_link_padded = is_independent_link_padded.at[found_indep_indices].set(True)
                        
                # Identify valid links that are independent
                # Must be valid (in valid_mask) AND found independent
                final_remove_mask = jnp.logical_and(is_independent_link_padded, valid_mask)
                
                links_to_remove = active_links_padded[final_remove_mask]
                if links_to_remove.shape[0] > 0:
                     parents_mask = parents_mask.at[links_to_remove[:, 0], links_to_remove[:, 1], links_to_remove[:, 2]].set(False)

        if self.verbosity >= 1:
            total_parents = jnp.sum(parents_mask).item()
            print(f"PC phase complete: {total_parents} total parent links")

        return self._mask_to_dict(parents_mask)

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
            
            # Process each group as a batch
            for max_lag, subsets_group in subsets_by_max_lag.items():
                if len(subsets_group) == 1:
                    # Single subset - just run directly
                    subset = subsets_group[0]
                    condition_indices = [(var, -neg_lag) for var, neg_lag in subset]
                    X, Y, Z = self.datahandler.get_variable_pair_data(
                        i, j, tau, condition_indices
                    )
                    result = self.test.run(X, Y, Z, alpha=pc_alpha)
                    if not result.significant:
                        return True
                else:
                    # Batch test subsets in memory-aware chunks
                    effective_T = self.T - max_lag
                    batch_size = self._get_effective_batch_size(n_samples=effective_T, n_conditions=cond_dim)
                    if batch_size is None:
                        chunk_ranges = [(0, len(subsets_group))]
                    else:
                        chunk_ranges = [
                            (start, min(start + batch_size, len(subsets_group)))
                            for start in range(0, len(subsets_group), batch_size)
                        ]

                    for start, end in chunk_ranges:
                        # Optimized subset unpacking
                        current_subsets = subsets_group[start:end]
                        
                        # Convert to numpy array: (batch, cond_dim, 2)
                        # entries are (var, neg_lag)
                        subset_arr = np.array(current_subsets, dtype=np.int32)
                        cond_vars = jnp.array(subset_arr[:, :, 0], dtype=jnp.int32)
                        cond_lags = jnp.array(-subset_arr[:, :, 1], dtype=jnp.int32)

                        batch_len = end - start
                        i_arr = jnp.full((batch_len,), i, dtype=jnp.int32)
                        j_arr = jnp.full((batch_len,), j, dtype=jnp.int32)
                        tau_arr = jnp.full((batch_len,), tau, dtype=jnp.int32)

                        X_arr, Y_arr, Z_arr = self.datahandler.get_variable_pair_batch(
                            i_arr,
                            j_arr,
                            tau_arr,
                            cond_vars=cond_vars,
                            cond_lags=cond_lags,
                            max_lag=max_lag,
                        )
                        
                        # Run batch test
                        stats, pvals = self.test.run_batch(X_arr, Y_arr, Z_arr, alpha=pc_alpha)
                        
                        # Check if any are independent (p-value > alpha)
                        if bool(jnp.any(pvals > pc_alpha)):
                            return True
            
            return False
        
        # Fallback to sequential testing
        for subset in subsets_to_test:
            condition_indices = [(var, -neg_lag) for var, neg_lag in subset]
            X, Y, Z = self.datahandler.get_variable_pair_data(
                i, j, tau, condition_indices
            )
            result = self.test.run(X, Y, Z, alpha=pc_alpha)

            if not result.significant:
                # Found a conditioning set that makes them independent
                return True

        return False

    def run_mci(
        self,
        tau_max: int = 1,
        tau_min: int = 1,
        parents: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        max_conds_py: Optional[int] = None,
        max_conds_px: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Run the MCI (Momentary Conditional Independence) phase.

        Tests each potential causal link using momentary conditional
        independence, where the conditioning set includes parents of
        both the source and target variables.

        Parameters
        ----------
        tau_max : int, default=1
            Maximum time lag.
        tau_min : int, default=1
            Minimum time lag.
        parents : dict or None
            Parent sets from PC phase. If None, uses stored parents.
        max_conds_py : int or None
            Maximum conditions from target's parents.
        max_conds_px : int or None
            Maximum conditions from source's parents.

        Returns
        -------
        val_matrix : jax.Array
            Test statistics, shape (N, N, tau_max + 1).
        pval_matrix : jax.Array
            P-values, shape (N, N, tau_max + 1).

        Notes
        -----
        The MCI test for link (i, -tau) -> j conditions on:
        - Parents(j) minus (i, -tau): Parents of target excluding tested link
        - Parents(i, -tau): Parents of source (shifted by tau)
        """
        if parents is None:
            parents = self._parents

        val_matrix = jnp.zeros((self.N, self.N, tau_max + 1))
        pval_matrix = jnp.ones((self.N, self.N, tau_max + 1))

        # Collect all tests to run
        tests_to_run = []

        for j in self.selected_variables:
            for i in range(self.N):
                for tau in range(tau_min, tau_max + 1):
                    if tau == 0 and i == j:
                        continue
                    tests_to_run.append((i, j, tau))

        if self.verbosity >= 1:
            tests_iterator = tqdm(tests_to_run, desc="MCI tests", leave=False)
        else:
            tests_iterator = tests_to_run

        for i, j, tau in tests_iterator:
            # Build conditioning set
            cond_set = self._get_mci_conditions(
                i, j, tau, parents, max_conds_py, max_conds_px
            )

            # Get data
            if cond_set:
                X, Y, Z = self.datahandler.get_variable_pair_data(i, j, tau, cond_set)
            else:
                X, Y, Z = self.datahandler.get_variable_pair_data(i, j, tau, None)
                Z = None

            # Run test
            result = self.test.run(X, Y, Z)

            # Store results
            val_matrix = val_matrix.at[i, j, tau].set(result.statistic)
            pval_matrix = pval_matrix.at[i, j, tau].set(result.pvalue)

        return val_matrix, pval_matrix

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
        if j in parents:
            for var, neg_lag in parents[j]:
                # Skip the link being tested
                if var == i and neg_lag == -tau:
                    continue
                # Convert negative lag to positive
                pos_lag = -neg_lag
                conditions.append((var, pos_lag))
            
            if max_conds_py is not None:
                conditions = conditions[:max_conds_py]

        # Parents of X (source i), shifted by tau
        n_from_py = len(conditions)
        if i in parents:
            for var, neg_lag in parents[i]:
                # Shift: if (k, -tau') is parent of i at time t,
                # then (k, -(tau' + tau)) is parent of i at time t - tau
                # Convert to positive lag for data handler
                pos_lag = -neg_lag + tau
                conditions.append((var, pos_lag))
            
            if max_conds_px is not None:
                # Limit only the parents from X
                conditions = conditions[:n_from_py + max_conds_px]

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
        tau_max: int = 1,
        tau_min: int = 1,
        parents: Optional[Dict[int, Set[Tuple[int, int]]]] = None,
        max_conds_py: Optional[int] = None,
        max_conds_px: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Run MCI tests in parallel batches for maximum GPU utilization.

        This method groups tests by conditioning set size and runs them
        in vectorized batches, providing significant speedup on GPU/TPU.

        Parameters
        ----------
        tau_max : int, default=1
            Maximum time lag.
        tau_min : int, default=1
            Minimum time lag.
        parents : dict or None
            Parent sets from PC phase.
        max_conds_py : int or None
            Maximum conditions from target's parents.
        max_conds_px : int or None
            Maximum conditions from source's parents.

        Returns
        -------
        val_matrix : jax.Array
            Test statistics.
        pval_matrix : jax.Array
            P-values.

        Notes
        -----
        This is an optimized version of run_mci that leverages JAX's
        vmap for parallel test execution. It's particularly effective
        when running many tests with similar conditioning set sizes.
        """
        if parents is None:
            parents = self._parents

        val_matrix = jnp.zeros((self.N, self.N, tau_max + 1))
        pval_matrix = jnp.ones((self.N, self.N, tau_max + 1))

        # Group tests by (n_conditions, max_lag) for proper batching
        # Same n_cond AND same max_lag = same data shapes = can batch
        tests_by_shape: Dict[Tuple[int, int], List[Tuple]] = {}

        for j in self.selected_variables:
            for i in range(self.N):
                for tau in range(tau_min, tau_max + 1):
                    if tau == 0 and i == j:
                        continue

                    cond_set = self._get_mci_conditions(
                        i,
                        j,
                        tau,
                        parents,
                        max_conds_py,
                        max_conds_px,
                    )
                    n_cond = len(cond_set)
                    
                    # Compute max lag for this test
                    if cond_set:
                        max_cond_lag = max(lag for _, lag in cond_set)
                        effective_max_lag = max(tau, max_cond_lag)
                    else:
                        effective_max_lag = tau

                    key = (n_cond, effective_max_lag)
                    if key not in tests_by_shape:
                        tests_by_shape[key] = []
                    tests_by_shape[key].append((i, j, tau, cond_set))

        # Process each batch
        for (n_cond, max_lag), tests in tests_by_shape.items():
            if self.verbosity >= 2:
                print(f"Processing {len(tests)} tests with {n_cond} conditions, max_lag={max_lag}")

            if len(tests) == 0:
                continue

            effective_T = self.T - max_lag
            batch_size = self._get_effective_batch_size(n_samples=effective_T, n_conditions=n_cond)
            if batch_size is None:
                chunk_ranges = [(0, len(tests))]
            else:
                chunk_ranges = [
                    (start, min(start + batch_size, len(tests)))
                    for start in range(0, len(tests), batch_size)
                ]

            for start, end in chunk_ranges:
                i_list = []
                j_list = []
                tau_list = []
                cond_vars_list = [] if n_cond > 0 else None
                cond_lags_list = [] if n_cond > 0 else None

                for i, j, tau, cond_set in tests[start:end]:
                    i_list.append(i)
                    j_list.append(j)
                    tau_list.append(tau)
                    if n_cond > 0:
                        cond_vars_list.append([var for var, _ in cond_set])
                        cond_lags_list.append([lag for _, lag in cond_set])

                i_arr = jnp.asarray(i_list, dtype=jnp.int32)
                j_arr = jnp.asarray(j_list, dtype=jnp.int32)
                tau_arr = jnp.asarray(tau_list, dtype=jnp.int32)

                if n_cond > 0:
                    cond_vars = jnp.asarray(cond_vars_list, dtype=jnp.int32)
                    cond_lags = jnp.asarray(cond_lags_list, dtype=jnp.int32)
                    X_arr, Y_arr, Z_arr = self.datahandler.get_variable_pair_batch(
                        i_arr,
                        j_arr,
                        tau_arr,
                        cond_vars=cond_vars,
                        cond_lags=cond_lags,
                        max_lag=max_lag,
                    )
                else:
                    X_arr, Y_arr, Z_arr = self.datahandler.get_variable_pair_batch(
                        i_arr,
                        j_arr,
                        tau_arr,
                        max_lag=max_lag,
                    )

                # Run batch test
                stats, pvals = self.test.run_batch(X_arr, Y_arr, Z_arr)

                # Store results (vectorized scatter)
                val_matrix = val_matrix.at[i_arr, j_arr, tau_arr].set(stats)
                pval_matrix = pval_matrix.at[i_arr, j_arr, tau_arr].set(pvals)

        return val_matrix, pval_matrix

    def get_parents(
        self, variable: int
    ) -> Set[Tuple[int, int]]:
        """
        Get the identified parents of a variable.

        Parameters
        ----------
        variable : int
            Variable index.

        Returns
        -------
        set of (int, int)
            Set of (variable_index, lag) tuples representing parents.

        Examples
        --------
        >>> parents = pcmci.get_parents(0)
        >>> for var, lag in parents:
        ...     print(f"X{var}(t{lag}) -> X0(t)")
        """
        if variable not in self._parents:
            return set()
        return self._parents[variable].copy()

    @property
    def val_matrix(self) -> Optional[jax.Array]:
        """Get the test statistic matrix from the last run."""
        return self._val_matrix

    @property
    def pval_matrix(self) -> Optional[jax.Array]:
        """Get the p-value matrix from the last run."""
        return self._pval_matrix

    def __repr__(self) -> str:
        return (
            f"PCMCI(N={self.N}, T={self.T}, test={self.test.name}, "
            f"verbosity={self.verbosity})"
        )
