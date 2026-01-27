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

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from tqdm import tqdm
from itertools import combinations
import random
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

        rng = random.Random(seed)
        reservoir: List[Tuple[Tuple[int, int], ...]] = []

        for idx, combo in enumerate(combinations(items, k)):
            if idx < max_subsets:
                reservoir.append(combo)
            else:
                j = rng.randint(0, idx)
                if j < max_subsets:
                    reservoir[j] = combo

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
            try:
                device = jax.devices()[0]
                if hasattr(device, 'memory_stats'):
                    mem_stats = device.memory_stats() or {}
                    # Use bytes_limit if available, otherwise estimate 8GB
                    total_mem = mem_stats.get('bytes_limit', 8 * 1024**3)
                    in_use = mem_stats.get('bytes_in_use', 0)
                    available = (total_mem - in_use) * 0.7  # Use 70% of available
                    
                    # Estimate bytes per test: (X + Y + Z) * dtype_size * 3 (intermediates)
                    dtype_size = 4 if config.precision.value == 'float32' else 8
                    bytes_per_test = n_samples * (2 + n_conditions) * dtype_size * 3
                    
                    if bytes_per_test > 0:
                        computed_batch = max(64, int(available / bytes_per_test))
                        return min(computed_batch, 4096)  # Cap at reasonable max
            except Exception:
                pass  # Fall through to None
        
        return None

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

    def run_pc_stable(
        self,
        tau_max: int = 1,
        tau_min: int = 1,
        pc_alpha: Optional[float] = 0.05,
        max_conds_dim: Optional[int] = None,
        max_subsets: int = 100,
    ) -> Dict[int, Set[Tuple[int, int]]]:
        """
        Run the PC-stable condition selection algorithm.

        For each target variable, iteratively tests potential parents
        and removes those that are conditionally independent given
        subsets of other potential parents.

        Parameters
        ----------
        tau_max : int, default=1
            Maximum time lag.
        tau_min : int, default=1
            Minimum time lag.
        pc_alpha : float or None, default=0.05
            Significance level. If None, keeps all parents.
        max_conds_dim : int or None
            Maximum conditioning set dimension.
        max_subsets : int, default=100
            Maximum number of conditioning subsets to test per parent.
            Randomly samples if more subsets are available.

        Returns
        -------
        dict
            Dictionary mapping each variable index to its set of
            parents as (variable, lag) tuples.

        Examples
        --------
        >>> parents = pcmci.run_pc_stable(tau_max=3, pc_alpha=0.05)
        >>> print(f"Parents of X0: {parents[0]}")

        Notes
        -----
        This implements the "stable" version of PC, where the removal
        decisions are made based on the parents at the start of each
        iteration, not the current (changing) parent set.
        """
        parents: Dict[int, Set[Tuple[int, int]]] = {}

        # Initialize: all lagged variables are potential parents
        for j in self.selected_variables:
            parents[j] = set()
            for i in range(self.N):
                for tau in range(tau_min, tau_max + 1):
                    # Include all (i, -tau) as potential parents of j
                    # except (j, 0) which is the target itself
                    if not (i == j and tau == 0):
                        parents[j].add((i, -tau))

        if pc_alpha is None:
            # No selection - keep all parents
            return parents

        # Iteratively increase conditioning set size
        cond_dim = 0
        max_dim = max_conds_dim if max_conds_dim is not None else self.N * tau_max

        iterator = range(max_dim + 1)
        if self.verbosity >= 1:
            iterator = tqdm(iterator, desc="PC iterations", leave=False)

        for cond_dim in iterator:
            # Store parents at start of iteration (stable PC)
            parents_snapshot = {j: parents[j].copy() for j in self.selected_variables}
            any_removed = False

            # For cond_dim=0, use batch testing for efficiency
            if cond_dim == 0 and hasattr(self.test, 'run_batch'):
                for j in self.selected_variables:
                    current_parents = list(parents_snapshot[j])
                    if not current_parents:
                        continue
                    
                    # Group parents by lag (same lag = same data length)
                    parents_by_lag: Dict[int, List[Tuple[int, int]]] = {}
                    for parent in current_parents:
                        i, neg_tau = parent
                        tau = -neg_tau
                        if tau not in parents_by_lag:
                            parents_by_lag[tau] = []
                        parents_by_lag[tau].append(parent)
                    
                    parents_to_remove = []
                    
                    # Process each lag group in memory-aware batches
                    # Estimate n_samples for this tau
                    effective_T = self.T - max(parents_by_lag.keys())
                    batch_size = self._get_effective_batch_size(n_samples=effective_T, n_conditions=0)
                    for tau, parents_with_tau in parents_by_lag.items():
                        if batch_size is None:
                            chunk_ranges = [(0, len(parents_with_tau))]
                        else:
                            chunk_ranges = [
                                (start, min(start + batch_size, len(parents_with_tau)))
                                for start in range(0, len(parents_with_tau), batch_size)
                            ]

                        for start, end in chunk_ranges:
                            parent_list = []
                            for parent in parents_with_tau[start:end]:
                                parent_list.append(parent)
                            
                            i_arr = jnp.asarray([p[0] for p in parent_list], dtype=jnp.int32)
                            j_arr = jnp.full((len(parent_list),), j, dtype=jnp.int32)
                            tau_arr = jnp.full((len(parent_list),), tau, dtype=jnp.int32)
                            X_batch, Y_batch, _ = self.datahandler.get_variable_pair_batch(
                                i_arr,
                                j_arr,
                                tau_arr,
                                max_lag=tau,
                            )
                            
                            # Run batch test for this lag group chunk
                            stats, pvals = self.test.run_batch(X_batch, Y_batch, None, alpha=pc_alpha)
                            
                            # Mark non-significant (independent) parents for removal
                            for idx, (val, pval) in enumerate(zip(stats, pvals)):
                                if float(pval) > pc_alpha:  # Not significant = independent
                                    parents_to_remove.append(parent_list[idx])
                                    any_removed = True
                    
                    for parent in parents_to_remove:
                        parents[j].discard(parent)
                    
                    if self.verbosity >= 2 and parents_to_remove:
                        print(f"  Removed {len(parents_to_remove)} parents from X{j} (batch)")
            else:
                # Cond_dim > 0: Optimize with batching, but keep memory bounded
                if hasattr(self.test, 'run_batch'):
                    for j in self.selected_variables:
                        current_parents = list(parents_snapshot[j])
                        if len(current_parents) <= cond_dim:
                            continue

                        specs_by_lag: Dict[int, List[Tuple[int, Tuple[int, int], int, Tuple[Tuple[int, int], ...]]]] = {}

                        for parent in current_parents:
                            i, neg_tau = parent
                            tau = -neg_tau
                            other_parents = [p for p in current_parents if p != parent]

                            if len(other_parents) < cond_dim:
                                continue

                            subsets_to_test = self._sample_condition_subsets(
                                other_parents,
                                cond_dim,
                                max_subsets,
                                seed=i * 1000 + j * 100 + tau + cond_dim,
                            )

                            for subset in subsets_to_test:
                                max_lag_in_subset = max(-p[1] for p in subset) if subset else 0
                                effective_max_lag = max(tau, max_lag_in_subset)
                                if effective_max_lag not in specs_by_lag:
                                    specs_by_lag[effective_max_lag] = []
                                specs_by_lag[effective_max_lag].append((j, parent, tau, subset))

                        for max_lag, specs in specs_by_lag.items():
                            effective_T = self.T - max_lag
                            batch_size = self._get_effective_batch_size(
                                n_samples=effective_T, n_conditions=cond_dim
                            )

                            if batch_size is None or batch_size > len(specs):
                                chunk_ranges = [(0, len(specs))]
                            else:
                                chunk_ranges = [
                                    (start, min(start + batch_size, len(specs)))
                                    for start in range(0, len(specs), batch_size)
                                ]

                            for start, end in chunk_ranges:
                                batch_specs = specs[start:end]

                                i_list = []
                                j_list = []
                                tau_list = []
                                cond_vars_list = []
                                cond_lags_list = []

                                for j_idx, parent, tau, subset in batch_specs:
                                    i_list.append(parent[0])
                                    j_list.append(j_idx)
                                    tau_list.append(tau)

                                    c_vars = [p[0] for p in subset]
                                    c_lags = [-p[1] for p in subset]
                                    cond_vars_list.append(c_vars)
                                    cond_lags_list.append(c_lags)

                                i_arr = jnp.asarray(i_list, dtype=jnp.int32)
                                j_arr = jnp.asarray(j_list, dtype=jnp.int32)
                                tau_arr = jnp.asarray(tau_list, dtype=jnp.int32)
                                cond_vars = jnp.asarray(cond_vars_list, dtype=jnp.int32)
                                cond_lags = jnp.asarray(cond_lags_list, dtype=jnp.int32)

                                X_b, Y_b, Z_b = self.datahandler.get_variable_pair_batch(
                                    i_arr, j_arr, tau_arr, cond_vars, cond_lags, max_lag=max_lag
                                )

                                stats, pvals = self.test.run_batch(X_b, Y_b, Z_b, alpha=pc_alpha)

                                for idx, pval in enumerate(pvals):
                                    if float(pval) > pc_alpha:
                                        j_idx = batch_specs[idx][0]
                                        p_idx = batch_specs[idx][1]
                                        if p_idx in parents[j_idx]:
                                            parents[j_idx].discard(p_idx)
                                            any_removed = True

                    if self.verbosity >= 2 and any_removed:
                        print(f"  Batch removal completed for cond_dim {cond_dim}")

                else:
                    # Sequential fallback
                    for j in self.selected_variables:
                        current_parents = list(parents_snapshot[j])

                        if len(current_parents) <= cond_dim:
                            continue

                        # Test each parent
                        parents_to_remove = []

                        for parent in current_parents:
                            i, neg_tau = parent
                            tau = -neg_tau

                            # Get possible conditioning sets (subsets of other parents)
                            other_parents = [p for p in current_parents if p != parent]

                            # Test with subsets of size cond_dim
                            if len(other_parents) >= cond_dim:
                                is_independent = self._test_with_conditioning_subsets(
                                    i=i,
                                    j=j,
                                    tau=tau,
                                    other_parents=other_parents,
                                    cond_dim=cond_dim,
                                    pc_alpha=pc_alpha,
                                    max_subsets=max_subsets,
                                )

                                if is_independent:
                                    parents_to_remove.append(parent)
                                    any_removed = True

                        # Remove independent parents
                        for parent in parents_to_remove:
                            parents[j].discard(parent)

                        if self.verbosity >= 2 and parents_to_remove:
                            print(f"  Removed {len(parents_to_remove)} parents from X{j}")

            if not any_removed:
                # No removals - algorithm converged
                break

        if self.verbosity >= 1:
            total_parents = sum(len(p) for p in parents.values())
            print(f"PC phase complete: {total_parents} total parent links")

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
        from itertools import combinations
        import random

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
                        cond_vars_list = []
                        cond_lags_list = []
                        for subset in subsets_group[start:end]:
                            condition_indices = [(var, -neg_lag) for var, neg_lag in subset]
                            cond_vars_list.append([var for var, _ in condition_indices])
                            cond_lags_list.append([lag for _, lag in condition_indices])

                        batch_len = end - start
                        i_arr = jnp.full((batch_len,), i, dtype=jnp.int32)
                        j_arr = jnp.full((batch_len,), j, dtype=jnp.int32)
                        tau_arr = jnp.full((batch_len,), tau, dtype=jnp.int32)
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
                test_indices = []
                i_list = []
                j_list = []
                tau_list = []
                cond_vars_list = [] if n_cond > 0 else None
                cond_lags_list = [] if n_cond > 0 else None

                for i, j, tau, cond_set in tests[start:end]:
                    test_indices.append((i, j, tau))
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
                i_arr = jnp.asarray([idx[0] for idx in test_indices], dtype=jnp.int32)
                j_arr = jnp.asarray([idx[1] for idx in test_indices], dtype=jnp.int32)
                tau_arr = jnp.asarray([idx[2] for idx in test_indices], dtype=jnp.int32)
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
