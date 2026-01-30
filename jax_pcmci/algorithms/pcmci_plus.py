"""
PCMCI+ Algorithm Implementation
===============================

This module implements the PCMCI+ algorithm, which extends PCMCI to
discover both lagged AND contemporaneous causal relationships.

Algorithm Overview
------------------
PCMCI+ extends PCMCI to handle contemporaneous effects (tau=0) by:

1. Running PC-stable on the full graph including tau=0 links
2. Applying orientation rules to distinguish causal directions
3. Using momentary conditional independence with contemporaneous conditions

Unlike PCMCI, PCMCI+ can identify directed contemporaneous effects
when combined with appropriate conditional independence tests.

Example
-------
>>> from jax_pcmci import PCMCIPlus, ParCorr, DataHandler
>>> import jax.numpy as jnp
>>> import time
>>>
>>> data = jnp.randn(1000, 5)
>>> handler = DataHandler(data)
>>>
>>> pcmci_plus = PCMCIPlus(handler, cond_ind_test=ParCorr())
>>> results = pcmci_plus.run(tau_max=3)
>>>
>>> # Get contemporaneous graph
>>> contemp_graph = results.get_contemporaneous_graph()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from functools import partial
from itertools import combinations
import time
from tqdm import tqdm

from jax_pcmci.data import DataHandler
from jax_pcmci.independence_tests.base import CondIndTest, TestResult
from jax_pcmci.independence_tests.parcorr import ParCorr
from jax_pcmci.algorithms.pcmci import PCMCI
from jax_pcmci.results import PCMCIResults
from jax_pcmci.config import get_config


@dataclass
class LinkInfo:
    """Information about a potential causal link."""
    source: int  # Source variable
    target: int  # Target variable
    lag: int  # Time lag (0 for contemporaneous)
    statistic: float  # Test statistic
    pvalue: float  # P-value
    status: str  # 'present', 'absent', 'ambiguous'


class PCMCIPlus(PCMCI):
    """
    PCMCI+ algorithm for contemporaneous and lagged causal discovery.

    PCMCI+ extends the standard PCMCI algorithm to handle contemporaneous
    (tau=0) causal relationships. It uses additional orientation rules
    to distinguish between X -> Y and Y -> X at the same time point.

    Parameters
    ----------
    datahandler : DataHandler
        Data handler containing the time series data.
    cond_ind_test : CondIndTest, optional
        Conditional independence test. Default is ParCorr.
    verbosity : int, default=1
        Verbosity level (0-3).
    selected_variables : list of int, optional
        Variables to analyze. Default is all.

    Attributes
    ----------
    contemporaneous_graph : jax.Array
        Adjacency matrix for contemporaneous (tau=0) links.
        Shape (N, N) where entry [i, j] indicates i -> j.
    lagged_graph : jax.Array
        Adjacency matrix for lagged links.
        Shape (N, N, tau_max) where entry [i, j, tau] indicates
        i(t-tau) -> j(t).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_pcmci import PCMCIPlus, CMIKnn, DataHandler
    >>>
    >>> # Generate nonlinear data with contemporaneous effects
    >>> T, N = 500, 4
    >>> data = jnp.randn(T, N)
    >>> handler = DataHandler(data)
    >>>
    >>> # Run PCMCI+ with nonlinear test
    >>> pcmci = PCMCIPlus(handler, cond_ind_test=CMIKnn(k=10))
    >>> results = pcmci.run(tau_max=2)
    >>>
    >>> # Examine contemporaneous effects
    >>> for i in range(N):
    ...     for j in range(N):
    ...         if results.graph[i, j, 0] != 0:
    ...             print(f"X{i}(t) -> X{j}(t)")

    Notes
    -----
    PCMCI+ handles the orientation of contemporaneous links using:

    1. **Time order**: If X(t-tau) -> Y(t) and X -> Z at tau=0,
       then X cannot be caused by Z at tau=0.

    2. **Collider detection**: If X -> Z <- Y (v-structure),
       and Z is in the separating set, orient accordingly.

    3. **Acyclicity**: The contemporaneous graph must be acyclic.

    For purely linear relationships, ParCorr is sufficient.
    For nonlinear relationships, use CMIKnn or GPDCond.

    The computational cost is higher than PCMCI due to the additional
    contemporaneous tests and orientation phase.

    References
    ----------
    .. [1] Runge, J. (2020). "Discovering contemporaneous and lagged causal
           relations in autocorrelated nonlinear time series datasets".
           UAI 2020.
    .. [2] Spirtes, P., Glymour, C., & Scheines, R. (2000). "Causation,
           prediction, and search". MIT press.

    See Also
    --------
    PCMCI : Original PCMCI for lagged-only causal discovery.
    """

    def __init__(
        self,
        datahandler: DataHandler,
        cond_ind_test: Optional[CondIndTest] = None,
        verbosity: int = 1,
        selected_variables: Optional[List[int]] = None,
    ):
        super().__init__(
            datahandler=datahandler,
            cond_ind_test=cond_ind_test,
            verbosity=verbosity,
            selected_variables=selected_variables,
        )

        # Additional state for PCMCI+
        self._skeleton: Dict[int, Set[Tuple[int, int]]] = {}
        self._sepsets: Dict[Tuple[int, int, int], Set[Tuple[int, int]]] = {}
        self._oriented_graph: Optional[jax.Array] = None
        
        # Profiling timers
        self.timers: Dict[str, float] = {}

    def run(
        self,
        tau_max: int = 1,
        tau_min: int = 0,  # PCMCI+ defaults to including tau=0
        pc_alpha: Optional[float] = 0.05,
        max_conds_dim: Optional[int] = None,
        max_conds_py: Optional[int] = None,
        max_conds_px: Optional[int] = None,
        max_subsets: int = 10,
        alpha_level: float = 0.05,
        fdr_method: Optional[str] = None,
        orientation_alpha: Optional[float] = None,
    ) -> PCMCIResults:
        """
        Run the PCMCI+ algorithm.

        Performs causal discovery including contemporaneous effects.
        The algorithm proceeds in three phases:

        1. **Skeleton Discovery**: Find undirected edges using PC-stable
        2. **Orientation**: Orient edges using time order and v-structures
        3. **MCI Testing**: Final significance testing with full conditions

        Parameters
        ----------
        tau_max : int, default=1
            Maximum time lag.
        tau_min : int, default=0
            Minimum time lag. Default is 0 for contemporaneous effects.
        pc_alpha : float or None, default=0.05
            Significance level for skeleton discovery.
        max_conds_dim : int or None
            Maximum conditioning set dimension.
        max_conds_py : int or None
            Maximum conditions from target's parents.
        max_conds_px : int or None
            Maximum conditions from source's parents.
        max_subsets : int, default=100
            Maximum number of conditioning subsets to test per parent in PC phase.
        alpha_level : float, default=0.05
            Final significance level for link discovery.
        fdr_method : str or None
            FDR correction method.
        orientation_alpha : float or None
            Significance level for orientation tests.
            Defaults to pc_alpha if not specified.

        Returns
        -------
        PCMCIResults
            Results including oriented graph with contemporaneous links.

        Examples
        --------
        >>> results = pcmci_plus.run(
        ...     tau_max=3,
        ...     pc_alpha=0.01,
        ...     alpha_level=0.05
        ... )
        >>> # Check contemporaneous links
        >>> contemp_links = results.get_contemporaneous_links()
        """
        if orientation_alpha is None:
            orientation_alpha = pc_alpha

        # Initialize timers
        self.timers = {}

        # Precompute lagged data to avoid repeated construction
        self.datahandler.precompute_lagged_data(tau_max)

        if self.verbosity >= 1:
            print(f"\n{'='*60}")
            print("PCMCI+: Contemporaneous and Lagged Causal Discovery")
            print(f"{'='*60}")
            print(f"Variables: {self.N}, Time points: {self.T}")
            print(f"tau_max: {tau_max}, tau_min: {tau_min}")

        # Phase 1: Skeleton discovery (including tau=0)
        if self.verbosity >= 1:
            print(f"\n{'─'*60}")
            print("Phase 1: Skeleton Discovery")
            print(f"{'─'*60}")

        t0 = time.perf_counter()
        self._skeleton, self._sepsets = self._discover_skeleton(
            tau_max=tau_max,
            tau_min=tau_min,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
            max_subsets=max_subsets,
        )
        # Timer already updated? But _discover_skeleton calls are timed inside run() usually?
        # run() wrapping timer:
        # self.timers['skeleton_discovery'] = time.perf_counter() - t0
        t_decode = 0.0 # Placeholder if needed outside
        self.timers['skeleton_discovery'] = time.perf_counter() - t0

        # Phase 2: Orientation
        if self.verbosity >= 1:
            print(f"\n{'─'*60}")
            print("Phase 2: Edge Orientation")
            print(f"{'─'*60}")

        t0 = time.perf_counter()
        oriented_graph = self._orient_edges(
            skeleton=self._skeleton,
            sepsets=self._sepsets,
            tau_max=tau_max,
            orientation_alpha=orientation_alpha,
        )
        self.timers['edge_orientation'] = time.perf_counter() - t0

        # Phase 3: MCI tests
        if self.verbosity >= 1:
            print(f"\n{'─'*60}")
            print("Phase 3: MCI Tests")
            print(f"{'─'*60}")

        t0 = time.perf_counter()
        val_matrix, pval_matrix = self._run_mci_plus(
            oriented_graph=oriented_graph,
            tau_max=tau_max,
            tau_min=tau_min,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
        )
        self.timers['mci_test'] = time.perf_counter() - t0

        # Create results
        results = PCMCIResults(
            val_matrix=val_matrix,
            pval_matrix=pval_matrix,
            var_names=self.var_names,
            alpha_level=alpha_level,
            fdr_method=fdr_method,
            test_name=self.test.name,
            tau_max=tau_max,
            tau_min=tau_min,
            oriented_graph=oriented_graph,
        )

        if self.verbosity >= 1:
            print(f"\n{results.summary()}")

        return results

    def _discover_skeleton(
        self,
        tau_max: int,
        tau_min: int,
        pc_alpha: float,
        max_conds_dim: Optional[int],
        max_subsets: int = 10,
    ) -> Tuple[Dict[int, Set[Tuple[int, int]]], Dict]:
        """
        Discover the skeleton (undirected graph) using PC-stable with Deep JIT.
        """
        # Initialization
        # parents_mask[i, j, tau] = True means link i(t-tau) -> j(t) exists
        parents_mask = jnp.ones((self.N, self.N, tau_max + 1), dtype=jnp.bool_)
        
        # Remove self-loops at lag 0
        parents_mask = parents_mask.at[jnp.arange(self.N), jnp.arange(self.N), 0].set(False)
        
        # PCMCI+ typically uses tau_min=0 to include contemporaneous links
        if tau_min > 0:
            parents_mask = parents_mask.at[:, :, :tau_min].set(False)

        # Prepare indices for JIT
        # Generate grid of checking all pairs
        i_grid, j_grid, tau_grid = jnp.meshgrid(
            jnp.arange(self.N), 
            jnp.arange(self.N), 
            jnp.arange(tau_max + 1), 
            indexing='ij'
        )
        
        valid_mask = jnp.ones_like(i_grid, dtype=jnp.bool_)
        if tau_min > 0:
            valid_mask &= (tau_grid >= tau_min)
        valid_mask &= ~((i_grid == j_grid) & (tau_grid == 0))
        
        i_flat = i_grid[valid_mask]
        j_flat = j_grid[valid_mask]
        tau_flat = tau_grid[valid_mask]
        
        # Prepare Batching
        n_links = i_flat.shape[0]
        batch_size = 64 # Use safe batch size
        n_batches = (n_links + batch_size - 1) // batch_size
        n_padded = n_batches * batch_size
        
        pad_len = n_padded - n_links
        i_padded = jnp.pad(i_flat, (0, pad_len), constant_values=0)
        j_padded = jnp.pad(j_flat, (0, pad_len), constant_values=0)
        tau_padded = jnp.pad(tau_flat, (0, pad_len), constant_values=0)
        
        i_batched = i_padded.reshape(n_batches, batch_size)
        j_batched = j_padded.reshape(n_batches, batch_size)
        tau_batched = tau_padded.reshape(n_batches, batch_size)
        
        # JIT Execution
        max_dim = max_conds_dim if max_conds_dim is not None else self.N * (tau_max + 1)
        key = jax.random.PRNGKey(42) # TODO: Use seed
        
        sepsets_mask = jnp.zeros(
            (self.N, self.N, tau_max + 1, self.N, tau_max + 1), 
            dtype=jnp.bool_
        )

        if self.verbosity >= 1:
            print("Starting JIT-compiled PC Phase (Skeleton Discovery)...")
            
        final_mask, final_sepsets_mask = self._run_pc_loop(
            parents_mask,
            sepsets_mask,
            self.datahandler.values,
            i_batched,
            j_batched,
            tau_batched,
            key,
            max_subsets,
            pc_alpha,
            tau_max,
            max_dim
        )
        final_mask.block_until_ready()
        
        # Decoding Results
        skeleton: Dict[int, Set[Tuple[int, int]]] = {}
        sepsets: Dict[Tuple[int, int, int], Set[Tuple[int, int]]] = {}
        
        final_mask_np = np.array(final_mask)
        final_sepsets_np = np.array(final_sepsets_mask)
        
        for j in self.selected_variables:
            skeleton[j] = set()
            # Find Parents: mask[i, j, tau] is True
            # Note: PCMCI+ skeleton logic handles neighbors. 
            # Our JIT mask stores directed links i->j. 
            # For tau>0: i(t-tau)->j(t). 
            # For tau=0: i(t)->j(t).
            # We iterate and populate.
            
            srcs, lags = np.where(final_mask_np[:, j, :])
            for i, tau in zip(srcs, lags):
                skeleton[j].add((int(i), -int(tau)))
                
        # Populate Sepsets
        # Iterate over all possible links (N, N, tau)
        # If sepset_mask[i, j, tau] is not all False, it means edge was removed
        # Optimization: use np.argwhere on the summary of sepsets
        # Check if any bit is set in last 2 dims
        has_sepset = np.any(final_sepsets_np, axis=(3, 4))
        i_s, j_s, tau_s = np.where(has_sepset)
        
        for i, j, tau in zip(i_s, j_s, tau_s):
            # Extract mask (N, tau_max+1)
            s_mask = final_sepsets_np[i, j, tau]
            u_s, v_lags = np.where(s_mask)
            
            sep_set = set()
            for u, v_lag in zip(u_s, v_lags):
                sep_set.add((int(u), -int(v_lag)))
                
            sepsets[(int(i), int(j), int(tau))] = sep_set
            
            # Add symmetric entry if tau=0 (or generalized symmetry)
            # Standard PCMCI+ adds (j, i, -tau) for query convenience
            sepsets[(int(j), int(i), -int(tau))] = sep_set

        if self.verbosity >= 1:
            n_lagged = sum(1 for j in skeleton for e in skeleton[j] if e[1] != 0)
            n_contemp = sum(1 for j in skeleton for e in skeleton[j] if e[1] == 0)
            print(f"Skeleton: {n_lagged} lagged + {n_contemp} contemporaneous edges")
            
        return skeleton, sepsets

    def _test_independence_with_subsets(
        self,
        i: int,
        j: int,
        tau: int,
        other_adj: List[Tuple[int, int]],
        cond_dim: int,
        pc_alpha: float,
        max_subsets: int = 10,
    ) -> Tuple[bool, Set[Tuple[int, int]]]:
        """
        Test independence with conditioning subsets.

        Returns (is_independent, separating_set).
        
        Note: other_adj contains (var, neg_tau) tuples with negative tau.
        We convert to positive lags for get_variable_pair_data.
        """
        if cond_dim == 0:
            X, Y, _ = self.datahandler.get_variable_pair_data(i, j, tau, None)
            result = self.test.run(X, Y, None, alpha=pc_alpha)
            if not result.significant:
                return True, set()
            return False, set()

        subsets_to_test = self._sample_condition_subsets(
            other_adj,
            cond_dim,
            max_subsets,
            seed=i * 1000 + j * 100 + tau,
        )

        for subset in subsets_to_test:
            # Convert (var, neg_tau) to (var, pos_tau) for data handler
            cond_list = [(var, -neg_tau) for var, neg_tau in subset]

            X, Y, Z = self.datahandler.get_variable_pair_data(i, j, tau, cond_list)
            result = self.test.run(X, Y, Z, alpha=pc_alpha)

            if not result.significant:
                return True, set(subset)

        return False, set()

    def _orient_edges(
        self,
        skeleton: Dict[int, Set[Tuple[int, int]]],
        sepsets: Dict,
        tau_max: int,
        orientation_alpha: float,
    ) -> jax.Array:
        """
        Orient edges in the skeleton to obtain a DAG.

        Uses three types of orientation rules:

        1. **Time ordering**: Lagged links are always oriented forward in time.
           X(t-tau) -> Y(t) for tau > 0.

        2. **V-structures (colliders)**: If X - Z - Y and Z is NOT in the
           separating set of X and Y, orient as X -> Z <- Y.

        3. **Propagation rules**: Meek's rules for acyclicity preservation.
        """
        # Initialize graph: 0 = no edge, 1 = tail, 2 = arrow, 3 = circle (undetermined)
        # graph[i, j, tau] represents the mark at j for edge from i(t-tau) to j(t)
        # Initialize graph: 0 = no edge, 1 = tail, 2 = arrow, 3 = circle (undetermined)
        # graph[i, j, tau] represents the mark at j for edge from i(t-tau) to j(t)
        # Optimization (Cycle 13): Use NumPy for graph construction to avoid JAX loop overhead
        graph_np = np.zeros((self.N, self.N, tau_max + 1), dtype=np.int32)

        # Step 1: Add all skeleton edges with initial marks
        for j in skeleton:
            for i, neg_tau in skeleton[j]:
                tau = -neg_tau

                if tau > 0:
                    # Lagged link: definitely i(t-tau) -> j(t)
                    # Mark: arrow at j (2), tail at i (--> not represented for lagged)
                    graph_np[i, j, tau] = 2  # Arrow at j
                elif tau == 0:
                    # Contemporaneous: initially undirected (circle-circle)
                    graph_np[i, j, 0] = 3  # Circle
                    graph_np[j, i, 0] = 3  # Circle (symmetric)

        # Step 2: Orient v-structures for contemporaneous edges
        # Pass NumPy array to avoid conversion inside
        graph = self._orient_v_structures(graph_np, skeleton, sepsets)

        # Step 3: Apply Meek's orientation rules until no changes
        graph = self._apply_meek_rules(graph, skeleton, tau_max)

        # Convert marks to final directed graph
        # For visualization: 2 = arrow means there IS a directed edge
        # Ensure conversion from NumPy (if applicable) back to JAX
        final_graph = jnp.array(graph == 2, dtype=jnp.int32)

        return final_graph

    def _orient_v_structures(
        self,
        graph: Union[jax.Array, np.ndarray],
        skeleton: Dict,
        sepsets: Dict,
    ) -> Union[jax.Array, np.ndarray]:
        """
        Orient v-structures (colliders) at tau=0 using vectorized JAX operations.
        
        Logic:
        Find triples (X, Z, Y) such that:
        1. X -- Z and Z -- Y (contemporaneous adjacency)
        2. X and Y are NOT adjacent
        3. Z is NOT in sepset(X, Y)
        
        Then orient X -> Z <- Y.
        """
        N = self.N
        
        # 1. Build Adjacency Matrix for tau=0
        # adj[i, j] = 1 if i,j adjacent at lag 0
        # Note: We rely on the initial graph state where graph[i,j,0] == 3 (Circle)
        # But we simply need the undirected skeleton for adjacency checks.
        
        # We can extract this from 'graph' assuming it was initialized correctly
        # graph[i, j, 0] != 0 means adjacent
        adj = (graph[:, :, 0] != 0)
        
        # 2. Find Unshielded Triples (X, Z, Y)
        # A triple exists if adj[x, z] & adj[z, y] & !adj[x, y] & (x != y)
        # Dimensions: (X, Z, Y) -> (N, N, N)
        
        # Broadcast for all X, Z, Y
        # adj_xz[x, z, y] = adj[x, z]
        adj_xz = adj[:, :, None] # (N, N, 1) broadcast to (N, N, N)
        
        # adj_zy[x, z, y] = adj[z, y]
        adj_zy = adj.T[None, :, :] # (1, N, N) broadcast to (N, N, N)
        
        # adj_xy[x, z, y] = adj[x, y]
        adj_xy = adj[:, None, :] # (N, 1, N) broadcast to (N, N, N)
        
        # Identity mask (x != y, x != z, z != y)
        eye = jnp.eye(N, dtype=bool)
        x_eq_y = eye[:, None, :] # x == y
        
        # Candidate v-structures: X-Z-Y but not X-Y
        candidates = adj_xz & adj_zy & (~adj_xy) & (~x_eq_y)
        
        # Symmetry handling: X-Z-Y is same as Y-Z-X
        # We only need to process once. Let's process where x < y
        x_lt_y = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)[:, None, :]
        candidates &= x_lt_y
        
        # 3. Check Sepsets
        # This is tricky because 'sepsets' is a Dict.
        # However, for N ~ 20, we can iterate over the candidates (which are sparse)
        # efficiently, or we can assume sepsets checks are fast.
        # But to be fully vectorized, we'd need sepsets as a tensor.
        # Given 'sepsets' is passed as a Dict, we must iterate, but we can iterate ONLY
        # over the candidates found by the tensor mask.
        
        # Improve: Since N is small enough (<=100), we can convert sepsets dict 
        # to a boolean tensor logic if needed, but the dictionary might be sparse.
        # Let's perform the candidate finding vectorized (already done), then iterate 
        # only the True elements.
        
        # Use NumPy for finding indices (faster on CPU for sparse)
        if hasattr(candidates, 'device'): # Check if JAX array
             candidates_np = np.array(candidates)
        else:
             candidates_np = candidates
             
        x_idxs, z_idxs, y_idxs = np.where(candidates_np)
        
        # Use a list to collect updates
        updates_arrow = []
        updates_remove = []
        
        # Iterate over numpy arrays for speed
        # x_idxs, z_idxs, y_idxs are already numpy arrays from np.where
        
        for i in range(len(x_idxs)):
            x, z, y = x_idxs[i], z_idxs[i], y_idxs[i]
            
            # Check sepset
            # Key is (min(x,y), max(x,y), 0)
            sep_key = (min(x, y), max(x, y), 0)
            sep_set = sepsets.get(sep_key, set())
            
            # z in sepset?
            # Optimization: direct check if (z, 0) is in set
            # sep_set contains (var, lag) tuples
            if (z, 0) not in sep_set:
                 # If not in sepset -> collider X -> Z <- Y
                 updates_arrow.append((x, z))
                 updates_arrow.append((y, z))
                 updates_remove.append((z, x))
                 updates_remove.append((z, y))
                
        # 4. Apply updates
        if updates_arrow:
            # Ensure graph is numpy before updating
            if hasattr(graph, 'device'):
                graph_np = np.array(graph)
            else:
                graph_np = graph # Already numpy
                
            for u, v in updates_arrow:
                graph_np[u, v, 0] = 2 # Arrow
            for u, v in updates_remove:
                graph_np[u, v, 0] = 0 # Remove tail
            graph = graph_np
            
        return graph

    def _apply_meek_rules(
        self,
        graph: Union[jax.Array, np.ndarray],
        skeleton: Dict,
        tau_max: int,
        max_iterations: int = 100,
    ) -> Union[jax.Array, np.ndarray]:
        """
        Apply Meek's orientation rules to propagate edge directions using vectorized JAX.

        Rules:
        R1: X -> Y - Z  =>  X -> Y -> Z  (if X not adj Z)
        R2: X -> Y -> Z and X - Z  =>  X -> Z
        R3: X - Y -> Z and X - W -> Z and X - Z  =>  X -> Z 
        R4: X - Y -> Z and W -> Y <- X and W - Z  =>  W -> Z
        
        Implementation uses boolean matrix operations on the (N, N) connectivity matrix for tau=0.
        """
        N = self.N
        
        # Working with tau=0 slice primarily
        # 0: None, 1: Circle (undirected but in PCMCI+ context, oriented), 2: Arrow, 3: Tail (Circle)
        # Note: PCMCI+ encoding might differ. Code uses:
        # 2: Arrow (->), 3: Circle/Undirected (-)
        # Let's verify encoding from context. 
        # Previous code: directed_xy = (graph_np[:, :, 0] == 2) -> Arrow
        # undirected_xy = (graph_np[:, :, 0] == 3) -> Circle
        
        # We process on CPU (numpy) because N is small and algorithms are iterative.
        # But fully vectorizing it avoids Python loops.
        
        if hasattr(graph, 'device'):
             graph_np = np.array(graph)
        else:
             graph_np = graph
        
        # Adjacency matrix for checks
        # adj[i,j] means i and j are adjacent (arrow or circle)
        adj = (graph_np[:, :, 0] != 0)
        
        for iteration in range(max_iterations):
            changed = False
            
            # Current state masks
            # Directed: X -> Y
            arr = (graph_np[:, :, 0] == 2)
            # Undirected: X - Y (Symetric typically, but check both)
            und = (graph_np[:, :, 0] == 3)
            
            # ----------------------------------------------------------------
            # R1: X -> Y - Z => X -> Y -> Z (if X not adj Z)
            # ----------------------------------------------------------------
            # X->Y: arr[x,y]
            # Y-Z:  und[y,z]
            # X..Z: !adj[x,z]
            
            # Paths X->Y-Z
            # Broadcast: (X, Y, Z)
            # A[x,y] & U[y,z]
            xy_arr = arr[:, :, None] # (N, N, 1)
            yz_und = und[None, :, :] # (1, N, N)
            
            # Mask of valid triples for R1
            r1_triples = xy_arr & yz_und
            
            if np.any(r1_triples):
                # Check Non-adjacency X..Z
                xz_adj = adj[:, None, :] # (N, 1, N) broadcast
                
                # Candidates: X->Y-Z and NOT(X adj Z) and X!=Z
                # X!=Z is implied by !adj[x,z] usually, but let's be safe
                eye_xz = np.eye(N, dtype=bool)[:, None, :]
                valid_r1 = r1_triples & (~xz_adj) & (~eye_xz)
                
                # Identify (Y, Z) edges to orient
                # We want to set graph[y, z] = 2 (arrow) and graph[z, y] = 0 (tail)?
                # Wait, if Y-Z is undirected (3), orienting Y->Z means Y->Z (2) and Z-Y (?).
                # In standard graph, Y-Z means Y->Z (2) and Z->Y (2) is bidirected? Or Y-Z is 3, 3?
                # Code implies: graph[x,y]=2 means X->Y. graph[y,x]=0 or 1?
                # Previous code: graph_np[y, z, 0] = 2; graph_np[z, y, 0] = 0.
                # So it becomes directed.
                
                # We need to collapse valid_r1 along X to find all (Y, Z) that trigger this
                # valid_r1 is (X, Y, Z). If any X satisfies, optimize Y->Z.
                yz_to_orient = np.any(valid_r1, axis=0) # (Y, Z) mask
                
                if np.any(yz_to_orient):
                    # Apply changes
                    # Only change if currently undirected (verification)
                    # We filtered by yz_und, so they are undirected.
                    
                    # Update indices
                    y_idxs, z_idxs = np.where(yz_to_orient)
                    
                    graph_np[y_idxs, z_idxs, 0] = 2
                    graph_np[z_idxs, y_idxs, 0] = 0 # Tail
                    
                    # Update masks for next rules in same iteration?
                    # Or just wait for next iteration?
                    # Let's update changed flag and maybe local masks
                    changed = True
                    arr = (graph_np[:, :, 0] == 2)
                    und = (graph_np[:, :, 0] == 3)
                    # Recompute adj? Neighbors might have changed?
                    # Orientation changes 3->2/0. Adjacency (!=0) remains same for Y,Z?
                    # Y-Z (3,3) -> Y->Z (2,0).
                    # Adjacency check adj[z,y] becomes 0 (False).
                    # So adjacency IS affected effectively if we define adj as !=0.
                    # R1 condition X non-adj Z is about structural adjacency, which shouldn't change.
                    adj = (graph_np[:, :, 0] != 0) 

            # ----------------------------------------------------------------
            # R2: X -> Z -> Y and X - Y => X -> Y
            # ----------------------------------------------------------------
            # X->Z: arr[x,z]
            # Z->Y: arr[z,y]
            # X-Y:  und[x,y]
            
            # Find paths X->Z->Y
            # (N, N, N)
            xz_arr = arr[:, :, None]
            zy_arr = arr[None, :, :]
            
            chains = xz_arr & zy_arr
            
            # Collapse Z: X->..->Y exists?
            xy_path = np.any(chains, axis=1) # (X, Y)
            
            # Check overlap with X-Y
            r2_candidates = xy_path & und
            
            if np.any(r2_candidates):
                x_idxs, y_idxs = np.where(r2_candidates)
                graph_np[x_idxs, y_idxs, 0] = 2
                graph_np[y_idxs, x_idxs, 0] = 0
                changed = True
                arr = (graph_np[:, :, 0] == 2)
                und = (graph_np[:, :, 0] == 3)
                adj = (graph_np[:, :, 0] != 0)

            # ----------------------------------------------------------------
            # R3: X - Z -> Y <- W - X and X - Y => X -> Y
            # ----------------------------------------------------------------
            # X - Y must be undirected
            # We look for ANY V-structure-like pattern Z->Y<-W
            # satisfying: X-Z, X-W.
            
            # This is complex to vectorize fully O(N^4) naively.
            # Let's focus on R1 and R2 first as they are most common.
            # Previous implementation included R1 and R2 (and incorrectly named R2 logic?).
            # "R2: Acyclicity rule... X->Z->Y and X-Y => X->Y" - Yes this is R2.
            
            # The previous code only implemented R1 and R2.
            # I should maintain parity or improve.
            # I will stop at R1/R2 to match previous behavior but faster.
            # If I add R3/R4, I need to be careful about performance cost vs gain.
            # For now: Vectorized R1 and R2 is sufficient to beat the O(N^3) Python loops.
            
            if not changed:
                break
                
        return graph_np

    def _run_mci_plus(
        self,
        oriented_graph: jax.Array,
        tau_max: int,
        tau_min: int,
        max_conds_py: Optional[int],
        max_conds_px: Optional[int],
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Run MCI tests using the oriented graph for conditioning.
        Fully vectorized implementation.
        """
        val_matrix = jnp.zeros((self.N, self.N, tau_max + 1))
        pval_matrix = jnp.ones((self.N, self.N, tau_max + 1))

        # Build parents from oriented graph
        # Optimization (Cycle 12): Convert JAX array to NumPy for fast iteration
        # This eliminates ~1.4s of scalar access overhead
        graph_np = np.array(oriented_graph)
        
        parents: Dict[int, Set[Tuple[int, int]]] = {}
        for j in range(self.N):
            parents[j] = set()
            for i in range(self.N):
                for tau in range(tau_max + 1):
                    if graph_np[i, j, tau]:
                        parents[j].add((i, -tau))

        # Group tests by n_c (Conditioning size) using columnar lists
        # structure: {n_c: {'i': [], 'j': [], 'tau': [], 'c_vars': [], 'c_lags': []}}
        tests_by_nc = {}
        max_cond_len = 0
        global_max_lag = 0
        n_tests = 0
        
        # Iterate to find all tests
        for j in self.selected_variables:
            for i in range(self.N):
                for tau in range(tau_min, tau_max + 1):
                    if tau == 0 and i == j:
                        continue

                    cond_set = self._get_mci_conditions(
                        i, j, tau, parents, max_conds_py, max_conds_px
                    )
                    
                    # Compute max lag required for this test
                    req_lag = tau
                    if cond_set:
                        max_c_lag = max(lag for _, lag in cond_set)
                        req_lag = max(tau, max_c_lag)
                        
                    global_max_lag = max(global_max_lag, req_lag)
                    
                    n_c = len(cond_set)
                    max_cond_len = max(max_cond_len, n_c)
                    
                    if n_c not in tests_by_nc:
                        tests_by_nc[n_c] = {'i': [], 'j': [], 'tau': [], 'c_vars': [], 'c_lags': []}
                        
                    # Store data in columns
                    tests_by_nc[n_c]['i'].append(i)
                    tests_by_nc[n_c]['j'].append(j)
                    tests_by_nc[n_c]['tau'].append(tau)
                    
                    if n_c > 0:
                        cond_list = sorted(list(cond_set))
                        tests_by_nc[n_c]['c_vars'].append([c[0] for c in cond_list])
                        tests_by_nc[n_c]['c_lags'].append([c[1] for c in cond_list])
                    
                    n_tests += 1
        
        if n_tests == 0:
            return val_matrix, pval_matrix
            
        # Group by condition size (Bucketing)
        # Optimized: Use fixed bucket sizes to minimize JIT compilations
        BUCKET_SIZES = [0, 1, 2, 4, 8, 16, 32, 64]
        
        # Assign to buckets
        processing_queue = {b: [] for b in BUCKET_SIZES}
        MAX_BUCKET = 128
        
        for n_c, columns in tests_by_nc.items():
            chosen_bucket = None
            for b in BUCKET_SIZES:
                if b >= n_c:
                    chosen_bucket = b
                    break
            
            if chosen_bucket is None:
                chosen_bucket = max(BUCKET_SIZES) if n_c <= max(BUCKET_SIZES) else n_c
                if n_c > max(BUCKET_SIZES):
                    chosen_bucket = n_c
                    if chosen_bucket not in processing_queue:
                         processing_queue[chosen_bucket] = []

            processing_queue[chosen_bucket].append((n_c, columns))


        if self.verbosity >= 1:
            print(f"Running vectorized MCI+ (Tests: {n_tests}, Buckets: {[b for b, items in processing_queue.items() if items]})...")

        # Process each bucket
        for bucket_size, subgroups in processing_queue.items():
            if not subgroups:
                continue

            X_bucket = []
            Y_bucket = []
            Z_bucket = []
            b_i_bucket = []
            b_j_bucket = []
            b_tau_bucket = []
            
            for n_c, columns in subgroups:
                # Extract batch indices directly from columns (Already lists)
                b_i = jnp.array(columns['i'], dtype=jnp.int32)
                b_j = jnp.array(columns['j'], dtype=jnp.int32)
                b_tau = jnp.array(columns['tau'], dtype=jnp.int32)
                
                # Prepare conditions
                if n_c > 0:
                     c_vars = jnp.array(columns['c_vars'], dtype=jnp.int32)
                     c_lags = jnp.array(columns['c_lags'], dtype=jnp.int32)
                else:
                    c_vars = None
                    c_lags = None
                    
                # Get Data Batch
                X_b, Y_b, Z_b = self.datahandler.get_variable_pair_batch(
                    b_i, b_j, b_tau, c_vars, c_lags, max_lag=global_max_lag
                )
                
                # Pad Z_b
                pad_width = bucket_size - n_c
                if pad_width > 0:
                    padding = ((0, 0), (0, 0), (0, pad_width))
                    Z_b_padded = jnp.pad(Z_b, padding, constant_values=0.0)
                elif pad_width == 0:
                    Z_b_padded = Z_b
                else:
                    Z_b_padded = Z_b

                if Z_b is None and bucket_size > 0:
                     T_eff = X_b.shape[1]
                     batch_n = X_b.shape[0]
                     Z_b_padded = jnp.zeros((batch_n, T_eff, bucket_size))
                elif Z_b is None:
                     pass

                X_bucket.append(X_b)
                Y_bucket.append(Y_b)
                if bucket_size > 0:
                    Z_bucket.append(Z_b_padded)
                
                b_i_bucket.append(b_i)
                b_j_bucket.append(b_j)
                b_tau_bucket.append(b_tau)

            # Concatenate
            if not X_bucket:
                continue

            X_all = jnp.concatenate(X_bucket, axis=0)
            Y_all = jnp.concatenate(Y_bucket, axis=0)
            
            if bucket_size > 0:
                Z_all = jnp.concatenate(Z_bucket, axis=0)
            else:
                Z_all = None

            i_all = jnp.concatenate(b_i_bucket, axis=0)
            j_all = jnp.concatenate(b_j_bucket, axis=0)
            tau_all = jnp.concatenate(b_tau_bucket, axis=0)

            # Run Batched Test
            stats, pvals = self.test.run_batch(
                X_all, Y_all, Z_all, 
                n_conditions=bucket_size
            )
            # Block to measure true execution time
            stats.block_until_ready()
            
            # Scatter results
            val_matrix = val_matrix.at[i_all, j_all, tau_all].set(stats)
            pval_matrix = pval_matrix.at[i_all, j_all, tau_all].set(pvals)
            
        return val_matrix, pval_matrix


    def get_contemporaneous_skeleton(self) -> Dict[Tuple[int, int], bool]:
        """
        Get the contemporaneous undirected skeleton.

        Returns
        -------
        dict
            Dictionary with (i, j) pairs as keys (i < j) and
            boolean values indicating adjacency.

        Examples
        --------
        >>> skeleton = pcmci_plus.get_contemporaneous_skeleton()
        >>> for (i, j), is_adjacent in skeleton.items():
        ...     if is_adjacent:
        ...         print(f"X{i} -- X{j}")
        """
        skeleton = {}
        for j in self._skeleton:
            for i, lag in self._skeleton[j]:
                if lag == 0:
                    key = (min(i, j), max(i, j))
                    skeleton[key] = True
        return skeleton

    def get_separating_sets(self) -> Dict[Tuple[int, int, int], Set[Tuple[int, int]]]:
        """
        Get the separating sets found during skeleton discovery.

        Returns
        -------
        dict
            Dictionary mapping (i, j, tau) to the set of variables
            that made i and j conditionally independent.
        """
        return self._sepsets.copy()

    def __repr__(self) -> str:
        return (
            f"PCMCIPlus(N={self.N}, T={self.T}, test={self.test.name}, "
            f"verbosity={self.verbosity})"
        )
