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

    def run(
        self,
        tau_max: int = 1,
        tau_min: int = 0,  # PCMCI+ defaults to including tau=0
        pc_alpha: Optional[float] = 0.05,
        max_conds_dim: Optional[int] = None,
        max_conds_py: Optional[int] = None,
        max_conds_px: Optional[int] = None,
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

        self._skeleton, self._sepsets = self._discover_skeleton(
            tau_max=tau_max,
            tau_min=tau_min,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
        )

        # Phase 2: Orientation
        if self.verbosity >= 1:
            print(f"\n{'─'*60}")
            print("Phase 2: Edge Orientation")
            print(f"{'─'*60}")

        oriented_graph = self._orient_edges(
            skeleton=self._skeleton,
            sepsets=self._sepsets,
            tau_max=tau_max,
            orientation_alpha=orientation_alpha,
        )

        # Phase 3: MCI tests
        if self.verbosity >= 1:
            print(f"\n{'─'*60}")
            print("Phase 3: MCI Tests")
            print(f"{'─'*60}")

        val_matrix, pval_matrix = self._run_mci_plus(
            oriented_graph=oriented_graph,
            tau_max=tau_max,
            tau_min=tau_min,
            max_conds_py=max_conds_py,
            max_conds_px=max_conds_px,
        )

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
    ) -> Tuple[Dict[int, Set[Tuple[int, int]]], Dict]:
        """
        Discover the skeleton (undirected graph) using PC-stable.

        This phase identifies which pairs of variables are adjacent
        (have an edge between them) without determining direction.
        """
        skeleton: Dict[int, Set[Tuple[int, int]]] = {}
        sepsets: Dict[Tuple[int, int, int], Set[Tuple[int, int]]] = {}

        # Initialize with all possible edges
        for j in self.selected_variables:
            skeleton[j] = set()
            for i in range(self.N):
                for tau in range(tau_min, tau_max + 1):
                    if tau == 0 and i >= j:
                        # For contemporaneous, only add i < j to avoid duplicates
                        continue
                    if tau == 0 and i == j:
                        continue
                    skeleton[j].add((i, -tau))

            # For contemporaneous links from j to others (j < i)
            for i in range(j + 1, self.N):
                skeleton[j].add((i, 0))

        # PC-stable iteration
        max_dim = max_conds_dim if max_conds_dim is not None else self.N * (tau_max + 1)

        for cond_dim in range(max_dim + 1):
            if self.verbosity >= 2:
                print(f"  Conditioning set size: {cond_dim}")

            skeleton_snapshot = {j: skeleton[j].copy() for j in self.selected_variables}
            any_removed = False

            # For cond_dim=0, use batched testing if available
            if cond_dim == 0 and hasattr(self.test, 'run_batch'):
                # Batch test all edges at once grouped by lag
                for j in self.selected_variables:
                    current_adj = list(skeleton_snapshot[j])
                    if not current_adj:
                        continue
                    
                    # Group by lag for proper batching
                    edges_by_lag: Dict[int, List[Tuple[int, int]]] = {}
                    for adj in current_adj:
                        i, neg_tau = adj
                        tau = -neg_tau
                        if tau not in edges_by_lag:
                            edges_by_lag[tau] = []
                        edges_by_lag[tau].append(adj)
                    
                    edges_to_remove = []
                    
                    for tau, edges_at_tau in edges_by_lag.items():
                        i_list = [adj[0] for adj in edges_at_tau]
                        i_arr = jnp.asarray(i_list, dtype=jnp.int32)
                        j_arr = jnp.full((len(edges_at_tau),), j, dtype=jnp.int32)
                        tau_arr = jnp.full((len(edges_at_tau),), tau, dtype=jnp.int32)
                        
                        X_batch, Y_batch, _ = self.datahandler.get_variable_pair_batch(
                            i_arr, j_arr, tau_arr, max_lag=tau
                        )
                        
                        stats, pvals = self.test.run_batch(X_batch, Y_batch, None, alpha=pc_alpha)
                        
                        # Vectorized: find independent edges (pvalue > alpha)
                        independent_mask = np.asarray(pvals > pc_alpha)
                        for idx in np.where(independent_mask)[0]:
                            edges_to_remove.append(edges_at_tau[idx])
                            sepsets[(edges_at_tau[idx][0], j, tau)] = set()
                            sepsets[(j, edges_at_tau[idx][0], -tau)] = set()
                            any_removed = True
                    
                    for edge in edges_to_remove:
                        skeleton[j].discard(edge)
            else:
                # Cond_dim > 0: Optimize with batching if available
                if hasattr(self.test, 'run_batch'):
                    specs_by_lag = {}

                    for j in self.selected_variables:
                        current_adj = list(skeleton_snapshot[j])
                        if len(current_adj) <= cond_dim:
                            continue

                        for adj in current_adj:
                            i, neg_tau = adj
                            tau = -neg_tau
                            other_adj = [a for a in current_adj if a != adj]

                            if len(other_adj) < cond_dim:
                                continue

                            # Test with ALL subsets of size cond_dim (PC-stable standard)
                            # Note: For very large sets, this could explode, but typical PCMCI usage involves
                            # sparse graphs or limited max_conds_dim.
                            for subset in combinations(other_adj, cond_dim):
                                max_lag_in_subset = max(-p[1] for p in subset) if subset else 0
                                effective_max_lag = max(tau, max_lag_in_subset)
                                if effective_max_lag not in specs_by_lag:
                                    specs_by_lag[effective_max_lag] = []
                                specs_by_lag[effective_max_lag].append((j, adj, tau, subset))

                    # Process batches grouped by max_lag
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
                            j_list, adj_edges, tau_list, subset_list = zip(*batch_specs)

                            i_list = [p[0] for p in adj_edges]
                            i_arr = jnp.asarray(i_list, dtype=jnp.int32)
                            j_arr = jnp.asarray(j_list, dtype=jnp.int32)
                            tau_arr = jnp.asarray(tau_list, dtype=jnp.int32)

                            # Handle subsets -> numpy for speed
                            subset_arr = np.array(subset_list, dtype=np.int32)
                            cond_vars = jnp.array(subset_arr[:, :, 0], dtype=jnp.int32)
                            cond_lags = jnp.array(-subset_arr[:, :, 1], dtype=jnp.int32)

                            X_b, Y_b, Z_b = self.datahandler.get_variable_pair_batch(
                                i_arr, j_arr, tau_arr, cond_vars, cond_lags, max_lag=max_lag
                            )

                            stats, pvals = self.test.run_batch(X_b, Y_b, Z_b, alpha=pc_alpha)

                            # Vectorized: find independent pairs (pvalue > alpha)
                            independent_mask = np.asarray(pvals > pc_alpha)
                            for idx in np.where(independent_mask)[0]:
                                j_idx = j_list[idx]
                                adj_edge = adj_edges[idx]
                                tau_val = tau_list[idx]
                                subset_val = subset_list[idx]

                                if adj_edge in skeleton[j_idx]:
                                    skeleton[j_idx].discard(adj_edge)
                                    # Store separating set
                                    sepsets[(adj_edge[0], j_idx, tau_val)] = set(subset_val)
                                    sepsets[(j_idx, adj_edge[0], -tau_val)] = set(subset_val)
                                    any_removed = True
                else:
                    # Original sequential code path fallback
                    for j in self.selected_variables:
                        current_adj = list(skeleton_snapshot[j])

                        if len(current_adj) <= cond_dim:
                            continue

                        edges_to_remove = []

                        for adj in current_adj:
                            i, neg_tau = adj
                            tau = -neg_tau

                            # Get other adjacent nodes as potential conditioning set
                            other_adj = [a for a in current_adj if a != adj]

                            # Test with subsets of size cond_dim
                            independent, sep_set = self._test_independence_with_subsets(
                                i=i,
                                j=j,
                                tau=tau,
                                other_adj=other_adj,
                                cond_dim=cond_dim,
                                pc_alpha=pc_alpha,
                            )

                            if independent:
                                edges_to_remove.append(adj)
                                # Store separating set
                                sepsets[(i, j, tau)] = sep_set
                                sepsets[(j, i, -tau)] = sep_set  # Symmetric
                                any_removed = True

                        for edge in edges_to_remove:
                            skeleton[j].discard(edge)

            if not any_removed:
                break

        # Count edges
        n_lagged = sum(1 for j in skeleton for e in skeleton[j] if e[1] != 0)
        n_contemp = sum(1 for j in skeleton for e in skeleton[j] if e[1] == 0)
        if self.verbosity >= 1:
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

        for subset in combinations(other_adj, cond_dim):
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
        graph = jnp.zeros((self.N, self.N, tau_max + 1), dtype=jnp.int32)

        # Step 1: Add all skeleton edges with initial marks
        for j in skeleton:
            for i, neg_tau in skeleton[j]:
                tau = -neg_tau

                if tau > 0:
                    # Lagged link: definitely i(t-tau) -> j(t)
                    # Mark: arrow at j (2), tail at i (--> not represented for lagged)
                    graph = graph.at[i, j, tau].set(2)  # Arrow at j
                elif tau == 0:
                    # Contemporaneous: initially undirected (circle-circle)
                    graph = graph.at[i, j, 0].set(3)  # Circle
                    graph = graph.at[j, i, 0].set(3)  # Circle (symmetric)

        # Step 2: Orient v-structures for contemporaneous edges
        graph = self._orient_v_structures(graph, skeleton, sepsets)

        # Step 3: Apply Meek's orientation rules until no changes
        graph = self._apply_meek_rules(graph, skeleton, tau_max)

        # Convert marks to final directed graph
        # For visualization: 2 = arrow means there IS a directed edge
        final_graph = (graph == 2).astype(jnp.int32)

        return final_graph

    def _orient_v_structures(
        self,
        graph: jax.Array,
        skeleton: Dict,
        sepsets: Dict,
    ) -> jax.Array:
        """
        Orient v-structures (colliders) at tau=0.

        If X - Z - Y (X and Y not adjacent) and Z not in sepset(X,Y),
        then X -> Z <- Y.
        
        Optimized: Collects all updates and applies them in batch.
        """
        N = self.N
        
        # Collect all v-structure updates to batch them
        updates_arrow = []  # List of (i, j) to set as 2 (arrow)
        updates_remove = []  # List of (i, j) to set as 0 (remove)

        for z in range(N):
            # Find all contemporaneous neighbors of z
            neighbors = []
            for j in skeleton:
                for adj, lag in skeleton[j]:
                    if j == z and lag == 0:
                        neighbors.append(adj)
                    if adj == z and lag == 0:
                        neighbors.append(j)
            neighbors = list(set(neighbors))

            # Check each pair of neighbors
            for idx1, x in enumerate(neighbors):
                for y in neighbors[idx1 + 1:]:
                    # Check if x and y are NOT adjacent
                    x_adj_to_y = (y, 0) in skeleton.get(x, set()) or (x, 0) in skeleton.get(y, set())

                    if not x_adj_to_y:
                        # Check if z is in the separating set
                        sep_key = (min(x, y), max(x, y), 0)
                        sep_set = sepsets.get(sep_key, set())

                        z_in_sepset = any(var == z and lag == 0 for var, lag in sep_set)

                        if not z_in_sepset:
                            # Orient as X -> Z <- Y (v-structure)
                            updates_arrow.append((x, z))
                            updates_arrow.append((y, z))
                            updates_remove.append((z, x))
                            updates_remove.append((z, y))

                            if self.verbosity >= 2:
                                print(f"  V-structure: X{x} -> X{z} <- X{y}")

        # Apply all updates using numpy for efficiency (avoid repeated .at[].set())
        if updates_arrow or updates_remove:
            graph_np = np.array(graph)
            for i, j in updates_arrow:
                graph_np[i, j, 0] = 2
            for i, j in updates_remove:
                graph_np[i, j, 0] = 0
            graph = jnp.array(graph_np)

        return graph

    def _apply_meek_rules(
        self,
        graph: jax.Array,
        skeleton: Dict,
        tau_max: int,
        max_iterations: int = 100,
    ) -> jax.Array:
        """
        Apply Meek's orientation rules to propagate edge directions.

        Rules:
        R1: X -> Y - Z  =>  X -> Y -> Z  (if X and Z not adjacent)
        R2: X -> Z -> Y and X - Y  =>  X -> Y
        R3: X - Z -> Y <- W - X  and X - Y  =>  X -> Y
        R4: X - Z -> Y and W -> Z <- X  and X - Y  =>  X -> Y
        
        Optimized: Uses numpy for graph operations since N is typically small
        and avoids creating new JAX arrays in the inner loop.
        """
        # Convert to numpy for faster in-place updates (small N)
        graph_np = np.array(graph)
        N = self.N
        
        for iteration in range(max_iterations):
            changed = False

            # R1: Chain rule - vectorized check
            # Find all (x, y) where X -> Y (graph[x,y,0] == 2)
            directed_xy = (graph_np[:, :, 0] == 2)
            # Find all (y, z) where Y - Z (graph[y,z,0] == 3)
            undirected_yz = (graph_np[:, :, 0] == 3)
            
            for x in range(N):
                for y in range(N):
                    if directed_xy[x, y]:
                        for z in range(N):
                            if undirected_yz[y, z]:
                                # Check X not adjacent to Z
                                x_adj_z = graph_np[x, z, 0] != 0 or graph_np[z, x, 0] != 0
                                if not x_adj_z:
                                    graph_np[y, z, 0] = 2  # Y -> Z
                                    graph_np[z, y, 0] = 0
                                    changed = True

            # R2: Acyclicity rule
            undirected_xy = (graph_np[:, :, 0] == 3)
            directed = (graph_np[:, :, 0] == 2)
            
            for x in range(N):
                for y in range(N):
                    if undirected_xy[x, y]:
                        # Check if exists z such that X -> Z -> Y
                        for z in range(N):
                            if directed[x, z] and directed[z, y]:
                                graph_np[x, y, 0] = 2  # X -> Y
                                graph_np[y, x, 0] = 0
                                changed = True
                                break  # No need to check more z

            if not changed:
                break

        return jnp.array(graph_np)

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
        """
        val_matrix = jnp.zeros((self.N, self.N, tau_max + 1))
        pval_matrix = jnp.ones((self.N, self.N, tau_max + 1))

        # Build parents from oriented graph
        parents: Dict[int, Set[Tuple[int, int]]] = {}
        for j in range(self.N):
            parents[j] = set()
            for i in range(self.N):
                for tau in range(tau_max + 1):
                    if oriented_graph[i, j, tau]:
                        parents[j].add((i, -tau))

        # Run MCI tests (batched when available)
        if hasattr(self.test, "run_batch"):
            tests_by_shape: Dict[Tuple[int, int], List[Tuple]] = {}

            for j in self.selected_variables:
                for i in range(self.N):
                    for tau in range(tau_min, tau_max + 1):
                        if tau == 0 and i == j:
                            continue

                        cond_set = self._get_mci_conditions(
                            i, j, tau, parents, max_conds_py, max_conds_px
                        )
                        n_cond = len(cond_set)

                        if cond_set:
                            max_cond_lag = max(lag for _, lag in cond_set)
                            effective_max_lag = max(tau, max_cond_lag)
                        else:
                            effective_max_lag = tau

                        key = (n_cond, effective_max_lag)
                        if key not in tests_by_shape:
                            tests_by_shape[key] = []
                        tests_by_shape[key].append((i, j, tau, cond_set))

            if self.verbosity >= 1:
                iterable = tqdm(tests_by_shape.items(), desc="MCI+ batches", leave=False)
            else:
                iterable = tests_by_shape.items()

            for (n_cond, max_lag), tests in iterable:
                if len(tests) == 0:
                    continue

                effective_T = self.T - max_lag
                batch_size = self._get_effective_batch_size(
                    n_samples=effective_T, n_conditions=n_cond
                )
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

                    stats, pvals = self.test.run_batch(X_arr, Y_arr, Z_arr)
                    val_matrix = val_matrix.at[i_arr, j_arr, tau_arr].set(stats)
                    pval_matrix = pval_matrix.at[i_arr, j_arr, tau_arr].set(pvals)
        else:
            tests = []
            for j in self.selected_variables:
                for i in range(self.N):
                    for tau in range(tau_min, tau_max + 1):
                        if tau == 0 and i == j:
                            continue
                        tests.append((i, j, tau))

            if self.verbosity >= 1:
                tests = tqdm(tests, desc="MCI+ tests", leave=False)

            for i, j, tau in tests:
                cond_set = self._get_mci_conditions(
                    i, j, tau, parents, max_conds_py, max_conds_px
                )

                if cond_set:
                    X, Y, Z = self.datahandler.get_variable_pair_data(i, j, tau, cond_set)
                else:
                    X, Y, Z = self.datahandler.get_variable_pair_data(i, j, tau, None)
                    Z = None

                result = self.test.run(X, Y, Z)

                val_matrix = val_matrix.at[i, j, tau].set(result.statistic)
                pval_matrix = pval_matrix.at[i, j, tau].set(result.pvalue)

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
