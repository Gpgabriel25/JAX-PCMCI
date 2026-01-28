"""
Results Handling for JAX-PCMCI
==============================

This module provides classes for storing, analyzing, and visualizing
the results of PCMCI and PCMCI+ analyses.

Key Features
------------
- Comprehensive result storage with metadata
- Multiple significance testing with FDR correction
- Graph visualization and export
- Summary statistics and reporting

Example
-------
>>> results = pcmci.run(tau_max=3)
>>>
>>> # Get summary
>>> print(results.summary())
>>>
>>> # Visualize
>>> results.plot_graph()
>>> results.plot_time_series_graph()
>>>
>>> # Export
>>> results.to_networkx()
>>> results.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class PCMCIResults:
    """
    Container for PCMCI/PCMCI+ analysis results.

    This class stores all results from a PCMCI analysis and provides
    methods for analysis, visualization, and export.

    Parameters
    ----------
    val_matrix : jax.Array
        Test statistic values, shape (N, N, tau_max + 1).
        Entry [i, j, tau] is the statistic for link i(t-tau) -> j(t).
    pval_matrix : jax.Array
        P-values, shape (N, N, tau_max + 1).
    var_names : list of str
        Variable names.
    alpha_level : float, default=0.05
        Significance level for determining significant links.
    fdr_method : str or None
        FDR correction method ('fdr_bh', 'bonferroni', or None).
    test_name : str
        Name of the independence test used.
    tau_max : int
        Maximum time lag tested.
    tau_min : int
        Minimum time lag tested.
    oriented_graph : jax.Array, optional
        For PCMCI+, the oriented causal graph.
    conf_matrix : jax.Array, optional
        Confidence intervals for test statistics.

    Attributes
    ----------
    graph : jax.Array
        Binary adjacency matrix of significant links.
    significant_links : list
        List of significant (source, target, lag, stat, pval) tuples.
    n_significant_links : int
        Number of significant causal links found.

    Examples
    --------
    >>> results = pcmci.run(tau_max=3)
    >>>
    >>> # Access raw matrices
    >>> print(results.val_matrix.shape)  # (N, N, tau_max + 1)
    >>>
    >>> # Get significant links
    >>> for link in results.significant_links:
    ...     src, tgt, tau, stat, pval = link
    ...     print(f"X{src}(t-{tau}) -> X{tgt}(t): stat={stat:.3f}, p={pval:.4f}")
    >>>
    >>> # Check specific link
    >>> is_sig = results.is_significant(source=0, target=1, lag=2)
    """

    val_matrix: jax.Array
    pval_matrix: jax.Array
    var_names: List[str]
    alpha_level: float = 0.05
    fdr_method: Optional[str] = None
    test_name: str = "Unknown"
    tau_max: int = 1
    tau_min: int = 0
    oriented_graph: Optional[jax.Array] = None
    conf_matrix: Optional[jax.Array] = None
    _graph: Optional[jax.Array] = field(default=None, repr=False)
    _adjusted_pvalues: Optional[jax.Array] = field(default=None, repr=False)

    def __post_init__(self):
        """Compute derived attributes."""
        self._compute_graph()

    @property
    def N(self) -> int:
        """Number of variables."""
        return self.val_matrix.shape[0]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of result matrices (N, N, tau_max + 1)."""
        return self.val_matrix.shape

    @property
    def graph(self) -> jax.Array:
        """
        Binary adjacency matrix of significant links.

        Returns
        -------
        jax.Array
            Shape (N, N, tau_max + 1) where entry [i, j, tau] is 1
            if the link i(t-tau) -> j(t) is significant.
        """
        if self._graph is None:
            self._compute_graph()
        return self._graph

    @property
    def adjusted_pvalues(self) -> jax.Array:
        """
        P-values after FDR/multiple testing correction.

        Returns
        -------
        jax.Array
            Adjusted p-values with same shape as pval_matrix.
        """
        if self._adjusted_pvalues is None:
            self._compute_graph()
        return self._adjusted_pvalues

    def _compute_graph(self) -> None:
        """Compute the significant link graph with optional FDR correction."""
        pvals = np.array(self.pval_matrix)

        if self.fdr_method is None:
            # No correction
            self._adjusted_pvalues = jnp.array(pvals)
            graph = (pvals < self.alpha_level).astype(np.int32)

        elif self.fdr_method == "bonferroni":
            # Bonferroni correction
            n_tests = pvals.size
            adjusted = np.minimum(pvals * n_tests, 1.0)
            self._adjusted_pvalues = jnp.array(adjusted)
            graph = (adjusted < self.alpha_level).astype(np.int32)

        elif self.fdr_method == "fdr_bh":
            # Benjamini-Hochberg FDR correction
            adjusted = self._benjamini_hochberg(pvals)
            self._adjusted_pvalues = jnp.array(adjusted)
            graph = (adjusted < self.alpha_level).astype(np.int32)

        else:
            raise ValueError(f"Unknown FDR method: {self.fdr_method}")

        # Set diagonal at tau=0 to zero (no self-loops at same time)
        for i in range(self.N):
            graph[i, i, 0] = 0
        
        self._graph = jnp.array(graph)

    @staticmethod
    def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR correction.

        Parameters
        ----------
        pvals : np.ndarray
            Raw p-values (any shape).

        Returns
        -------
        np.ndarray
            Adjusted p-values (same shape).
        """
        shape = pvals.shape
        pvals_flat = pvals.flatten()
        n = len(pvals_flat)

        # Sort p-values
        sorted_idx = np.argsort(pvals_flat)
        sorted_pvals = pvals_flat[sorted_idx]

        # Compute adjusted p-values (vectorized)
        # adjusted_p[i] = p[i] * n / (i+1)
        ranks = np.arange(1, n + 1)
        raw_adjusted = sorted_pvals * n / ranks
        
        # Apply cumulative minimum from the end (reverse cummin)
        # np.minimum.accumulate on reversed array
        cummin_adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
        
        # Cap at 1.0
        cummin_adjusted = np.minimum(cummin_adjusted, 1.0)
        
        # Put back in original order
        adjusted = np.empty(n)
        adjusted[sorted_idx] = cummin_adjusted

        return adjusted.reshape(shape)

    @property
    def significant_links(self) -> List[Tuple[int, int, int, float, float]]:
        """
        Get list of all significant causal links.

        Returns
        -------
        list of tuples
            Each tuple is (source, target, lag, statistic, pvalue).
            Sorted by absolute statistic value (strongest first).

        Examples
        --------
        >>> for src, tgt, tau, stat, pval in results.significant_links[:5]:
        ...     print(f"X{src}(t-{tau}) -> X{tgt}: stat={stat:.3f}")
        """
        links = []
        graph = np.array(self.graph)
        vals = np.array(self.val_matrix)
        pvals = np.array(self.adjusted_pvalues)

        for i in range(self.N):
            for j in range(self.N):
                for tau in range(self.tau_max + 1):
                    if graph[i, j, tau]:
                        links.append((i, j, tau, float(vals[i, j, tau]), float(pvals[i, j, tau])))

        # Sort by absolute statistic (strongest links first)
        links.sort(key=lambda x: abs(x[3]), reverse=True)

        return links

    @property
    def n_significant_links(self) -> int:
        """Number of significant causal links."""
        return len(self.significant_links)

    def is_significant(self, source: int, target: int, lag: int) -> bool:
        """
        Check if a specific link is significant.

        Parameters
        ----------
        source : int
            Source variable index.
        target : int
            Target variable index.
        lag : int
            Time lag.

        Returns
        -------
        bool
            True if the link is significant.

        Examples
        --------
        >>> if results.is_significant(0, 1, 2):
        ...     print("X0(t-2) -> X1(t) is significant")
        """
        return bool(self.graph[source, target, lag])

    def get_link_info(
        self, source: int, target: int, lag: int
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific link.

        Parameters
        ----------
        source : int
            Source variable index.
        target : int
            Target variable index.
        lag : int
            Time lag.

        Returns
        -------
        dict
            Dictionary with keys: 'source', 'target', 'lag', 'source_name',
            'target_name', 'statistic', 'pvalue', 'adjusted_pvalue',
            'significant'.
        """
        return {
            "source": source,
            "target": target,
            "lag": lag,
            "source_name": self.var_names[source],
            "target_name": self.var_names[target],
            "statistic": float(self.val_matrix[source, target, lag]),
            "pvalue": float(self.pval_matrix[source, target, lag]),
            "adjusted_pvalue": float(self.adjusted_pvalues[source, target, lag]),
            "significant": self.is_significant(source, target, lag),
        }

    def get_parents(self, variable: int) -> List[Tuple[int, int]]:
        """
        Get all significant parents of a variable.

        Parameters
        ----------
        variable : int
            Variable index.

        Returns
        -------
        list of (int, int)
            List of (parent_variable, lag) tuples.
        """
        parents = []
        for i in range(self.N):
            for tau in range(self.tau_max + 1):
                if self.graph[i, variable, tau]:
                    parents.append((i, tau))
        return parents

    def get_children(self, variable: int) -> List[Tuple[int, int]]:
        """
        Get all significant children (effects) of a variable.

        Parameters
        ----------
        variable : int
            Variable index.

        Returns
        -------
        list of (int, int)
            List of (child_variable, lag) tuples.
        """
        children = []
        for j in range(self.N):
            for tau in range(self.tau_max + 1):
                if self.graph[variable, j, tau]:
                    children.append((j, tau))
        return children

    def get_contemporaneous_links(self) -> List[Tuple[int, int, float, float]]:
        """
        Get contemporaneous (tau=0) links.

        Returns
        -------
        list of tuples
            Each tuple is (source, target, statistic, pvalue).
        """
        links = []
        for i in range(self.N):
            for j in range(self.N):
                if i != j and self.graph[i, j, 0]:
                    links.append((
                        i, j,
                        float(self.val_matrix[i, j, 0]),
                        float(self.adjusted_pvalues[i, j, 0])
                    ))
        return links

    def get_lagged_links(self) -> List[Tuple[int, int, int, float, float]]:
        """
        Get lagged (tau > 0) links.

        Returns
        -------
        list of tuples
            Each tuple is (source, target, lag, statistic, pvalue).
        """
        links = []
        for i in range(self.N):
            for j in range(self.N):
                for tau in range(1, self.tau_max + 1):
                    if self.graph[i, j, tau]:
                        links.append((
                            i, j, tau,
                            float(self.val_matrix[i, j, tau]),
                            float(self.adjusted_pvalues[i, j, tau])
                        ))
        return links

    def summary(self) -> str:
        """
        Generate a human-readable summary of results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            "=" * 60,
            "PCMCI Results Summary",
            "=" * 60,
            f"Test: {self.test_name}",
            f"Variables: {self.N}",
            f"Tau range: {self.tau_min} to {self.tau_max}",
            f"Alpha level: {self.alpha_level}",
            f"FDR correction: {self.fdr_method or 'None'}",
            "",
            f"Significant links found: {self.n_significant_links}",
        ]

        # Contemporaneous
        contemp = self.get_contemporaneous_links()
        if contemp:
            lines.append(f"  - Contemporaneous (tau=0): {len(contemp)}")

        # Lagged
        lagged = self.get_lagged_links()
        if lagged:
            lines.append(f"  - Lagged (tau>0): {len(lagged)}")

        lines.append("")
        lines.append("Top links by strength:")
        lines.append("-" * 60)

        for i, (src, tgt, tau, stat, pval) in enumerate(self.significant_links[:10]):
            src_name = self.var_names[src]
            tgt_name = self.var_names[tgt]
            sig_stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
            lines.append(
                f"  {src_name}(t-{tau}) -> {tgt_name}(t): "
                f"stat={stat:+.4f}, p={pval:.4f} {sig_stars}"
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    def plot_graph(
        self,
        figsize: Tuple[float, float] = (10, 10),
        node_size: int = 2000,
        font_size: int = 12,
        edge_width_scale: float = 3.0,
        show_edge_labels: bool = True,
        layout: str = "circular",
        ax=None,
        save_path: Optional[str] = None,
    ):
        """
        Plot the causal graph as a directed network.

        Parameters
        ----------
        figsize : tuple, default=(10, 10)
            Figure size in inches.
        node_size : int, default=2000
            Size of variable nodes.
        font_size : int, default=12
            Font size for labels.
        edge_width_scale : float, default=3.0
            Scaling factor for edge widths.
        show_edge_labels : bool, default=True
            Whether to show lag values on edges.
        layout : str, default='circular'
            Graph layout ('circular', 'spring', 'shell').
        ax : matplotlib Axes, optional
            Axes to plot on. Creates new figure if None.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.

        Examples
        --------
        >>> results.plot_graph(layout='spring', save_path='causal_graph.png')
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting")
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for plotting")

        # Create graph
        G = self.to_networkx()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Layout
        if layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        else:
            pos = nx.circular_layout(G)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_size,
            node_color="lightblue",
            edgecolors="black",
            linewidths=2,
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=font_size,
            font_weight="bold",
        )

        # Draw edges with width based on statistic
        edges = G.edges(data=True)
        if edges:
            edge_widths = [abs(d.get("statistic", 1.0)) * edge_width_scale for u, v, d in edges]
            edge_colors = ["red" if d.get("statistic", 0) < 0 else "blue" for u, v, d in edges]

            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=edge_widths,
                edge_color=edge_colors,
                arrows=True,
                arrowsize=20,
                connectionstyle="arc3,rad=0.1",
                alpha=0.7,
            )

            if show_edge_labels:
                edge_labels = {
                    (u, v): f"τ={d['lag']}"
                    for u, v, d in edges
                }
                nx.draw_networkx_edge_labels(
                    G, pos, ax=ax,
                    edge_labels=edge_labels,
                    font_size=font_size - 2,
                )

        ax.set_title(f"Causal Graph ({self.n_significant_links} links)", fontsize=14)
        ax.axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_time_series_graph(
        self,
        figsize: Tuple[float, float] = (12, 8),
        node_size: int = 1500,
        save_path: Optional[str] = None,
    ):
        """
        Plot the time series causal graph with explicit time lags.

        This visualization shows variables at different time points as
        separate nodes, making the temporal structure explicit.

        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size.
        node_size : int, default=1500
            Node size.
        save_path : str, optional
            Path to save figure.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting")

        fig, ax = plt.subplots(figsize=figsize)

        N = self.N
        tau_max = self.tau_max

        # Create node positions: (variable, time) -> (x, y)
        positions = {}
        for t in range(tau_max + 1):
            for v in range(N):
                x = tau_max - t  # Time flows left to right
                y = N - 1 - v  # Variables top to bottom
                positions[(v, t)] = (x, y)

        # Draw nodes
        for (v, t), (x, y) in positions.items():
            color = "lightblue" if t == 0 else "lightgray"
            circle = plt.Circle((x, y), 0.15, color=color, ec="black", lw=2)
            ax.add_patch(circle)

            label = f"{self.var_names[v]}\nt-{t}" if t > 0 else self.var_names[v]
            ax.text(x, y, label, ha="center", va="center", fontsize=9)

        # Draw edges
        for src, tgt, tau, stat, pval in self.significant_links:
            x1, y1 = positions[(src, tau)]
            x2, y2 = positions[(tgt, 0)]

            color = "blue" if stat > 0 else "red"
            width = min(abs(stat) * 3, 4)

            ax.annotate(
                "",
                xy=(x2 - 0.15, y2),
                xytext=(x1 + 0.15, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=width,
                    alpha=0.7,
                ),
            )

        ax.set_xlim(-0.5, tau_max + 0.5)
        ax.set_ylim(-0.5, N - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Time Series Causal Graph", fontsize=14)

        # Legend
        pos_patch = mpatches.Patch(color="blue", alpha=0.7, label="Positive effect")
        neg_patch = mpatches.Patch(color="red", alpha=0.7, label="Negative effect")
        ax.legend(handles=[pos_patch, neg_patch], loc="upper left")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_matrix(
        self,
        matrix: str = "val",
        figsize: Tuple[float, float] = (10, 8),
        cmap: str = "RdBu_r",
        save_path: Optional[str] = None,
    ):
        """
        Plot val_matrix or pval_matrix as a heatmap.

        Parameters
        ----------
        matrix : str, default='val'
            Which matrix to plot ('val' or 'pval').
        figsize : tuple
            Figure size.
        cmap : str
            Colormap name.
        save_path : str, optional
            Path to save figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting")

        data = self.val_matrix if matrix == "val" else self.pval_matrix
        title = "Test Statistics" if matrix == "val" else "P-values"

        n_lags = self.tau_max + 1
        fig, axes = plt.subplots(1, n_lags, figsize=figsize)

        if n_lags == 1:
            axes = [axes]

        for tau, ax in enumerate(axes):
            mat = np.array(data[:, :, tau])

            if matrix == "val":
                vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
                im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)

            ax.set_title(f"τ = {tau}")
            ax.set_xticks(range(self.N))
            ax.set_yticks(range(self.N))
            ax.set_xticklabels(self.var_names, rotation=45, ha="right")
            ax.set_yticklabels(self.var_names)

            if tau == 0:
                ax.set_ylabel("Source (cause)")
            ax.set_xlabel("Target (effect)")

        plt.colorbar(im, ax=axes, label=title)
        fig.suptitle(f"{title} Matrix", fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def to_networkx(self):
        """
        Convert results to a NetworkX DiGraph.

        Returns
        -------
        networkx.DiGraph
            Directed graph with edge attributes.

        Examples
        --------
        >>> G = results.to_networkx()
        >>> print(G.number_of_edges())
        >>> nx.write_gexf(G, "causal_graph.gexf")
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required")

        G = nx.DiGraph()

        # Add nodes
        for i, name in enumerate(self.var_names):
            G.add_node(name, index=i)

        # Add edges
        for src, tgt, tau, stat, pval in self.significant_links:
            src_name = self.var_names[src]
            tgt_name = self.var_names[tgt]
            G.add_edge(
                src_name,
                tgt_name,
                lag=tau,
                statistic=stat,
                pvalue=pval,
            )

        return G

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        dict
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "var_names": self.var_names,
            "tau_max": self.tau_max,
            "tau_min": self.tau_min,
            "alpha_level": self.alpha_level,
            "fdr_method": self.fdr_method,
            "test_name": self.test_name,
            "n_significant_links": self.n_significant_links,
            "val_matrix": np.array(self.val_matrix).tolist(),
            "pval_matrix": np.array(self.pval_matrix).tolist(),
            "graph": np.array(self.graph).tolist(),
            "significant_links": [
                {
                    "source": src,
                    "source_name": self.var_names[src],
                    "target": tgt,
                    "target_name": self.var_names[tgt],
                    "lag": tau,
                    "statistic": stat,
                    "pvalue": pval,
                }
                for src, tgt, tau, stat, pval in self.significant_links
            ],
        }

    def __repr__(self) -> str:
        return (
            f"PCMCIResults(N={self.N}, tau_max={self.tau_max}, "
            f"significant_links={self.n_significant_links})"
        )
