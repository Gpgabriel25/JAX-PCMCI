"""
Example: Nonlinear Causal Discovery with PCMCI
==============================================

This example demonstrates how to use JAX-PCMCI with nonlinear
independence tests (CMI-kNN) to discover causal relationships
in data with nonlinear dependencies.

Linear tests like ParCorr would fail to detect these relationships.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from jax_pcmci import PCMCI, ParCorr, CMIKnn, DataHandler


def generate_nonlinear_data(T: int = 500, seed: int = 42) -> tuple[jnp.ndarray, dict]:
    """
    Generate data with nonlinear causal relationships.
    
    Ground truth:
    - X0 is independent noise
    - X1(t) = sin(X0(t-1)) + noise
    - X2(t) = X1(t-1)^2 + noise  (quadratic relationship)
    - X3(t) = X0(t-2) * X1(t-1) + noise  (multiplicative interaction)
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    
    N = 4
    data = np.zeros((T, N))
    
    # X0: Independent noise
    data[:, 0] = np.array(jax.random.normal(keys[0], (T,)))
    
    # Generate with dependencies
    for t in range(2, T):
        # X1(t) = sin(X0(t-1)) + noise
        data[t, 1] = np.sin(data[t-1, 0]) + 0.3 * np.array(jax.random.normal(keys[1], ()))
        
        # X2(t) = X1(t-1)^2 + noise (nonlinear: quadratic)
        data[t, 2] = 0.5 * data[t-1, 1]**2 + 0.2 * np.array(jax.random.normal(keys[2], ()))
        
        # X3(t) = X0(t-2) * X1(t-1) + noise (nonlinear: multiplicative)
        data[t, 3] = 0.3 * data[t-2, 0] * data[t-1, 1] + 0.2 * np.array(jax.random.normal(keys[3], ()))
        
        # Refresh random keys
        keys = jax.random.split(keys[4], 5)
    
    ground_truth = {
        (0, 1, 1): "sin",        # X0(t-1) -> X1(t) via sine
        (1, 2, 1): "quadratic",  # X1(t-1) -> X2(t) via square
        (0, 3, 2): "multiplicative",  # X0(t-2) -> X3(t)
        (1, 3, 1): "multiplicative",  # X1(t-1) -> X3(t)
    }
    
    return jnp.array(data), ground_truth


def main():
    print("=" * 60)
    print("JAX-PCMCI Example: Nonlinear Causal Discovery")
    print("=" * 60)
    
    # Generate nonlinear data
    print("\n1. Generating data with nonlinear relationships...")
    data, ground_truth = generate_nonlinear_data(T=500, seed=42)
    print(f"   Data shape: {data.shape}")
    
    print("\n   Ground truth nonlinear links:")
    for (src, tgt, lag), func_type in ground_truth.items():
        print(f"   - X{src}(t-{lag}) -> X{tgt}(t) [{func_type}]")
    
    # Create data handler
    datahandler = DataHandler(data, normalize=True, var_names=[f"X{i}" for i in range(4)])
    
    # First, try with linear ParCorr (expected to miss nonlinear relationships)
    print("\n2. Running PCMCI with LINEAR test (ParCorr)...")
    print("   (Expected to miss nonlinear relationships)")
    
    pcmci_linear = PCMCI(datahandler, cond_ind_test=ParCorr(), verbosity=0)
    results_linear = pcmci_linear.run(tau_max=3, alpha_level=0.05)
    
    print(f"\n   ParCorr found {results_linear.n_significant_links} links:")
    for src, tgt, tau, stat, pval in results_linear.significant_links:
        print(f"   - X{src}(t-{tau}) -> X{tgt}(t) [stat={stat:.3f}]")
    
    # Now use nonlinear CMI-kNN test
    print("\n3. Running PCMCI with NONLINEAR test (CMI-kNN)...")
    print("   (Should detect all nonlinear relationships)")
    
    cmi_test = CMIKnn(
        k=10,
        significance='permutation',
        n_permutations=200,  # Reduced for speed; use 500+ for publication
        metric='chebyshev'
    )
    
    pcmci_nonlinear = PCMCI(datahandler, cond_ind_test=cmi_test, verbosity=1)
    results_nonlinear = pcmci_nonlinear.run(tau_max=3, alpha_level=0.05)
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON: Linear vs Nonlinear Tests")
    print("=" * 60)
    
    true_links = set(ground_truth.keys())
    
    # Linear test results
    linear_discovered = set()
    for src, tgt, tau, stat, pval in results_linear.significant_links:
        linear_discovered.add((src, tgt, tau))
    
    linear_tp = linear_discovered & true_links
    linear_fp = linear_discovered - true_links
    linear_fn = true_links - linear_discovered
    
    # Nonlinear test results
    nonlinear_discovered = set()
    for src, tgt, tau, stat, pval in results_nonlinear.significant_links:
        nonlinear_discovered.add((src, tgt, tau))
    
    nonlinear_tp = nonlinear_discovered & true_links
    nonlinear_fp = nonlinear_discovered - true_links
    nonlinear_fn = true_links - nonlinear_discovered
    
    print(f"\n{'Metric':<20} {'ParCorr':<15} {'CMI-kNN':<15}")
    print("-" * 50)
    print(f"{'True Positives':<20} {len(linear_tp):<15} {len(nonlinear_tp):<15}")
    print(f"{'False Positives':<20} {len(linear_fp):<15} {len(nonlinear_fp):<15}")
    print(f"{'False Negatives':<20} {len(linear_fn):<15} {len(nonlinear_fn):<15}")
    
    linear_recall = len(linear_tp) / len(true_links) if true_links else 0
    nonlinear_recall = len(nonlinear_tp) / len(true_links) if true_links else 0
    
    print(f"{'Recall':<20} {linear_recall:.1%}{'':9} {nonlinear_recall:.1%}")
    
    # Visualize data and relationships
    print("\n4. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot data
    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(data[100:200, i], label=f"X{i}")
        ax.set_title(f"Variable X{i}")
        ax.set_xlabel("Time")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('example_nonlinear_data.png', dpi=150)
    print("   Saved: example_nonlinear_data.png")
    
    # Plot scatter plots showing nonlinear relationships
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # X0(t-1) vs X1(t) - sine relationship
    ax = axes[0]
    ax.scatter(data[:-1, 0], data[1:, 1], alpha=0.5, s=10)
    ax.set_xlabel("X0(t-1)")
    ax.set_ylabel("X1(t)")
    ax.set_title("Sine relationship: X1 = sin(X0)")
    
    # X1(t-1) vs X2(t) - quadratic relationship  
    ax = axes[1]
    ax.scatter(data[:-1, 1], data[1:, 2], alpha=0.5, s=10)
    ax.set_xlabel("X1(t-1)")
    ax.set_ylabel("X2(t)")
    ax.set_title("Quadratic relationship: X2 = X1²")
    
    # 3D plot for multiplicative interaction
    ax = axes[2]
    ax.scatter(data[:-1, 0] * data[1:, 1], data[2:, 3][:len(data[:-1,0])-1], alpha=0.5, s=10)
    ax.set_xlabel("X0(t-2) * X1(t-1)")
    ax.set_ylabel("X3(t)")
    ax.set_title("Multiplicative: X3 = X0 * X1")
    
    plt.tight_layout()
    plt.savefig('example_nonlinear_relationships.png', dpi=150)
    print("   Saved: example_nonlinear_relationships.png")
    
    # Plot causal graph from nonlinear test
    fig = results_nonlinear.plot_graph()
    plt.savefig('example_nonlinear_graph.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_nonlinear_graph.png")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("CMI-kNN successfully detected nonlinear causal relationships")
    print("that ParCorr missed due to its linearity assumption.")
    print("=" * 60)


if __name__ == "__main__":
    main()
