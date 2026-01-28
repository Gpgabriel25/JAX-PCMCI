"""
Example: Basic PCMCI with Linear VAR Process
=============================================

This example demonstrates how to use JAX-PCMCI to discover causal
relationships in a simulated linear VAR (Vector Autoregressive) process.

The ground truth causal structure is known, allowing us to verify
the accuracy of the discovered links.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)

from jax_pcmci import PCMCI, ParCorr, DataHandler, set_device, get_device_info


def generate_linear_var_data(
    T: int = 1000,
    seed: int = 42,
) -> tuple[jnp.ndarray, dict]:
    """
    Generate data from a linear VAR process with known causal structure.
    
    Ground truth causal links:
    - X0(t-1) -> X0(t) (autocorrelation)
    - X0(t-1) -> X1(t) (cross-lag effect)
    - X1(t-2) -> X2(t) (delayed effect)
    - X2(t-1) -> X3(t)
    - X3(t-1) -> X4(t)
    """
    key = jax.random.PRNGKey(seed)
    N = 5
    
    # Initialize data
    data = np.zeros((T, N))
    noise = np.array(jax.random.normal(key, (T, N))) * 0.5
    
    # Define VAR coefficients (ground truth)
    # Format: (source, target, lag, coefficient)
    ground_truth = {
        (0, 0, 1): 0.7,   # X0(t-1) -> X0(t), coef=0.7
        (0, 1, 1): 0.5,   # X0(t-1) -> X1(t), coef=0.5
        (1, 2, 2): 0.6,   # X1(t-2) -> X2(t), coef=0.6
        (2, 3, 1): 0.8,   # X2(t-1) -> X3(t), coef=0.8
        (3, 4, 1): 0.4,   # X3(t-1) -> X4(t), coef=0.4
    }
    
    # Generate time series
    for t in range(2, T):
        for j in range(N):
            data[t, j] = noise[t, j]
            
            for (src, tgt, lag), coef in ground_truth.items():
                if tgt == j:
                    data[t, j] += coef * data[t - lag, src]
    
    return jnp.array(data), ground_truth


def main():
    print("=" * 60)
    print("JAX-PCMCI Example: Linear VAR Process")
    print("=" * 60)
    
    # Check device info
    info = get_device_info()
    print(f"\nDevice: {info['default_backend']}")
    print(f"GPUs available: {info['gpu_count']}")
    
    # Generate data
    print("\n1. Generating data from linear VAR process...")
    data, ground_truth = generate_linear_var_data(T=1000, seed=42)
    print(f"   Data shape: {data.shape} (T={data.shape[0]}, N={data.shape[1]})")
    
    # Display ground truth
    print("\n   Ground truth causal links:")
    for (src, tgt, lag), coef in ground_truth.items():
        print(f"   - X{src}(t-{lag}) -> X{tgt}(t) [coef={coef}]")
    
    # Create data handler
    print("\n2. Creating data handler...")
    datahandler = DataHandler(
        data,
        normalize=True,
        var_names=[f"X{i}" for i in range(5)]
    )
    print(f"   Normalized: {datahandler.is_normalized}")
    
    # Initialize PCMCI with ParCorr test
    print("\n3. Initializing PCMCI with ParCorr test...")
    pcmci = PCMCI(
        datahandler,
        cond_ind_test=ParCorr(significance='analytic'),
        verbosity=1
    )
    
    # Run PCMCI
    print("\n4. Running PCMCI algorithm...")
    results = pcmci.run(
        tau_max=3,
        tau_min=1,
        pc_alpha=0.05,
        alpha_level=0.01,
        fdr_method='fdr_bh'
    )
    
    # Evaluate results
    print("\n" + "=" * 60)
    print("RESULTS EVALUATION")
    print("=" * 60)
    
    # Compare with ground truth
    discovered = set()
    for src, tgt, tau, stat, pval in results.significant_links:
        discovered.add((src, tgt, tau))
    
    true_links = set((src, tgt, lag) for (src, tgt, lag) in ground_truth.keys())
    
    # True positives
    tp = discovered & true_links
    # False positives
    fp = discovered - true_links
    # False negatives
    fn = true_links - discovered
    
    print(f"\nTrue Positives ({len(tp)}/{len(true_links)}):")
    for link in sorted(tp):
        src, tgt, tau = link
        stat = float(results.val_matrix[src, tgt, tau])
        print(f"  ✓ X{src}(t-{tau}) -> X{tgt}(t) [stat={stat:.3f}]")
    
    if fp:
        print(f"\nFalse Positives ({len(fp)}):")
        for link in sorted(fp):
            src, tgt, tau = link
            stat = float(results.val_matrix[src, tgt, tau])
            print(f"  ✗ X{src}(t-{tau}) -> X{tgt}(t) [stat={stat:.3f}]")
    
    if fn:
        print(f"\nFalse Negatives ({len(fn)}):")
        for link in sorted(fn):
            src, tgt, tau = link
            print(f"  ✗ X{src}(t-{tau}) -> X{tgt}(t) [missed]")
    
    # Compute metrics
    precision = len(tp) / len(discovered) if discovered else 0
    recall = len(tp) / len(true_links) if true_links else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    
    # Plot results
    print("\n5. Creating visualizations...")
    
    fig = results.plot_graph(layout='circular')
    plt.savefig('example_linear_var_graph.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_linear_var_graph.png")
    
    fig = results.plot_time_series_graph()
    plt.savefig('example_linear_var_ts_graph.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_linear_var_ts_graph.png")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
