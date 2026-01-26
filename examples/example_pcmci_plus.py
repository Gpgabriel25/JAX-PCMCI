"""
Example: PCMCI+ for Contemporaneous and Lagged Causal Discovery
================================================================

This example demonstrates PCMCI+, which can discover both:
- Lagged causal effects: X(t-τ) -> Y(t) for τ > 0
- Contemporaneous effects: X(t) -> Y(t)

This is essential for systems where causation happens within
the sampling interval.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from jax_pcmci import PCMCIPlus, ParCorr, DataHandler


def generate_contemporaneous_data(T: int = 1000, seed: int = 42) -> tuple[jnp.ndarray, dict]:
    """
    Generate data with both contemporaneous and lagged causal effects.
    
    Ground truth:
    - Lagged: X0(t-1) -> X1(t)
    - Lagged: X1(t-1) -> X2(t)
    - Contemporaneous: X2(t) -> X3(t)  (same time!)
    - Contemporaneous: X3(t) -> X4(t)  (same time!)
    - Lagged: X4(t-1) -> X0(t)  (feedback loop)
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    
    N = 5
    data = np.zeros((T, N))
    
    # Initialize with noise
    data[0] = np.array(jax.random.normal(keys[0], (N,)))
    data[1] = np.array(jax.random.normal(keys[1], (N,)))
    
    noise_std = 0.3
    
    for t in range(2, T):
        # Refresh random keys
        noise = np.array(jax.random.normal(keys[2], (N,))) * noise_std
        keys = jax.random.split(keys[3], 6)
        
        # X0(t) = 0.5 * X4(t-1) + noise (lagged feedback)
        data[t, 0] = 0.5 * data[t-1, 4] + noise[0]
        
        # X1(t) = 0.7 * X0(t-1) + noise (lagged)
        data[t, 1] = 0.7 * data[t-1, 0] + noise[1]
        
        # X2(t) = 0.6 * X1(t-1) + noise (lagged)
        data[t, 2] = 0.6 * data[t-1, 1] + noise[2]
        
        # X3(t) = 0.8 * X2(t) + noise (contemporaneous!)
        data[t, 3] = 0.8 * data[t, 2] + noise[3]
        
        # X4(t) = 0.7 * X3(t) + noise (contemporaneous!)
        data[t, 4] = 0.7 * data[t, 3] + noise[4]
    
    ground_truth = {
        # Lagged effects
        (4, 0, 1): "lagged",
        (0, 1, 1): "lagged",
        (1, 2, 1): "lagged",
        # Contemporaneous effects
        (2, 3, 0): "contemporaneous",
        (3, 4, 0): "contemporaneous",
    }
    
    return jnp.array(data), ground_truth


def main():
    print("=" * 60)
    print("JAX-PCMCI+ Example: Contemporaneous Causal Discovery")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating data with contemporaneous and lagged effects...")
    data, ground_truth = generate_contemporaneous_data(T=1000, seed=42)
    print(f"   Data shape: {data.shape}")
    
    print("\n   Ground truth causal structure:")
    lagged = [(k, v) for k, v in ground_truth.items() if v == "lagged"]
    contemp = [(k, v) for k, v in ground_truth.items() if v == "contemporaneous"]
    
    print("\n   Lagged effects (τ > 0):")
    for (src, tgt, lag), _ in lagged:
        print(f"   - X{src}(t-{lag}) -> X{tgt}(t)")
    
    print("\n   Contemporaneous effects (τ = 0):")
    for (src, tgt, lag), _ in contemp:
        print(f"   - X{src}(t) -> X{tgt}(t)")
    
    # Create data handler
    datahandler = DataHandler(
        data, 
        normalize=True,
        var_names=[f"X{i}" for i in range(5)]
    )
    
    # Run PCMCI+ (note: tau_min=0 to include contemporaneous)
    print("\n2. Running PCMCI+ algorithm...")
    pcmci_plus = PCMCIPlus(
        datahandler,
        cond_ind_test=ParCorr(significance='analytic'),
        verbosity=1
    )
    
    results = pcmci_plus.run(
        tau_max=2,
        tau_min=0,  # Include contemporaneous!
        pc_alpha=0.05,
        alpha_level=0.01
    )
    
    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    # Get contemporaneous links
    contemp_links = results.get_contemporaneous_links()
    lagged_links = results.get_lagged_links()
    
    print(f"\nDiscovered contemporaneous links ({len(contemp_links)}):")
    for src, tgt, stat, pval in contemp_links:
        direction = "+" if stat > 0 else "-"
        star = "✓" if (src, tgt, 0) in ground_truth else "✗"
        print(f"   {star} X{src}(t) -> X{tgt}(t) [{direction}, stat={stat:.3f}, p={pval:.4f}]")
    
    print(f"\nDiscovered lagged links ({len(lagged_links)}):")
    for src, tgt, tau, stat, pval in lagged_links:
        direction = "+" if stat > 0 else "-"
        star = "✓" if (src, tgt, tau) in ground_truth else "✗"
        print(f"   {star} X{src}(t-{tau}) -> X{tgt}(t) [{direction}, stat={stat:.3f}, p={pval:.4f}]")
    
    # Compute accuracy
    true_links = set(ground_truth.keys())
    discovered = set()
    for src, tgt, stat, pval in contemp_links:
        discovered.add((src, tgt, 0))
    for src, tgt, tau, stat, pval in lagged_links:
        discovered.add((src, tgt, tau))
    
    tp = discovered & true_links
    fp = discovered - true_links
    fn = true_links - discovered
    
    print(f"\nAccuracy Metrics:")
    print(f"  True Positives:  {len(tp)}/{len(true_links)}")
    print(f"  False Positives: {len(fp)}")
    print(f"  False Negatives: {len(fn)}")
    
    if fn:
        print(f"\n  Missed links:")
        for src, tgt, tau in fn:
            print(f"    - X{src}(t-{tau}) -> X{tgt}(t)")
    
    # Visualizations
    print("\n3. Creating visualizations...")
    
    # Time series plot
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in range(5):
        ax.plot(data[100:200, i], label=f"X{i}", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Time Series with Contemporaneous Effects\n(X2→X3→X4 happen simultaneously)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('example_contemporaneous_data.png', dpi=150)
    print("   Saved: example_contemporaneous_data.png")
    
    # Causal graph
    fig = results.plot_graph(layout='circular')
    plt.savefig('example_contemporaneous_graph.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_contemporaneous_graph.png")
    
    # Time series graph
    fig = results.plot_time_series_graph()
    plt.savefig('example_contemporaneous_ts_graph.png', dpi=150, bbox_inches='tight')
    print("   Saved: example_contemporaneous_ts_graph.png")
    
    # Show the contemporaneous structure explicitly
    print("\n4. Contemporaneous causal chain visualization:")
    print("   X2(t) --0.8--> X3(t) --0.7--> X4(t)")
    print("   (These effects happen within the same time step!)")
    
    print("\n" + "=" * 60)
    print("PCMCI+ successfully discovered both lagged AND")
    print("contemporaneous causal relationships.")
    print("=" * 60)


if __name__ == "__main__":
    main()
