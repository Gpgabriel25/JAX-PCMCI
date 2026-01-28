"""
Example: EEG Connectivity Analysis with 64 Channels
====================================================

This example demonstrates JAX-PCMCI on a realistic EEG-like dataset
with 64 channels, testing how the library handles large connectivity
matrices (64x64 = 4,096 potential connections per lag).

This is a stress test for:
- Scalability to many variables
- Memory efficiency
- GPU utilization with large matrices
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from jax_pcmci import PCMCI, PCMCIPlus, ParCorr, DataHandler, get_device_info


# Standard 10-20 system electrode names (64 channels)
EEG_CHANNELS = [
    # Frontal
    'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'AFz',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz',
    # Frontocentral
    'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz',
    # Central
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz',
    # Centroparietal
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz',
    # Parietal
    'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz',
    # Parietooccipital
    'PO3', 'PO4', 'PO7', 'PO8', 'POz',
    # Occipital
    'O1', 'O2', 'Oz',
    # Temporal
    'T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8', 'FT9', 'FT10',
]


def generate_eeg_like_data(
    n_channels: int = 64,
    n_samples: int = 1000,
    sampling_rate: int = 256,
    seed: int = 42,
    n_true_connections: int = 20,
) -> tuple[jnp.ndarray, dict, list]:
    """
    Generate synthetic EEG-like data with known causal structure.
    
    Simulates:
    - Oscillatory activity in alpha (8-12 Hz) and beta (12-30 Hz) bands
    - A sparse set of true causal connections between regions
    - Realistic noise characteristics
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Number of time samples.
    sampling_rate : int
        Sampling rate in Hz.
    seed : int
        Random seed.
    n_true_connections : int
        Number of true causal connections to embed.
        
    Returns
    -------
    data : jnp.ndarray
        EEG data of shape (n_samples, n_channels).
    ground_truth : dict
        Dictionary of true causal links.
    channel_names : list
        Names of the channels.
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)
    
    # Use standard channel names or generate generic ones
    if n_channels <= len(EEG_CHANNELS):
        channel_names = EEG_CHANNELS[:n_channels]
    else:
        channel_names = [f"Ch{i:02d}" for i in range(n_channels)]
    
    # Initialize data with noise
    data = np.zeros((n_samples, n_channels))
    
    # Add oscillatory components (alpha and beta)
    t = np.arange(n_samples) / sampling_rate
    
    for ch in range(n_channels):
        # Random alpha frequency (8-12 Hz) with random phase
        alpha_freq = 8 + 4 * np.random.rand()
        alpha_phase = 2 * np.pi * np.random.rand()
        alpha_amp = 0.3 + 0.4 * np.random.rand()
        
        # Random beta frequency (12-30 Hz)
        beta_freq = 12 + 18 * np.random.rand()
        beta_phase = 2 * np.pi * np.random.rand()
        beta_amp = 0.2 + 0.3 * np.random.rand()
        
        # Add oscillations
        data[:, ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t + alpha_phase)
        data[:, ch] += beta_amp * np.sin(2 * np.pi * beta_freq * t + beta_phase)
        
        # Add pink noise (1/f characteristic of EEG)
        noise_key = jax.random.fold_in(keys[0], ch)
        white_noise = np.array(jax.random.normal(noise_key, (n_samples,)))
        # Simple pink noise approximation
        pink_noise = np.cumsum(white_noise) / np.sqrt(np.arange(1, n_samples + 1))
        pink_noise = (pink_noise - pink_noise.mean()) / (pink_noise.std() + 1e-8)
        data[:, ch] += 0.5 * pink_noise
    
    # Create sparse causal structure
    ground_truth = {}
    np.random.seed(seed + 1)
    
    # Generate random causal connections
    for _ in range(n_true_connections):
        src = np.random.randint(0, n_channels)
        tgt = np.random.randint(0, n_channels)
        while tgt == src:
            tgt = np.random.randint(0, n_channels)
        
        lag = np.random.randint(1, 4)  # 1-3 sample lag (~4-12ms at 256Hz)
        strength = 0.3 + 0.4 * np.random.rand()  # 0.3-0.7
        
        ground_truth[(src, tgt, lag)] = strength
    
    # Embed causal relationships
    for (src, tgt, lag), strength in ground_truth.items():
        for t_idx in range(lag, n_samples):
            data[t_idx, tgt] += strength * data[t_idx - lag, src]
    
    # Normalize channels
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    return jnp.array(data), ground_truth, channel_names


def analyze_connectivity(results, ground_truth, channel_names):
    """Analyze and visualize connectivity results."""
    
    # Get significant links
    discovered = set()
    for src, tgt, tau, stat, pval in results.significant_links:
        discovered.add((src, tgt, tau))
    
    true_links = set(ground_truth.keys())
    
    # Metrics
    tp = discovered & true_links
    fp = discovered - true_links
    fn = true_links - discovered
    
    precision = len(tp) / len(discovered) if discovered else 0
    recall = len(tp) / len(true_links) if true_links else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print("CONNECTIVITY ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"\nGround Truth: {len(true_links)} causal connections")
    print(f"Discovered:   {len(discovered)} connections")
    print(f"\nTrue Positives:  {len(tp)}")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")
    print(f"\nPrecision: {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1 Score:  {f1:.1%}")
    
    # Show some true positive connections
    if tp:
        print(f"\nSample True Positive Connections:")
        for src, tgt, tau in list(tp)[:5]:
            src_name = channel_names[src]
            tgt_name = channel_names[tgt]
            strength = ground_truth[(src, tgt, tau)]
            print(f"  {src_name} -> {tgt_name} (lag={tau}, true_strength={strength:.2f})")
    
    return precision, recall, f1


def main():
    print("=" * 60)
    print("JAX-PCMCI: EEG Connectivity Analysis (64 Channels)")
    print("=" * 60)
    
    # Device info
    info = get_device_info()
    print(f"\nDevice: {info['default_backend']}")
    print(f"GPUs: {info['gpu_count']}")
    
    # Parameters - Use 32 channels for stress test
    # For larger networks, consider reducing tau_max or max_conds_dim
    n_channels = 32
    n_samples = 1000  # ~4 seconds at 256 Hz
    tau_max = 3
    n_true_connections = 15
    
    print(f"\nConfiguration:")
    print(f"  Channels: {n_channels}")
    print(f"  Samples: {n_samples}")
    print(f"  Max lag: {tau_max}")
    print(f"  True connections: {n_true_connections}")
    print(f"  Potential connections: {n_channels * n_channels * tau_max:,}")
    
    # Generate data
    print("\n1. Generating EEG-like data...")
    start = time.perf_counter()
    data, ground_truth, channel_names = generate_eeg_like_data(
        n_channels=n_channels,
        n_samples=n_samples,
        n_true_connections=n_true_connections,
    )
    print(f"   Data shape: {data.shape}")
    print(f"   Generation time: {time.perf_counter() - start:.2f}s")
    
    # Create handler
    print("\n2. Creating data handler...")
    handler = DataHandler(data, var_names=channel_names, normalize=True)
    
    # Run PCMCI with ParCorr (fast, linear)
    print("\n3. Running PCMCI with ParCorr...")
    print(f"   Testing {n_channels}x{n_channels}x{tau_max} = {n_channels**2 * tau_max:,} potential links")
    print("   (Using max_conds_dim=2 for speed with 64 channels)")
    
    pcmci = PCMCI(
        handler,
        cond_ind_test=ParCorr(significance='analytic'),
        verbosity=1
    )
    
    start = time.perf_counter()
    results = pcmci.run(
        tau_max=tau_max,
        tau_min=1,
        pc_alpha=0.05,
        alpha_level=0.01,
        fdr_method='fdr_bh',  # Benjamini-Hochberg
        max_conds_dim=2,  # Limit conditioning set size for speed
        max_subsets=50,  # Limit subset testing
    )
    elapsed = time.perf_counter() - start
    
    print(f"\n   Total analysis time: {elapsed:.2f}s")
    print(f"   Tests per second: {n_channels**2 * tau_max / elapsed:.0f}")
    
    # Analyze results
    precision, recall, f1 = analyze_connectivity(results, ground_truth, channel_names)
    
    # Create connectivity matrix visualization
    print("\n4. Creating visualizations...")
    
    # Aggregate connectivity across lags
    conn_matrix = np.zeros((n_channels, n_channels))
    for src, tgt, tau, stat, pval in results.significant_links:
        conn_matrix[src, tgt] = max(conn_matrix[src, tgt], abs(stat))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Connectivity matrix
    ax = axes[0]
    im = ax.imshow(conn_matrix, cmap='hot', aspect='equal')
    ax.set_title(f'EEG Connectivity Matrix\n({len(results.significant_links)} significant links)')
    ax.set_xlabel('Target Channel')
    ax.set_ylabel('Source Channel')
    plt.colorbar(im, ax=ax, label='Max |Correlation|')
    
    # Histogram of connection strengths
    ax = axes[1]
    all_vals = np.array(results.val_matrix).flatten()
    all_pvals = np.array(results.pval_matrix).flatten()
    significant = all_pvals < 0.001
    
    ax.hist(all_vals[~significant], bins=50, alpha=0.7, label='Non-significant', color='gray')
    ax.hist(all_vals[significant], bins=50, alpha=0.7, label='Significant (p<0.001)', color='red')
    ax.set_xlabel('Partial Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Link Strengths')
    ax.legend()
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('example_eeg_connectivity.png', dpi=150)
    print("   Saved: example_eeg_connectivity.png")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Channels:           {n_channels}")
    print(f"Time points:        {n_samples}")
    print(f"Potential links:    {n_channels**2 * tau_max:,}")
    print(f"Total time:         {elapsed:.2f}s")
    print(f"Links/second:       {n_channels**2 * tau_max / elapsed:,.0f}")
    print(f"Significant links:  {len(results.significant_links)}")
    print(f"Sparsity:           {100 * len(results.significant_links) / (n_channels**2 * tau_max):.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
