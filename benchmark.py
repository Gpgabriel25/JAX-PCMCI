
import time
import argparse
import jax
import jax.numpy as jnp
from jax_pcmci import PCMCI, ParCorr, DataHandler
from jax_pcmci.algorithms.pcmci_plus import PCMCIPlus

def run_pcmci_benchmark(data, repeats, tau_max, pc_alpha, verbosity):
    handler = DataHandler(data)
    pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=verbosity)
    
    # Warmup / Compilation
    print(f"\n[Warmup] Compiling JAX functions (N={handler.N}, tau_max={tau_max})...")
    t0 = time.time()
    # Run once to compile
    pcmci.run(tau_max=tau_max, pc_alpha=pc_alpha)
    t1 = time.time()
    compile_time = t1 - t0
    print(f"[Warmup] Compilation/Warmup took: {compile_time:.4f}s")
    
    # Benchmark
    print(f"\n[Benchmark] Running PCMCI ({repeats} repeats)...")
    times = []
    
    pcmci.verbosity = 0
    
    for i in range(repeats):
        # Phase 1
        t_start = time.time()
        parents = pcmci.run_pc_stable(tau_max=tau_max, pc_alpha=pc_alpha)
        # Implicitly synced by data dependency usually, but for timing we assume sync on Python return if not async
        t_mid = time.time()
        
        # Phase 2
        pcmci.datahandler.precompute_lagged_data(tau_max)
        val, pval = pcmci.run_batch_mci(tau_max=tau_max, tau_min=1, parents=parents)
        val.block_until_ready()
        t_end = time.time()
        
        p1 = t_mid - t_start
        p2 = t_end - t_mid
        total = t_end - t_start
        
        times.append(total)
        print(f"Run {i+1}: {total:.4f}s (Phase 1: {p1:.4f}s, Phase 2: {p2:.4f}s)")
        
    return times, compile_time

def run_pcmci_plus_benchmark(data, repeats, tau_max, pc_alpha, verbosity):
    handler = DataHandler(data)
    pcmci_plus = PCMCIPlus(handler, cond_ind_test=ParCorr(), verbosity=verbosity)
    
    # Warmup / Compilation
    print(f"\n[Warmup] Compiling JAX functions (N={handler.N}, tau_max={tau_max})...")
    t0 = time.time()
    pcmci_plus.run(tau_max=tau_max, pc_alpha=pc_alpha)
    t1 = time.time()
    compile_time = t1 - t0
    print(f"[Warmup] Compilation/Warmup took: {compile_time:.4f}s")
    
    # Benchmark
    print(f"\n[Benchmark] Running PCMCI+ ({repeats} repeats)...")
    times = []
    
    pcmci_plus.verbosity = 0
    
    for i in range(repeats):
        t_start = time.time()
        
        # Ensure precomputation
        pcmci_plus.datahandler.precompute_lagged_data(tau_max)
        
        # Phase 1: Skeleton
        t0 = time.time()
        skeleton, sepsets = pcmci_plus._discover_skeleton(
            tau_max=tau_max,
            tau_min=0, # Default for PCMCI+
            pc_alpha=pc_alpha,
            max_conds_dim=None
        )
        # Skeleton is Python dict, implicit sync
        t1 = time.time()
        
        # Phase 2: Orientation
        oriented_graph = pcmci_plus._orient_edges(
            skeleton=skeleton,
            sepsets=sepsets,
            tau_max=tau_max,
            orientation_alpha=pc_alpha
        )
        # Block JAX array
        oriented_graph.block_until_ready()
        t2 = time.time()
        
        # Phase 3: MCI
        val, pval = pcmci_plus._run_mci_plus(
            oriented_graph=oriented_graph,
            tau_max=tau_max,
            tau_min=0,
            max_conds_py=None,
            max_conds_px=None
        )
        val.block_until_ready()
        t3 = time.time()
        
        p1 = t1 - t0
        p2 = t2 - t1
        p3 = t3 - t2
        total = t3 - t_start
        
        times.append(total)
        print(f"Run {i+1}: {total:.4f}s (P1: {p1:.4f}s, P2: {p2:.4f}s, P3: {p3:.4f}s)")
        
    return times, compile_time

def main():
    parser = argparse.ArgumentParser(description="JAX-PCMCI Benchmark Suite")
    parser.add_argument("--algo", type=str, choices=["pcmci", "pcmci+"], default="pcmci", help="Algorithm to benchmark")
    parser.add_argument("--N", type=int, default=10, help="Number of variables")
    parser.add_argument("--T", type=int, default=1000, help="Time points")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats")
    parser.add_argument("--tau_max", type=int, default=5, help="Max lag")
    parser.add_argument("--device", type=str, default=None, help="JAX device")
    args = parser.parse_args()
    
    print(f"JAX Backend: {jax.devices()}")
    
    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, (args.T, args.N))
    
    if args.algo == "pcmci":
        times, compile_time = run_pcmci_benchmark(data, args.repeats, args.tau_max, 0.05, 1)
    else:
        times, compile_time = run_pcmci_plus_benchmark(data, args.repeats, args.tau_max, 0.05, 1)
        
    avg_time = sum(times) / len(times)
    std_time = jnp.std(jnp.array(times))
    
    print(f"\n[Summary] {args.algo.upper()}")
    print(f"Average Runtime: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Compilation Time: {compile_time:.2f}s")
    print(f"Compile/Run Ratio: {compile_time/avg_time:.1f}x")

if __name__ == "__main__":
    main()
