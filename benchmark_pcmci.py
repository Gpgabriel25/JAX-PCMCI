
import time
import jax
import jax.numpy as jnp
from jax_pcmci import PCMCI, ParCorr, DataHandler

def benchmark():
    devices = jax.devices()
    print(f"JAX Backend: {devices}")
    
    # Configuration
    T = 1000
    N = 10
    tau_max = 5
    pc_alpha = 0.05
    repeats = 5
    
    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, (T, N))
    
    handler = DataHandler(data)
    pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=1)
    
    # Warmup
    print(f"\nWarming up (N={N}, tau_max={tau_max})...")
    print("This compiles the JAX functions for each PC condition dimension.")
    start_warm = time.time()
    pcmci.run(tau_max=tau_max, pc_alpha=pc_alpha)
    print(f"Warmup complete in {time.time() - start_warm:.2f}s")
    
    # Benchmark
    print(f"\nRunning benchmark ({repeats} repeats)...")
    times = []
    
    for i in range(repeats):
        start_time = time.time()
        # Suppress verbosity for timing
        # Run phases separately to measure split
        pcmci.verbosity = 0
        
        # Phase 1
        t0 = time.time()
        parents = pcmci.run_pc_stable(tau_max=tau_max, pc_alpha=pc_alpha)
        # Force block if needed (though run_pc_stable usually returns actual dict which implies sync)
        t1 = time.time()
        
        # Phase 2
        # Need to re-instantiate or use internal method if possible, but run_batch_mci is public
        # Logic from pcmci.run:
        # pcmci.datahandler.precompute_lagged_data(tau_max) # done implicitly or explicitly?
        # Let's ensure precomputation
        pcmci.datahandler.precompute_lagged_data(tau_max)
        
        val, pval = pcmci.run_batch_mci(
            tau_max=tau_max, 
            tau_min=1, 
            parents=parents
        )
        # Block output to ensure timing correctness
        val.block_until_ready()
        t2 = time.time()
        
        phase1_time = t1 - t0
        phase2_time = t2 - t1
        total_time = t2 - t0
        
        times.append(total_time)
        print(f"Run {i+1}: {total_time:.4f}s (PC: {phase1_time:.4f}s, MCI: {phase2_time:.4f}s)")
        
    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.4f} seconds")
    print(f"Std Dev: {jnp.std(jnp.array(times)):.4f} seconds")

if __name__ == "__main__":
    benchmark()
