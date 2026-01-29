
import time
import jax
import jax.numpy as jnp
from jax_pcmci.algorithms.pcmci_plus import PCMCIPlus
from jax_pcmci import ParCorr, DataHandler

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
    pcmci_plus = PCMCIPlus(handler, cond_ind_test=ParCorr(), verbosity=1)
    
    # Warmup
    print(f"\nWarming up PCMCI+ (N={N}, tau_max={tau_max})...")
    start_warm = time.time()
    pcmci_plus.run(tau_max=tau_max, pc_alpha=pc_alpha)
    print(f"Warmup complete in {time.time() - start_warm:.2f}s")
    
    # Benchmark
    print(f"\nRunning PCMCI+ benchmark ({repeats} repeats)...")
    times = []
    
    for i in range(repeats):
        pcmci_plus.verbosity = 0
        
        t0 = time.time()
        # PCMCI+ is contiguous, harder to split cleanly without internal access, 
        # but let's just time the whole thing for now as requested.
        pcmci_plus.run(tau_max=tau_max, pc_alpha=pc_alpha)
        
        # Ensure completion (results object creation does some cpu work, but graph is main thing)
        # Ideally we'd block on the graph or results
        t1 = time.time()
        
        duration = t1 - t0
        times.append(duration)
        print(f"Run {i+1}: {duration:.4f}s")
        
    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.4f} seconds")
    print(f"Std Dev: {jnp.std(jnp.array(times)):.4f} seconds")

if __name__ == "__main__":
    benchmark()
