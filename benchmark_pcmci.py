
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
        pcmci.verbosity = 0
        pcmci.run(tau_max=tau_max, pc_alpha=pc_alpha)
        duration = time.time() - start_time
        times.append(duration)
        print(f"Run {i+1}: {duration:.4f}s")
        
    avg_time = sum(times) / len(times)
    print(f"\nAverage time: {avg_time:.4f} seconds")
    print(f"Std Dev: {jnp.std(jnp.array(times)):.4f} seconds")

if __name__ == "__main__":
    benchmark()
