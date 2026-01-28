
import time
import jax
import jax.numpy as jnp
from jax_pcmci import PCMCI, ParCorr, DataHandler

def benchmark():
    print(f"JAX Backend: {jax.devices()}")
    
    # Generate data
    T = 1000
    N = 20
    tau_max = 5
    
    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, (T, N))
    
    handler = DataHandler(data)
    pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=1)
    
    # Warmup
    print("Warming up...")
    pcmci.run(tau_max=1, pc_alpha=0.05)
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    pcmci.run(tau_max=tau_max, pc_alpha=0.05)
    end_time = time.time()
    
    print(f"Total time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
