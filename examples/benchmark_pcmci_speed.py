"""
Benchmark: PCMCI and PCMCI+ end-to-end speed.

Run this script to benchmark PCMCI and PCMCI+ across a range of parameters.
Results are saved to `benchmark_results.csv`.

Environment variables:
- PCMCI_SPEED_T_VALUES: comma-separated list of T values (default "250,500,1000")
- PCMCI_SPEED_N_VALUES: comma-separated list of N values (default "5,10,20")
- PCMCI_SPEED_TAU_MAX: maximum lag (default 2)
- PCMCI_SPEED_PC_ALPHA: PC alpha (default 0.05)
- PCMCI_SPEED_DEVICE: cpu|gpu|tpu|auto (default auto)
"""

from __future__ import annotations

import os
import time
import resource
import csv
import jax
import jax.numpy as jnp
from jax_pcmci import (
    DataHandler,
    ParCorr,
    PCMCI,
    PCMCIPlus,
    PCMCIConfig,
    get_device_info,
    set_device,
)

def _env_list_int(name: str, default: str) -> list[int]:
    return [int(x) for x in os.environ.get(name, default).split(",")]

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _get_mem_usage_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def benchmark_run(label: str, fn, warmup: bool = True) -> dict:
    if warmup:
        try:
            res = fn()
            if hasattr(res, "block_until_ready"):
                 res.block_until_ready()
            elif hasattr(res, "val_matrix"):
                 res.val_matrix.block_until_ready()
            jax.block_until_ready(jnp.array(0))
        except Exception:
            pass # Warmup might fail if determinism checks etc, but usually fine

    jax.block_until_ready(jnp.array(0))  # clear pending
    start_mem = _get_mem_usage_mb()
    start_time = time.perf_counter()
    
    result = fn()
    # Force sync
    if hasattr(result, "val_matrix"):
         jax.block_until_ready(result.val_matrix)
    elif isinstance(result, dict):
        # PC phase result
        pass 
        
    elapsed = time.perf_counter() - start_time
    peak_mem = _get_mem_usage_mb()
    
    return {
        "time": elapsed,
        "mem_diff_mb": peak_mem - start_mem,
        "peak_mem_mb": peak_mem
    }

def main() -> None:
    t_values = _env_list_int("PCMCI_SPEED_T_VALUES", "250,500,1000")
    n_values = _env_list_int("PCMCI_SPEED_N_VALUES", "5,10,20")
    tau_max = _env_int("PCMCI_SPEED_TAU_MAX", 2)
    pc_alpha = _env_float("PCMCI_SPEED_PC_ALPHA", 0.05)
    
    device = os.environ.get("PCMCI_SPEED_DEVICE", "auto")
    set_device(device)
    PCMCIConfig().apply()
    
    info = get_device_info()
    print(f"Device: {info['default_backend']}")
    
    results = []
    
    # Warmup
    print("Warming up...")
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (100, 5))
    handler = DataHandler(data)
    test = ParCorr()
    pcmci = PCMCI(handler, cond_ind_test=test, verbosity=0)
    pcmci.run(tau_max=1, pc_alpha=0.05)

    print("\nStarting benchmarks...")
    print(f"{'Algorithm':<10} {'N':<5} {'T':<6} {'Time(s)':<10} {'Mem(MB)':<10}")
    print("-" * 45)

    for n in n_values:
        for t in t_values:
            key, subkey = jax.random.split(key)
            data = jax.random.normal(subkey, (t, n))
            handler = DataHandler(data)
            test = ParCorr()
            
            # PCMCI Benchmark
            pcmci = PCMCI(handler, cond_ind_test=test, verbosity=0)
            metrics = benchmark_run("PCMCI", lambda: pcmci.run(tau_max=tau_max, pc_alpha=pc_alpha))
            
            record = {
                "algorithm": "PCMCI",
                "n": n,
                "t": t,
                "tau_max": tau_max,
                "time": metrics["time"],
                "memory_mb": metrics["peak_mem_mb"]
            }
            results.append(record)
            print(f"{'PCMCI':<10} {n:<5} {t:<6} {metrics['time']:<10.4f} {metrics['peak_mem_mb']:<10.1f}")
            
            # PCMCI+ Benchmark
            pcmci_plus = PCMCIPlus(handler, cond_ind_test=test, verbosity=0)
            metrics_plus = benchmark_run("PCMCI+", lambda: pcmci_plus.run(tau_max=tau_max, pc_alpha=pc_alpha))
            
            record_plus = {
                "algorithm": "PCMCI+",
                "n": n,
                "t": t,
                "tau_max": tau_max,
                "time": metrics_plus["time"],
                "memory_mb": metrics_plus["peak_mem_mb"]
            }
            results.append(record_plus)
            print(f"{'PCMCI+':<10} {n:<5} {t:<6} {metrics_plus['time']:<10.4f} {metrics_plus['peak_mem_mb']:<10.1f}")

    # Save to CSV
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "n", "t", "tau_max", "time", "memory_mb"])
        writer.writeheader()
        writer.writerows(results)
    
    print("\nBenchmark complete. Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
