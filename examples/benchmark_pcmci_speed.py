"""
Benchmark: PCMCI and PCMCI+ end-to-end speed.

Environment variables:
- PCMCI_SPEED_T: time points (default 1000)
- PCMCI_SPEED_N: variables (default 10)
- PCMCI_SPEED_TAU_MAX: maximum lag (default 2)
- PCMCI_SPEED_PC_ALPHA: PC alpha (default 0.05)
- PCMCI_SPEED_ALPHA_LEVEL: MCI alpha (default 0.05)
- PCMCI_SPEED_DEVICE: cpu|gpu|tpu|auto (default auto)
- PCMCI_SPEED_WARMUP: 1 to run a warmup pass (default 1)
"""

from __future__ import annotations

import os
import time
import resource

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


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes"}


def _get_cpu_mem_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        # ru_maxrss is KB on Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _get_gpu_mem_stats() -> dict:
    stats = {}
    try:
        dev = jax.devices()[0]
        if hasattr(dev, "memory_stats"):
            mem = dev.memory_stats() or {}
            for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved"):
                if key in mem:
                    stats[key] = mem[key]
    except Exception:
        pass
    return stats


def _format_mem_stats(cpu_mb: float, gpu_stats: dict) -> str:
    parts = [f"CPU RSS: {cpu_mb:.1f} MB"]
    if gpu_stats:
        if "bytes_in_use" in gpu_stats:
            parts.append(f"GPU in use: {gpu_stats['bytes_in_use'] / 1e6:.1f} MB")
        if "peak_bytes_in_use" in gpu_stats:
            parts.append(f"GPU peak: {gpu_stats['peak_bytes_in_use'] / 1e6:.1f} MB")
        if "bytes_reserved" in gpu_stats:
            parts.append(f"GPU reserved: {gpu_stats['bytes_reserved'] / 1e6:.1f} MB")
    return " | ".join(parts)


def time_run(label: str, fn) -> float:
    cpu_before = _get_cpu_mem_mb()
    gpu_before = _get_gpu_mem_stats()
    start = time.perf_counter()
    result = fn()
    if hasattr(result, "val_matrix"):
        jax.block_until_ready(result.val_matrix)
    elapsed = time.perf_counter() - start
    cpu_after = _get_cpu_mem_mb()
    gpu_after = _get_gpu_mem_stats()
    print(f"{label}: {elapsed:.3f}s")
    print(f"{label} memory: {_format_mem_stats(cpu_after, gpu_after)}")
    return elapsed


def main() -> None:
    t = _env_int("PCMCI_SPEED_T", 500)
    n = _env_int("PCMCI_SPEED_N", 10)
    tau_max = _env_int("PCMCI_SPEED_TAU_MAX", 2)
    pc_alpha = _env_float("PCMCI_SPEED_PC_ALPHA", 0.05)
    alpha_level = _env_float("PCMCI_SPEED_ALPHA_LEVEL", 0.05)
    do_warmup = _env_bool("PCMCI_SPEED_WARMUP", True)
    max_conds_dim = _env_int("PCMCI_SPEED_MAX_CONDS_DIM", 3)

    device = os.environ.get("PCMCI_SPEED_DEVICE", "auto")
    set_device(device)

    config = PCMCIConfig()
    config.apply()

    info = get_device_info()
    print("=" * 60)
    print("PCMCI Speed Benchmark")
    print("=" * 60)
    print(f"Device: {info['default_backend']}")
    print(f"T={t}, N={n}, tau_max={tau_max}")

    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (t, n))
    handler = DataHandler(data)
    test = ParCorr()

    if do_warmup:
        # Full warmup: run PCMCI once with actual parameters to JIT compile
        # all necessary configurations. This simulates real-world usage where
        # the algorithm is run multiple times on similar data.
        print("Warming up JIT compilation...")
        warm_pcmci = PCMCI(handler, cond_ind_test=test, verbosity=0)
        _ = warm_pcmci.run(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
        )
        warm_pcmci_plus = PCMCIPlus(handler, cond_ind_test=test, verbosity=0)
        _ = warm_pcmci_plus.run(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            max_conds_dim=max_conds_dim,
        )
        handler.clear_cache()
        handler.precompute_lagged_data(tau_max)
        print("Warmup complete. Running timed benchmarks...")

    pcmci = PCMCI(handler, cond_ind_test=test, verbosity=0)
    pcmci_plus = PCMCIPlus(handler, cond_ind_test=test, verbosity=0)

    time_run(
        "PCMCI",
        lambda: pcmci.run(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            max_conds_dim=max_conds_dim,
        ),
    )

    time_run(
        "PCMCI+",
        lambda: pcmci_plus.run(
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=alpha_level,
            max_conds_dim=max_conds_dim,
        ),
    )


if __name__ == "__main__":
    main()
