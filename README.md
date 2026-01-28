# JAX-PCMCI

**High-Performance Causal Discovery from Time Series using JAX**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-accelerated-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Author's note: Hello, what brings you to this program? If you are here, I would love to hear your thoughts on this library and how you are using it. Just send me an email anytime. If you have any issues with it, please open an issue, or just tell me. I will likely get it fixed somewhat quickly.

Also, side note. A lot of the performance is based off of the parameters. So changes like batch size, tau, precision, or any other parameters can make a huge difference to speed.

JAX-PCMCI is a library for causal discovery from time series data, implementing the PCMCI family of algorithms with GPU/TPU acceleration through JAX. It provides significant speedups over CPU-based implementations while maintaining scientific rigor.

## Key Features

- **GPU/TPU Acceleration**: Leverages JAX for massive parallelization
- **PCMCI & PCMCI+**: Both lagged and contemporaneous causal discovery
- **Multiple Independence Tests**:
    - `ParCorr`: Partial correlation for linear dependencies
    - `CMIKnn`: k-NN conditional mutual information for nonlinear dependencies
    - `CMISymbolic`: fast symbolic CMI via discretization
    - `GPDCond`: Gaussian Process distance correlation for complex nonlinear relationships
- **Parallel Test Execution**: Vectorized batch testing with `vmap`/`pmap`
- **Memory-Aware Batching**: Automatic batch sizing when `batch_size` is not set
- **FDR Correction**: Built-in Benjamini-Hochberg and Bonferroni corrections
- **Publication-Ready Visualization**: Graph and time series plots

## Installation

### Basic Installation (CPU)

```bash
pip install jax-pcmci
```

### With GPU Support (CUDA 12)

```bash
pip install jax-pcmci[gpu]
```

### With TPU Support

```bash
pip install jax-pcmci[tpu]
```

### From Source

```bash
git clone https://github.com/gpgabriel25/jax-pcmci.git
cd jax-pcmci
pip install -e ".[dev]"
```

## Quick Start

### Basic PCMCI Analysis

```python
import jax.numpy as jnp
from jax_pcmci import PCMCI, ParCorr, DataHandler

# Generate sample data (T time points, N variables)
key = jax.random.PRNGKey(42)
T, N = 1000, 5
data = jax.random.normal(key, (T, N))

# Create data handler (automatically normalizes data)
datahandler = DataHandler(data, normalize=True)

# Run PCMCI with partial correlation test
pcmci = PCMCI(datahandler, cond_ind_test=ParCorr())
results = pcmci.run(tau_max=3, pc_alpha=0.05)

# View results
print(results.summary())

# Visualize causal graph
results.plot_graph()
```

### PCMCI+ for Contemporaneous Effects

```python
from jax_pcmci import PCMCIPlus, ParCorr, DataHandler

# PCMCI+ discovers both lagged AND contemporaneous causal links
pcmci_plus = PCMCIPlus(datahandler, cond_ind_test=ParCorr())
results = pcmci_plus.run(tau_max=3)

# Get contemporaneous links specifically
contemp_links = results.get_contemporaneous_links()
for src, tgt, stat, pval in contemp_links:
    print(f"X{src}(t) -> X{tgt}(t): stat={stat:.3f}, p={pval:.4f}")
```

### Nonlinear Causal Discovery

```python
from jax_pcmci import PCMCI, CMIKnn, DataHandler

# Use CMI-kNN for nonlinear relationships
test = CMIKnn(k=10, significance='permutation', n_permutations=200)
pcmci = PCMCI(datahandler, cond_ind_test=test)
results = pcmci.run(tau_max=3)
```

## Available Independence Tests

### ParCorr (Partial Correlation)

Best for linear dependencies. Fastest test with analytical p-values.

```python
from jax_pcmci import ParCorr

test = ParCorr(
    significance='analytic',  # or 'permutation'
    alpha=0.05
)
```

### CMIKnn (Conditional Mutual Information with k-NN)

Captures nonlinear dependencies. Uses permutation testing.

```python
from jax_pcmci import CMIKnn

test = CMIKnn(
    k=10,                        # Number of neighbors
    significance='permutation',   # Required for accurate p-values
    n_permutations=500,
    metric='chebyshev'           # or 'euclidean'
)
```

### CMISymbolic (Symbolic Conditional Mutual Information)

Fast discretized alternative to CMI-kNN.

```python
from jax_pcmci import CMISymbolic

test = CMISymbolic(
    n_symbols=6,              # Number of bins
    significance='analytic'
)
```

### GPDCond (Gaussian Process Distance Correlation)

Advanced nonlinear test using GP regression residuals.

```python
from jax_pcmci import GPDCond

test = GPDCond(
    kernel='rbf',           # or 'matern32', 'matern52'
    length_scale=1.0,
    significance='permutation'
)
```

## Configuration

### Device Selection

```python
from jax_pcmci import set_device, get_device_info

# Check available devices
info = get_device_info()
print(f"GPUs available: {info['gpu_count']}")
print(f"Default backend: {info['default_backend']}")

# Force specific device
set_device('gpu')   # Use GPU
set_device('tpu')   # Use TPU
set_device('cpu')   # Force CPU
set_device('auto')  # Auto-select best
```

### Global Configuration

```python
from jax_pcmci import PCMCIConfig

config = PCMCIConfig(
    precision='float32',       # default; use 'float64' for higher accuracy
    parallelization='auto',    # 'vmap', 'pmap', or 'sequential'
    random_seed=42,            # For reproducibility
    enable_x64=False,          # enable 64-bit if using float64
    progress_bar=True,
    verbosity=1                # 0=silent, 1=normal, 2=verbose
)
config.apply()
```

## Working with Results

### Accessing Causal Links

```python
results = pcmci.run(tau_max=3)

# All significant links
for src, tgt, tau, stat, pval in results.significant_links:
    print(f"X{src}(t-{tau}) -> X{tgt}(t)")

# Get parents of a specific variable
parents = results.get_parents(variable=0)

# Check specific link
is_causal = results.is_significant(source=1, target=0, lag=2)
```

### Visualization

```python
# Causal graph
fig = results.plot_graph(layout='circular', save_path='graph.png')

# Time series graph (shows temporal structure)
fig = results.plot_time_series_graph(save_path='ts_graph.png')

# Matrix heatmaps
fig = results.plot_matrix(matrix='val', save_path='values.png')
fig = results.plot_matrix(matrix='pval', save_path='pvalues.png')
```

### Export

```python
# To NetworkX
G = results.to_networkx()

# To dictionary (JSON-serializable)
data = results.to_dict()

# Save to file
import json
with open('results.json', 'w') as f:
    json.dump(data, f, indent=2)
```

## Advanced Usage

### Custom Independence Test

```python
from jax_pcmci.independence_tests import CondIndTest
import jax.numpy as jnp

class MyCustomTest(CondIndTest):
    name = "MyTest"
    measure = "custom_measure"
    
    def compute_statistic(self, X, Y, Z=None):
        # Your JAX-compatible computation here
        # Must return a scalar JAX array
        pass
    
    def compute_pvalue(self, statistic, n_samples, n_conditions):
        # Compute p-value from statistic
        pass

# Use with PCMCI
pcmci = PCMCI(datahandler, cond_ind_test=MyCustomTest())
```

### Batch Processing for Large Datasets

```python
# For very large datasets, use batch MCI
results = pcmci.run_batch_mci(tau_max=5)
```

### End-to-End Speed Benchmark

```bash
python examples/benchmark_pcmci_speed.py
```

Environment knobs (optional):
- `PCMCI_SPEED_T`, `PCMCI_SPEED_N`, `PCMCI_SPEED_TAU_MAX`
- `PCMCI_SPEED_PC_ALPHA`, `PCMCI_SPEED_ALPHA_LEVEL`, `PCMCI_SPEED_MAX_CONDS_DIM`
- `PCMCI_SPEED_DEVICE`, `PCMCI_SPEED_WARMUP`

### Memory-Efficient Mode

```python
config = PCMCIConfig(
    memory_efficient=True,  # Trades speed for memory
    batch_size=100          # Process tests in batches
)
config.apply()

### GPU Memory Controls

```python
config = PCMCIConfig(
    gpu_preallocate=False,   # Avoids full preallocation
    gpu_memory_fraction=0.7, # Allocate 70% of GPU memory
    gpu_allocator='bfc',     # or 'platform'
)
config.apply()
```
```

## Algorithm Details

### PCMCI

PCMCI (Peter and Clark Momentary Conditional Independence) is a two-phase algorithm:

1. **PC Phase**: Iteratively removes spurious parent candidates using conditional independence tests with increasing conditioning set sizes.

2. **MCI Phase**: Tests remaining links using Momentary Conditional Independence, conditioning on the parents of both source and target.

### PCMCI+

PCMCI+ extends PCMCI to handle contemporaneous (τ=0) causal links:

1. **Skeleton Discovery**: Finds undirected edges including contemporaneous
2. **Orientation**: Uses time order and v-structure rules to orient edges
3. **MCI Testing**: Final tests with full conditioning sets

##  Comparison with Tigramite



| Feature | JAX-PCMCI | Tigramite |
|---------|-----------|-----------|
| GPU/TPU Support |  Native |  CPU only |
| Parallelization |  vmap/pmap |  Limited |
| JIT Compilation |  Full |  No |
| Independence Tests | ParCorr, CMIKnn, CMISymbolic, GPDC | Many |
| Speed (GPU) | 10-100x faster | Baseline |

## 📖 References

1. Runge, J. et al. (2019). "Detecting and quantifying causal associations in large nonlinear time series datasets". Science Advances, 5(11), eaau4996.

2. Runge, J. (2020). "Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets". UAI 2020.

3. Spirtes, P., Glymour, C., & Scheines, R. (2000). "Causation, prediction, and search". MIT press.

## License

MIT License - see [LICENSE](LICENSE) for details.


## Contact

For questions or issues, please open a GitHub issue or contact me at gpgabriel25@gmail.com
