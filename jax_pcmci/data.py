"""
Data Handling Module for JAX-PCMCI
==================================

This module provides classes for handling time series data in JAX-PCMCI.
It includes data validation, normalization, missing value handling,
and efficient data access patterns for PCMCI algorithms.

Key Classes
-----------
- DataHandler: Main interface for preparing data for PCMCI analysis
- TimeSeriesData: Container for time series with metadata

Example
-------
>>> import jax.numpy as jnp
>>> from jax_pcmci.data import DataHandler
>>>
>>> # Create data (T timesteps, N variables)
>>> data = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], ...])
>>>
>>> # Initialize handler with automatic normalization
>>> handler = DataHandler(data, normalize=True)
>>>
>>> # Access lagged data for PCMCI
>>> X_t, X_lag = handler.get_lagged_data(tau=2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from collections import OrderedDict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_pcmci.config import get_config


@dataclass
class TimeSeriesData:
    """
    Container for time series data with metadata.

    This class stores the raw time series data along with optional metadata
    such as variable names, time indices, and data provenance information.

    Parameters
    ----------
    values : jax.Array
        The time series data array of shape (T, N) where T is the number
        of time points and N is the number of variables.
    var_names : list of str, optional
        Names for each variable. If None, uses ['X0', 'X1', ...].
    time_index : array-like, optional
        Time indices for each observation. If None, uses [0, 1, 2, ...].
    mask : jax.Array, optional
        Boolean mask of shape (T, N) where True indicates valid data.
        Used for handling missing values.
    metadata : dict, optional
        Additional metadata (e.g., units, source, sampling frequency).

    Attributes
    ----------
    T : int
        Number of time points.
    N : int
        Number of variables.
    shape : tuple
        Shape of the data array (T, N).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> values = jnp.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> ts = TimeSeriesData(
    ...     values=values,
    ...     var_names=['Temperature', 'Pressure', 'Humidity'],
    ...     metadata={'units': ['C', 'Pa', '%'], 'frequency': 'hourly'}
    ... )
    >>> print(f"Data shape: {ts.shape}")
    Data shape: (3, 3)
    """

    values: jax.Array
    var_names: Optional[List[str]] = None
    time_index: Optional[jax.Array] = None
    mask: Optional[jax.Array] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize data attributes."""
        # Convert to JAX array if needed
        if not isinstance(self.values, jax.Array):
            self.values = jnp.asarray(self.values, dtype=get_config().dtype)

        # Validate shape
        if self.values.ndim != 2:
            raise ValueError(
                f"Data must be 2D (T, N), got {self.values.ndim}D with shape {self.values.shape}"
            )

        # Initialize variable names
        if self.var_names is None:
            self.var_names = [f"X{i}" for i in range(self.N)]
        elif len(self.var_names) != self.N:
            raise ValueError(
                f"var_names length ({len(self.var_names)}) must match "
                f"number of variables ({self.N})"
            )

        # Initialize time index
        if self.time_index is None:
            self.time_index = jnp.arange(self.T)
        elif len(self.time_index) != self.T:
            raise ValueError(
                f"time_index length ({len(self.time_index)}) must match "
                f"number of time points ({self.T})"
            )

        # Initialize mask
        if self.mask is None:
            self.mask = jnp.ones_like(self.values, dtype=bool)
        elif self.mask.shape != self.values.shape:
            raise ValueError(
                f"mask shape {self.mask.shape} must match data shape {self.values.shape}"
            )

    @property
    def T(self) -> int:
        """Number of time points."""
        return self.values.shape[0]

    @property
    def N(self) -> int:
        """Number of variables."""
        return self.values.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the data (T, N)."""
        return self.values.shape

    def get_variable(self, var: Union[int, str]) -> jax.Array:
        """
        Get data for a single variable.

        Parameters
        ----------
        var : int or str
            Variable index or name.

        Returns
        -------
        jax.Array
            1D array of shape (T,) containing the variable's time series.

        Examples
        --------
        >>> temp = ts.get_variable('Temperature')
        >>> temp = ts.get_variable(0)  # Same as above
        """
        if isinstance(var, str):
            if var not in self.var_names:
                raise ValueError(f"Unknown variable: {var}")
            var = self.var_names.index(var)
        return self.values[:, var]

    def get_time_range(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> "TimeSeriesData":
        """
        Get a subset of the time series.

        Parameters
        ----------
        start : int, optional
            Starting time index (inclusive).
        end : int, optional
            Ending time index (exclusive).

        Returns
        -------
        TimeSeriesData
            New TimeSeriesData object with the selected time range.
        """
        return TimeSeriesData(
            values=self.values[start:end],
            var_names=self.var_names.copy(),
            time_index=self.time_index[start:end] if self.time_index is not None else None,
            mask=self.mask[start:end] if self.mask is not None else None,
            metadata=self.metadata.copy(),
        )


class DataHandler:
    """
    Main data handler for preparing time series for PCMCI analysis.

    DataHandler provides a unified interface for data preparation including:
    - Automatic normalization (z-score, min-max, etc.)
    - Missing value handling
    - Lagged data construction
    - Efficient batching for parallel processing

    Parameters
    ----------
    data : array-like or TimeSeriesData
        Input time series data. Can be:
        - numpy array of shape (T, N)
        - JAX array of shape (T, N)
        - TimeSeriesData object
    normalize : bool or str, default=True
        Normalization method:
        - True or 'zscore': Z-score normalization (zero mean, unit variance)
        - 'minmax': Min-max normalization to [0, 1]
        - 'robust': Robust scaling using median and IQR
        - False or None: No normalization
    missing_flag : float, optional
        Value used to indicate missing data (e.g., np.nan, -999).
    var_names : list of str, optional
        Variable names. Overrides names in TimeSeriesData if provided.
    dtype : jnp.dtype, optional
        Data type for computations. Uses global config if not specified.

    Attributes
    ----------
    data : TimeSeriesData
        The processed time series data.
    T : int
        Number of time points.
    N : int
        Number of variables.
    is_normalized : bool
        Whether data has been normalized.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jax_pcmci.data import DataHandler
    >>>
    >>> # Simple usage
    >>> data = jnp.randn(1000, 5)  # 1000 time points, 5 variables
    >>> handler = DataHandler(data)
    >>>
    >>> # With options
    >>> handler = DataHandler(
    ...     data,
    ...     normalize='zscore',
    ...     var_names=['V0', 'V1', 'V2', 'V3', 'V4']
    ... )
    >>>
    >>> # Access lagged data for analysis
    >>> X, Y, Z = handler.get_xyz_data(x_idx=0, y_idx=1, z_indices=[2, 3], tau=2)

    Notes
    -----
    DataHandler is designed to be immutable after creation. All data
    transformations return new arrays without modifying the original data.
    """

    def __init__(
        self,
        data: Union[np.ndarray, jax.Array, TimeSeriesData],
        normalize: Union[bool, str] = True,
        missing_flag: Optional[float] = None,
        var_names: Optional[List[str]] = None,
        dtype: Optional[jnp.dtype] = None,
    ):
        self._dtype = dtype or get_config().dtype

        # Handle input types
        if isinstance(data, TimeSeriesData):
            self._data = data
            if var_names is not None:
                self._data.var_names = var_names
        else:
            # Convert to JAX array
            data_array = jnp.asarray(data, dtype=self._dtype)
            self._data = TimeSeriesData(values=data_array, var_names=var_names)

        # Get config once
        config = get_config()
        
        # Cache for lagged data cubes - keyed by (tau_max, include_contemporaneous)
        # Use OrderedDict for LRU cache behavior
        self._lagged_cache: "OrderedDict[Tuple[int, bool], Tuple[jax.Array, jax.Array]]" = OrderedDict()
        self._cache_max_entries = config.cache_max_entries

        # Cache for variable pair slices
        self._pair_cache: Optional[
            "OrderedDict[Tuple, Tuple[jax.Array, jax.Array, Optional[jax.Array]]]"
        ] = (
            OrderedDict() if config.cache_results else None
        )
        self._pair_cache_max = config.cache_max_entries

        # Handle missing values
        if missing_flag is not None:
            self._handle_missing(missing_flag)
        else:
            # Check for NaN values
            has_nan = jnp.any(jnp.isnan(self._data.values))
            if has_nan:
                self._handle_missing(jnp.nan)

        # Normalize data
        self._is_normalized = False
        self._normalization_params: dict = {}
        if normalize:
            self._normalize(normalize if isinstance(normalize, str) else "zscore")

    @property
    def data(self) -> TimeSeriesData:
        """The processed time series data."""
        return self._data

    @property
    def values(self) -> jax.Array:
        """Raw data values array."""
        return self._data.values

    @property
    def T(self) -> int:
        """Number of time points."""
        return self._data.T

    @property
    def N(self) -> int:
        """Number of variables."""
        return self._data.N

    @property
    def var_names(self) -> List[str]:
        """Variable names."""
        return self._data.var_names

    @property
    def is_normalized(self) -> bool:
        """Whether data has been normalized."""
        return self._is_normalized

    def _handle_missing(self, missing_flag: float) -> None:
        """
        Handle missing values in the data.

        Creates a mask for valid data points and optionally imputes missing values.
        """
        if jnp.isnan(missing_flag):
            missing_mask = jnp.isnan(self._data.values)
        else:
            missing_mask = self._data.values == missing_flag

        # Update mask
        self._data.mask = ~missing_mask

        # Replace missing values with NaN to ensure downstream reductions ignore them
        # (normalization and statistics use nan-aware reductions).
        self._data.values = jnp.where(self._data.mask, self._data.values, jnp.nan)

        # Count missing values
        n_missing = jnp.sum(missing_mask)
        if n_missing > 0 and get_config().verbosity >= 1:
            total = self._data.values.size
            pct = 100 * float(n_missing) / total
            print(f"DataHandler: Found {int(n_missing)} missing values ({pct:.1f}%)")

    def _normalize(self, method: str) -> None:
        """
        Normalize the data using the specified method.

        Parameters
        ----------
        method : str
            Normalization method ('zscore', 'minmax', 'robust').
        """
        values = self._data.values
        mask = self._data.mask

        if method == "zscore":
            # Compute mean and std for each variable
            mean = jnp.nanmean(values, axis=0, keepdims=True)
            std = jnp.nanstd(values, axis=0, keepdims=True)
            # Avoid division by zero
            std = jnp.where(std == 0, 1.0, std)
            normalized = (values - mean) / std
            self._normalization_params = {"method": "zscore", "mean": mean, "std": std}

        elif method == "minmax":
            min_val = jnp.nanmin(values, axis=0, keepdims=True)
            max_val = jnp.nanmax(values, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = jnp.where(range_val == 0, 1.0, range_val)
            normalized = (values - min_val) / range_val
            self._normalization_params = {"method": "minmax", "min": min_val, "max": max_val}

        elif method == "robust":
            # Use median and IQR for robust scaling
            median = jnp.nanmedian(values, axis=0, keepdims=True)
            q75 = jnp.nanpercentile(values, 75, axis=0, keepdims=True)
            q25 = jnp.nanpercentile(values, 25, axis=0, keepdims=True)
            iqr = q75 - q25
            iqr = jnp.where(iqr == 0, 1.0, iqr)
            normalized = (values - median) / iqr
            self._normalization_params = {"method": "robust", "median": median, "iqr": iqr}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Apply mask to preserve missing values
        normalized = jnp.where(mask, normalized, jnp.nan)
        self._data.values = normalized
        self._is_normalized = True

    def get_lagged_data(
        self, tau_max: int, include_contemporaneous: bool = True
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Construct lagged data matrices for PCMCI analysis.

        Creates aligned arrays where each row corresponds to a valid time point
        with all lagged values available. Results are cached for repeated calls
        with the same parameters.

        Parameters
        ----------
        tau_max : int
            Maximum time lag to consider.
        include_contemporaneous : bool, default=True
            Whether to include lag 0 (contemporaneous) values.

        Returns
        -------
        X_current : jax.Array
            Current values, shape (T - tau_max, N).
        X_lagged : jax.Array
            Lagged values, shape (T - tau_max, N, tau_max) or
            (T - tau_max, N, tau_max + 1) if include_contemporaneous.

        Examples
        --------
        >>> handler = DataHandler(data)
        >>> X_current, X_lagged = handler.get_lagged_data(tau_max=3)
        >>> print(X_lagged.shape)  # (T-3, N, 3) - lags 1, 2, 3
        """
        if tau_max < 1:
            raise ValueError(f"tau_max must be at least 1, got {tau_max}")

        # Check cache
        cache_key = (tau_max, include_contemporaneous)
        if cache_key in self._lagged_cache:
            # Move to end (most recently used)
            self._lagged_cache.move_to_end(cache_key)
            return self._lagged_cache[cache_key]

        T, N = self.T, self.N
        effective_T = T - tau_max

        # Current values (t = tau_max, tau_max+1, ..., T-1)
        X_current = self.values[tau_max:]

        # Build lagged array using simple slicing (faster than lax.scan for this pattern)
        if include_contemporaneous:
            lags = list(range(0, tau_max + 1))
        else:
            lags = list(range(1, tau_max + 1))

        n_lags = len(lags)
        lagged_slices = []
        for lag in lags:
            start_t = tau_max - lag
            end_t = start_t + effective_T
            lagged_slices.append(self.values[start_t:end_t, :])
        
        X_lagged = jnp.stack(lagged_slices, axis=2)

        # Store in cache with LRU eviction
        if len(self._lagged_cache) >= self._cache_max_entries:
            # Remove oldest entry
            self._lagged_cache.popitem(last=False)
        self._lagged_cache[cache_key] = (X_current, X_lagged)

        return X_current, X_lagged

    def clear_cache(self) -> None:
        """Clear the lagged data cache to free memory."""
        self._lagged_cache.clear()
        if self._pair_cache is not None:
            self._pair_cache.clear()

    def precompute_lagged_data(self, tau_max: int) -> None:
        """
        Precompute and cache lagged data for both contemporaneous options.
        
        Call this before running PCMCI to avoid recomputation during tests.
        
        Parameters
        ----------
        tau_max : int
            Maximum time lag to precompute.
        """
        self.get_lagged_data(tau_max, include_contemporaneous=True)
        self.get_lagged_data(tau_max, include_contemporaneous=False)

    def get_variable_pair_data(
        self,
        i: int,
        j: int,
        tau: int,
        condition_indices: Optional[Sequence[Tuple[int, int]]] = None,
        cond_vars: Optional[Sequence[int]] = None,
        cond_lags: Optional[Sequence[int]] = None,
        max_lag: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """
        Get data for testing conditional independence between two variables.

        Extracts time series data aligned for testing X_i(t-tau) -> X_j(t)
        conditional on a set of conditioning variables.

        Parameters
        ----------
        i : int
            Index of the source variable (potential cause).
        j : int
            Index of the target variable (potential effect).
        tau : int
            Time lag from i to j.
        condition_indices : list of (int, int), optional
            List of (variable_index, lag) tuples for conditioning set.
            Mutually exclusive with cond_vars/cond_lags.
        cond_vars : list of int, optional
            Conditioning variable indices. Must be used with cond_lags.
        cond_lags : list of int, optional
            Conditioning variable lags. Must be used with cond_vars.
        max_lag : int, optional
            Precomputed maximum lag for alignment. If not provided, computed
            from tau and conditioning set.

        Returns
        -------
        X : jax.Array
            Source variable values at t-tau, shape (T_eff,).
        Y : jax.Array
            Target variable values at t, shape (T_eff,).
        Z : jax.Array or None
            Conditioning set values, shape (T_eff, |Z|), or None if empty.

        Examples
        --------
        >>> # Test X0(t-2) -> X1(t) | X2(t-1), X3(t-1)
        >>> X, Y, Z = handler.get_variable_pair_data(
        ...     i=0, j=1, tau=2,
        ...     condition_indices=[(2, 1), (3, 1)]
        ... )
        >>> # Or equivalently with separate arrays
        >>> X, Y, Z = handler.get_variable_pair_data(
        ...     i=0, j=1, tau=2,
        ...     cond_vars=[2, 3], cond_lags=[1, 1]
        ... )
        """
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau}")

        # Handle both API styles
        if cond_vars is not None or cond_lags is not None:
            if condition_indices is not None:
                raise ValueError(
                    "Cannot specify both condition_indices and cond_vars/cond_lags"
                )
            if (cond_vars is None) != (cond_lags is None):
                raise ValueError(
                    "cond_vars and cond_lags must both be specified or both be None"
                )
            # Convert to condition_indices format for processing
            if cond_vars is not None:
                condition_indices = list(zip(cond_vars, cond_lags))

        cache_key = None
        if self._pair_cache is not None:
            cond_key = None
            if condition_indices:
                cond_key = tuple(sorted(condition_indices))
            cache_key = (i, j, tau, cond_key)
            if cache_key in self._pair_cache:
                self._pair_cache.move_to_end(cache_key)
                return self._pair_cache[cache_key]

        # Determine the effective time range
        if max_lag is not None:
            # Use precomputed max_lag if provided
            pass
        else:
            max_lag = tau
            if condition_indices:
                max_lag = max(max_lag, max(lag for _, lag in condition_indices))

        effective_T = self.T - max_lag
        if effective_T <= 0:
            raise ValueError(
                f"Not enough data points. T={self.T}, max_lag={max_lag}"
            )

        # Source variable at t - tau
        if tau == 0:
            X = self.values[max_lag:, i]
        else:
            X = self.values[max_lag - tau : self.T - tau, i]

        # Target variable at t
        Y = self.values[max_lag:, j]

        # Conditioning set
        if condition_indices:
            Z_list = []
            for var_idx, lag in condition_indices:
                if lag == 0:
                    z_data = self.values[max_lag:, var_idx]
                else:
                    z_data = self.values[max_lag - lag : self.T - lag, var_idx]
                Z_list.append(z_data)
            Z = jnp.stack(Z_list, axis=1)
        else:
            Z = None

        if self._pair_cache is not None and cache_key is not None:
            self._pair_cache[cache_key] = (X, Y, Z)
            self._pair_cache.move_to_end(cache_key)
            if len(self._pair_cache) > self._pair_cache_max:
                self._pair_cache.popitem(last=False)

        return X, Y, Z


    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _batch_slice_1d(
        values: jax.Array,
        start_idxs: jax.Array,
        var_idxs: jax.Array,
        length: int,
    ) -> jax.Array:
        """Vectorized 1D slice extraction using dynamic slicing."""

        def slice_one(start_idx, var_idx):
            return lax.dynamic_slice(values, (start_idx, var_idx), (length, 1)).squeeze(1)

        return jax.vmap(slice_one)(
            start_idxs.astype(jnp.int32), 
            var_idxs.astype(jnp.int32)
        )

    def get_variable_pair_batch(
        self,
        i_arr: jax.Array,
        j_arr: jax.Array,
        tau_arr: jax.Array,
        cond_vars: Optional[jax.Array] = None,
        cond_lags: Optional[jax.Array] = None,
        max_lag: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        """
        Get batched data for testing conditional independence across pairs.

        Parameters
        ----------
        i_arr : jax.Array
            Source variable indices, shape (batch,).
        j_arr : jax.Array
            Target variable indices, shape (batch,).
        tau_arr : jax.Array
            Lags for each pair, shape (batch,).
        cond_vars : jax.Array, optional
            Conditioning variable indices, shape (batch, n_cond).
        cond_lags : jax.Array, optional
            Conditioning lags, shape (batch, n_cond).
        max_lag : int, optional
            Precomputed max lag for alignment.
        """
        if max_lag is None:
            max_lag = int(jnp.max(tau_arr))
            if cond_lags is not None:
                max_lag = max(max_lag, int(jnp.max(cond_lags)))

        effective_T = self.T - max_lag
        if effective_T <= 0:
            raise ValueError(
                f"Not enough data points. T={self.T}, max_lag={max_lag}"
            )

        start_x = max_lag - tau_arr
        start_y = jnp.full_like(tau_arr, max_lag)

        X = self._batch_slice_1d(self.values, start_x, i_arr, effective_T)
        Y = self._batch_slice_1d(self.values, start_y, j_arr, effective_T)

        if cond_vars is None or cond_lags is None:
            return X, Y, None

        n_cond = cond_vars.shape[1]
        
        # If conditioning set is empty, return None
        if n_cond == 0:
            return X, Y, None
            
        batch_size = cond_vars.shape[0]
        
        # Vectorize over conditioning variables: process all at once
        # Reshape to (batch * n_cond,) for vectorized slicing
        vars_flat = cond_vars.reshape(-1)
        lags_flat = cond_lags.reshape(-1)
        starts_flat = max_lag - lags_flat
        
        # Get all conditioning slices in one vectorized call
        Z_flat = self._batch_slice_1d(self.values, starts_flat, vars_flat, effective_T)
        
        # Reshape back to (batch, effective_T, n_cond)
        Z = Z_flat.reshape(batch_size, n_cond, effective_T).transpose(0, 2, 1)
        return X, Y, Z

    @staticmethod
    @jax.jit
    def _construct_lagged_batch(
        values: jax.Array, indices: jax.Array, lags: jax.Array
    ) -> jax.Array:
        """
        JIT-compiled helper for constructing lagged data in batches.

        This is used internally for efficient parallel construction of
        lagged arrays across multiple variable pairs.
        """

        def get_lagged_value(t_idx, var_idx, lag):
            return values[t_idx - lag, var_idx]

        # Vectorize over time, variable, and lag dimensions
        return jax.vmap(
            jax.vmap(jax.vmap(get_lagged_value, in_axes=(None, None, 0)), in_axes=(None, 0, None)),
            in_axes=(0, None, None),
        )(indices, jnp.arange(values.shape[1]), lags)

    def get_all_pairs_data(
        self, tau_max: int
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Get data structured for parallel testing of all variable pairs.

        This method prepares data in a format optimized for vectorized
        independence testing across all pairs (i, j, tau).

        Parameters
        ----------
        tau_max : int
            Maximum time lag.

        Returns
        -------
        sources : jax.Array
            Source variable values, shape (n_pairs, T_eff).
        targets : jax.Array
            Target variable values, shape (n_pairs, T_eff).
        pair_info : jax.Array
            Array of (i, j, tau) for each pair, shape (n_pairs, 3).

        Notes
        -----
        This method is designed for efficient batch processing with vmap/pmap.
        The pairs are ordered as: for each target j, for each source i != j,
        for each lag tau in 1..tau_max.
        """
        N = self.N
        effective_T = self.T - tau_max

        # Generate all pairs
        pairs = []
        sources_list = []
        targets_list = []

        target_data = self.values[tau_max:]  # Shape (T_eff, N)

        for j in range(N):
            for i in range(N):
                if i == j:
                    continue
                for tau in range(1, tau_max + 1):
                    # Source at t - tau
                    source_data = self.values[tau_max - tau : self.T - tau, i]
                    sources_list.append(source_data)
                    targets_list.append(target_data[:, j])
                    pairs.append([i, j, tau])

        sources = jnp.stack(sources_list, axis=0)
        targets = jnp.stack(targets_list, axis=0)
        pair_info = jnp.array(pairs, dtype=jnp.int32)

        return sources, targets, pair_info

    def summary(self) -> str:
        """
        Generate a summary of the data.

        Returns
        -------
        str
            Human-readable summary of the dataset.
        """
        lines = [
            "=" * 50,
            "JAX-PCMCI DataHandler Summary",
            "=" * 50,
            f"Time points (T): {self.T}",
            f"Variables (N): {self.N}",
            f"Normalized: {self.is_normalized}",
            "",
            "Variables:",
        ]
        for i, name in enumerate(self.var_names):
            var_data = self.values[:, i]
            valid = self._data.mask[:, i] if self._data.mask is not None else jnp.ones(self.T, dtype=bool)
            n_valid = int(jnp.sum(valid))
            mean = float(jnp.nanmean(var_data))
            std = float(jnp.nanstd(var_data))
            lines.append(f"  {i}: {name} (valid: {n_valid}/{self.T}, mean: {mean:.3f}, std: {std:.3f})")

        lines.append("=" * 50)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DataHandler(T={self.T}, N={self.N}, normalized={self.is_normalized})"
