"""Pure JAX implementations for subset sampling and other utilities."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple
from functools import partial


@partial(jax.jit, static_argnums=(1, 2))
def sample_condition_subsets_jax(
    items_flat: jax.Array,  # Shape: (n_items, 2) - flattened (var, lag) pairs
    k: int,
    max_subsets: int,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    JAX-compatible reservoir sampling of k-sized combinations.
    
    Args:
        items_flat: Array of shape (n_items, 2) containing (var, lag) pairs
        k: Size of each subset
        max_subsets: Maximum number of subsets to sample
        key: JAX random key
        
    Returns:
        subsets: Array of shape (actual_count, k, 2) containing sampled subsets
        valid_count: Scalar indicating how many subsets are valid
    """
    n_items = items_flat.shape[0]
    
    # Handle edge cases
    def handle_edge_cases():
        # k == 0 or n_items < k
        empty_subsets = jnp.zeros((1, max(1, k), 2), dtype=jnp.int32)
        return empty_subsets, jnp.array(1 if k == 0 else 0, dtype=jnp.int32)
    
    def sample_subsets():
        # Total possible combinations
        # For reservoir sampling: generate indices using JAX random
        
        # Pre-allocate subset array
        subsets = jnp.zeros((max_subsets, k, 2), dtype=jnp.int32)
        
        # Use JAX random to sample k indices from n_items, max_subsets times
        # This is a simplified approach - we sample with replacement for efficiency
        keys = jax.random.split(key, max_subsets)
        
        def sample_one_subset(key_i):
            # Sample k unique indices from n_items
            indices = jax.random.choice(key_i, n_items, shape=(k,), replace=False)
            return items_flat[indices]
        
        sampled = jax.vmap(sample_one_subset)(keys)
        return sampled, jnp.array(max_subsets, dtype=jnp.int32)
    
    # Conditional execution
    should_handle_edge = (k == 0) | (n_items < k)
    result_subsets, result_count = lax.cond(
        should_handle_edge,
        handle_edge_cases,
        sample_subsets
    )
    
    return result_subsets, result_count


@partial(jax.jit, static_argnums=(2,))
def get_subset_matrix_jax(
    parent_items: jax.Array,  # Shape: (n_parents, 2)
    subset_idx: jax.Array,    # Shape: (k,) - indices into parent_items
    max_conds: int,
) -> jax.Array:
    """
    Extract a subset from parent items and pad to max_conds.
    
    Args:
        parent_items: Array of (var, lag) pairs
        subset_idx: Indices to select
        max_conds: Maximum conditioning set size for padding
        
    Returns:
        padded_subset: Array of shape (max_conds, 2)
    """
    k = subset_idx.shape[0]
    subset = parent_items[subset_idx]
    
    # Pad to max_conds
    padded = jnp.pad(
        subset,
        ((0, max_conds - k), (0, 0)),
        mode='constant',
        constant_values=-1
    )
    
    return padded[:max_conds]
