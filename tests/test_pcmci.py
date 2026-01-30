"""
Tests for PCMCI and PCMCI+ Algorithms
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax_pcmci import PCMCI, PCMCIPlus, ParCorr, DataHandler
from jax_pcmci.results import PCMCIResults


def generate_simple_var_data(T: int = 500, seed: int = 42):
    """Generate simple VAR(1) data with known causal structure."""
    key = jax.random.PRNGKey(seed)
    N = 3
    
    data = np.zeros((T, N))
    noise = np.array(jax.random.normal(key, (T, N))) * 0.5
    
    # Simple structure: X0(t-1) -> X1(t), X1(t-1) -> X2(t)
    for t in range(1, T):
        data[t, 0] = 0.5 * data[t-1, 0] + noise[t, 0]
        data[t, 1] = 0.7 * data[t-1, 0] + 0.3 * data[t-1, 1] + noise[t, 1]
        data[t, 2] = 0.6 * data[t-1, 1] + noise[t, 2]
    
    ground_truth = {(0, 1, 1), (1, 2, 1), (0, 0, 1), (1, 1, 1)}
    
    return jnp.array(data), ground_truth


class TestDataHandler:
    """Tests for DataHandler class."""
    
    def test_basic_creation(self):
        """Test basic DataHandler creation."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (100, 5))
        handler = DataHandler(data)
        
        assert handler.T == 100
        assert handler.N == 5
        assert len(handler.var_names) == 5
    
    def test_normalization(self):
        """Test that normalization produces zero mean and unit variance."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (1000, 3)) * 5 + 10  # Non-standard data
        
        handler = DataHandler(data, normalize='zscore')
        
        # Check approximately zero mean and unit variance
        means = jnp.mean(handler.values, axis=0)
        stds = jnp.std(handler.values, axis=0)
        
        np.testing.assert_allclose(np.array(means), 0, atol=0.1)
        np.testing.assert_allclose(np.array(stds), 1, atol=0.1)
    
    def test_get_lagged_data(self):
        """Test lagged data construction."""
        data = jnp.arange(20).reshape(10, 2).astype(float)
        handler = DataHandler(data, normalize=False)
        
        X_current, X_lagged = handler.get_lagged_data(tau_max=2)
        
        # Check shapes
        assert X_current.shape == (8, 2)  # T - tau_max
        assert X_lagged.shape == (8, 2, 3)  # (T-tau_max, N, tau_max+1)
    
    def test_get_variable_pair_data(self):
        """Test variable pair data extraction."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (100, 4))
        handler = DataHandler(data, normalize=False)
        
        X, Y, Z = handler.get_variable_pair_data(
            i=0, j=1, tau=2,
            condition_indices=[(2, 1), (3, 1)]
        )
        
        assert X.shape == (98,)  # T - max_lag
        assert Y.shape == (98,)
        assert Z.shape == (98, 2)


class TestPCMCI:
    """Tests for PCMCI algorithm."""

    class _FakeBatchTest:
        """Minimal test double to capture batch conditioning sets."""

        name = "Fake"

        def run_batch(self, X_batch, Y_batch, Z_batch=None, alpha=None, n_conditions=None):
            # Store last Z_batch passed for assertions
            self.last_z_batch = Z_batch
            n = X_batch.shape[0]
            # Return zero statistics/pvalues of correct shape
            stats = jnp.zeros((n,))
            pvals = jnp.ones((n,))
            return stats, pvals
    
    def test_basic_run(self):
        """Test basic PCMCI execution."""
        data, ground_truth = generate_simple_var_data(T=300)
        handler = DataHandler(data)
        
        pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci.run(tau_max=2, pc_alpha=0.1)
        
        assert isinstance(results, PCMCIResults)
        assert results.val_matrix.shape == (3, 3, 3)
        assert results.pval_matrix.shape == (3, 3, 3)
    
    def test_discovers_true_links(self):
        """Test that PCMCI discovers true causal links."""
        data, ground_truth = generate_simple_var_data(T=500)
        handler = DataHandler(data)
        
        pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci.run(tau_max=2, pc_alpha=0.05, alpha_level=0.05)
        
        # Check that main links are discovered
        # X0(t-1) -> X1(t) should be significant
        assert results.is_significant(0, 1, 1)
        
        # X1(t-1) -> X2(t) should be significant
        assert results.is_significant(1, 2, 1)
    
    def test_pc_phase_reduces_parents(self):
        """Test that PC phase reduces the parent set."""
        data, _ = generate_simple_var_data(T=500)
        handler = DataHandler(data)
        
        pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=0)
        
        # Run PC phase
        parents = pcmci.run_pc_stable(tau_max=2, pc_alpha=0.05)
        
        # Should have fewer parents than all possible
        all_possible = 3 * 2  # N variables * tau_max lags per target
        for j in range(3):
            assert len(parents[j]) < all_possible
    
    def test_different_alpha_levels(self):
        """Test that stricter alpha finds fewer links."""
        data, _ = generate_simple_var_data(T=500)
        handler = DataHandler(data)
        
        pcmci = PCMCI(handler, cond_ind_test=ParCorr(), verbosity=0)
        
        results_loose = pcmci.run(tau_max=2, alpha_level=0.1)
        results_strict = pcmci.run(tau_max=2, alpha_level=0.001)
        
        # Stricter alpha should find fewer or equal links
        assert results_strict.n_significant_links <= results_loose.n_significant_links

    def test_batch_mci_respects_condition_limits(self):
        """Batch MCI should honor max_conds_py/px limits."""
        # Simple data; exact values don't matter for this structural test
        data = jnp.arange(20, dtype=float).reshape(10, 2)
        handler = DataHandler(data, normalize=False)

        fake_test = self._FakeBatchTest()
        pcmci = PCMCI(handler, cond_ind_test=fake_test, verbosity=0)

        # Construct parents with one parent each direction at tau=1
        parents = {
            0: {(1, -1)},
            1: {(0, -1)},
        }

        # Run batch MCI with zero conditioning limits
        pcmci.run_batch_mci(
            tau_max=1,
            tau_min=1,
            parents=parents,
            max_conds_py=0,
            max_conds_px=0,
        )

        # With both limits zero, conditioning sets should be empty => Z_batch is None
        assert getattr(fake_test, "last_z_batch", None) is None


class TestPCMCIPlus:
    """Tests for PCMCI+ algorithm."""
    
    def test_basic_run(self):
        """Test basic PCMCI+ execution."""
        data, _ = generate_simple_var_data(T=300)
        handler = DataHandler(data)
        
        pcmci_plus = PCMCIPlus(handler, cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci_plus.run(tau_max=2, tau_min=0)
        
        assert isinstance(results, PCMCIResults)
        assert results.val_matrix.shape == (3, 3, 3)
    
    def test_includes_contemporaneous(self):
        """Test that PCMCI+ tests contemporaneous links."""
        data, _ = generate_simple_var_data(T=300)
        handler = DataHandler(data)
        
        pcmci_plus = PCMCIPlus(handler, cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci_plus.run(tau_max=2, tau_min=0)
        
        # Should have tested tau=0
        contemporaneous = results.get_contemporaneous_links()
        # Either found some or tested and found none
        assert isinstance(contemporaneous, list)


class TestPCMCIResults:
    """Tests for PCMCIResults class."""
    
    def test_summary(self):
        """Test summary generation."""
        val_matrix = jnp.zeros((3, 3, 3))
        pval_matrix = jnp.ones((3, 3, 3))
        
        # Set one significant link
        val_matrix = val_matrix.at[0, 1, 1].set(0.5)
        pval_matrix = pval_matrix.at[0, 1, 1].set(0.001)
        
        results = PCMCIResults(
            val_matrix=val_matrix,
            pval_matrix=pval_matrix,
            var_names=['X0', 'X1', 'X2'],
            alpha_level=0.05,
            test_name='ParCorr',
            tau_max=2,
            tau_min=1
        )
        
        summary = results.summary()
        assert isinstance(summary, str)
        assert "Significant links found: 1" in summary
    
    def test_fdr_correction(self):
        """Test FDR correction reduces significant links."""
        val_matrix = jnp.zeros((3, 3, 3))
        pval_matrix = jnp.ones((3, 3, 3)) * 0.1  # Marginal significance
        
        # Without correction
        results_no_fdr = PCMCIResults(
            val_matrix=val_matrix,
            pval_matrix=pval_matrix,
            var_names=['X0', 'X1', 'X2'],
            alpha_level=0.15,
            fdr_method=None,
            tau_max=2
        )
        
        # With Bonferroni correction
        results_bonf = PCMCIResults(
            val_matrix=val_matrix,
            pval_matrix=pval_matrix,
            var_names=['X0', 'X1', 'X2'],
            alpha_level=0.15,
            fdr_method='bonferroni',
            tau_max=2
        )
        
        # Bonferroni should be more conservative
        assert results_bonf.n_significant_links <= results_no_fdr.n_significant_links
    
    def test_get_parents(self):
        """Test get_parents method."""
        val_matrix = jnp.zeros((3, 3, 2))
        pval_matrix = jnp.ones((3, 3, 2))
        
        # X0(t-1) -> X1(t) and X2(t-1) -> X1(t)
        pval_matrix = pval_matrix.at[0, 1, 1].set(0.001)
        pval_matrix = pval_matrix.at[2, 1, 1].set(0.01)
        
        results = PCMCIResults(
            val_matrix=val_matrix,
            pval_matrix=pval_matrix,
            var_names=['X0', 'X1', 'X2'],
            alpha_level=0.05,
            tau_max=1
        )
        
        parents = results.get_parents(1)
        assert (0, 1) in parents
        assert (2, 1) in parents
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        val_matrix = jnp.zeros((2, 2, 2))
        pval_matrix = jnp.ones((2, 2, 2))
        
        results = PCMCIResults(
            val_matrix=val_matrix,
            pval_matrix=pval_matrix,
            var_names=['X0', 'X1'],
            tau_max=1
        )
        
        d = results.to_dict()
        
        assert 'var_names' in d
        assert 'val_matrix' in d
        assert 'significant_links' in d
        assert d['var_names'] == ['X0', 'X1']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
