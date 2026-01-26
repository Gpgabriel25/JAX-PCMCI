"""
Tests for JAX-PCMCI Independence Tests
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)

from jax_pcmci.independence_tests import ParCorr, CMIKnn, GPDCond
from jax_pcmci.independence_tests.base import TestResult


class TestParCorr:
    """Tests for the ParCorr independence test."""
    
    def test_perfect_correlation(self):
        """Test that perfectly correlated variables have correlation ~1."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100,))
        Y = X * 2 + 1  # Perfect linear relationship
        
        test = ParCorr()
        result = test.run(X, Y)
        
        assert abs(result.statistic) > 0.99
        assert result.pvalue < 0.001
        assert result.significant
    
    def test_independent_variables(self):
        """Test that independent variables have low correlation."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        X = jax.random.normal(keys[0], (500,))
        Y = jax.random.normal(keys[1], (500,))
        
        test = ParCorr()
        result = test.run(X, Y)
        
        assert abs(result.statistic) < 0.15
        assert result.pvalue > 0.05
    
    def test_partial_correlation(self):
        """Test partial correlation with confounding variable."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        n = 500
        Z = jax.random.normal(keys[0], (n, 1))  # Confounder
        X = 0.8 * Z[:, 0] + 0.2 * jax.random.normal(keys[1], (n,))
        Y = 0.8 * Z[:, 0] + 0.2 * jax.random.normal(keys[2], (n,))
        
        test = ParCorr()
        
        # Without conditioning: should show correlation
        result_uncond = test.run(X, Y)
        assert abs(result_uncond.statistic) > 0.5
        
        # With conditioning on Z: correlation should decrease
        result_cond = test.run(X, Y, Z)
        assert abs(result_cond.statistic) < abs(result_uncond.statistic)
    
    def test_batch_computation(self):
        """Test batch computation produces same results as individual."""
        key = jax.random.PRNGKey(42)
        n_tests = 10
        n_samples = 100
        
        X_batch = jax.random.normal(key, (n_tests, n_samples))
        keys = jax.random.split(key, n_tests)
        Y_batch = jax.random.normal(jax.random.PRNGKey(123), (n_tests, n_samples))
        
        test = ParCorr()
        
        # Batch computation
        stats_batch, pvals_batch = test.run_batch(X_batch, Y_batch)
        
        # Individual computations
        stats_individual = []
        for i in range(n_tests):
            result = test.run(X_batch[i], Y_batch[i])
            stats_individual.append(result.statistic)
        
        # Compare
        np.testing.assert_allclose(
            np.array(stats_batch),
            np.array(stats_individual),
            rtol=1e-5
        )
    
    def test_result_structure(self):
        """Test that result has correct structure."""
        X = jnp.array([1., 2., 3., 4., 5.])
        Y = jnp.array([2., 4., 5., 4., 5.])
        
        test = ParCorr(alpha=0.1)
        result = test.run(X, Y)
        
        assert isinstance(result, TestResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.significant, bool)
        assert result.alpha == 0.1
        assert result.test_name == "ParCorr"
        assert 0 <= result.pvalue <= 1
        assert -1 <= result.statistic <= 1


class TestCMIKnn:
    """Tests for the CMI-kNN independence test."""
    
    def test_independent_variables(self):
        """Test that CMI is near zero for independent variables."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        X = jax.random.normal(keys[0], (200,))
        Y = jax.random.normal(keys[1], (200,))
        
        test = CMIKnn(k=5, significance='permutation', n_permutations=100)
        result = test.run(X, Y)
        
        # CMI should be small for independent variables
        assert result.statistic < 0.2
    
    def test_dependent_variables(self):
        """Test that CMI detects dependent variables."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (200,))
        Y = jnp.sin(X) + 0.1 * jax.random.normal(key, (200,))  # Nonlinear dependence
        
        test = CMIKnn(k=5, significance='permutation', n_permutations=100)
        result = test.run(X, Y)
        
        # CMI should be larger for dependent variables
        assert result.statistic > 0.1
    
    def test_cmi_non_negative(self):
        """Test that CMI is always non-negative."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100,))
        Y = jax.random.normal(jax.random.PRNGKey(123), (100,))
        
        test = CMIKnn(k=5)
        stat = test.compute_statistic(X, Y)
        
        assert stat >= 0
    
    def test_different_k_values(self):
        """Test that different k values work."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100,))
        Y = X + 0.1 * jax.random.normal(key, (100,))
        
        for k in [3, 5, 10, 20]:
            test = CMIKnn(k=k)
            stat = test.compute_statistic(X, Y)
            assert stat >= 0


class TestGPDCond:
    """Tests for the GPDC independence test."""
    
    def test_distance_correlation(self):
        """Test distance correlation for dependent variables."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100,))
        Y = X ** 2 + 0.1 * jax.random.normal(key, (100,))  # Quadratic dependence
        
        test = GPDCond()
        stat = test.compute_statistic(X, Y)
        
        # Should detect nonlinear dependence
        assert stat > 0.3
    
    def test_independent_variables(self):
        """Test that GPDC is low for independent variables."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        X = jax.random.normal(keys[0], (100,))
        Y = jax.random.normal(keys[1], (100,))
        
        test = GPDCond()
        stat = test.compute_statistic(X, Y)
        
        # Should be small for independent
        assert stat < 0.3
    
    def test_different_kernels(self):
        """Test that different kernels work."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (50,))
        Y = X + 0.1 * jax.random.normal(key, (50,))
        Z = jax.random.normal(jax.random.PRNGKey(123), (50, 2))
        
        for kernel in ['rbf', 'matern32', 'matern52']:
            test = GPDCond(kernel=kernel)
            stat = test.compute_statistic(X, Y, Z)
            assert stat >= 0


class TestPermutationTesting:
    """Tests for permutation-based significance testing."""
    
    def test_permutation_pvalue_range(self):
        """Test that permutation p-values are in [0, 1]."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100,))
        Y = jax.random.normal(jax.random.PRNGKey(123), (100,))
        
        test = ParCorr(significance='permutation', n_permutations=50)
        result = test.run(X, Y)
        
        assert 0 <= result.pvalue <= 1
    
    def test_permutation_significant_for_dependent(self):
        """Test that permutation testing detects strong dependence."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (200,))
        Y = X * 0.9 + 0.1 * jax.random.normal(key, (200,))
        
        test = ParCorr(significance='permutation', n_permutations=100)
        result = test.run(X, Y)
        
        assert result.pvalue < 0.05
        assert result.significant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
