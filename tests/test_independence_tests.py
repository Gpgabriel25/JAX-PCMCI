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


class TestBootstrapTesting:
    """Tests for bootstrap-based significance testing."""
    
    def test_bootstrap_pvalue_range(self):
        """Test that bootstrap p-values are in [0, 1]."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100,))
        Y = jax.random.normal(jax.random.PRNGKey(123), (100,))
        
        test = ParCorr(significance='bootstrap', n_permutations=50)
        result = test.run(X, Y)
        
        assert 0 <= result.pvalue <= 1
    
    def test_bootstrap_detects_independence(self):
        """Test that bootstrap gives high p-value for independent variables."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)
        X = jax.random.normal(keys[0], (200,))
        Y = jax.random.normal(keys[1], (200,))
        
        test = ParCorr(significance='bootstrap', n_permutations=100)
        result = test.run(X, Y)
        
        # Should not be significant for truly independent variables
        # Use a very lenient threshold due to randomness
        assert result.pvalue > 0.01
    
    def test_bootstrap_detects_dependence(self):
        """Test that bootstrap gives low p-value for dependent variables."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (200,))
        Y = X * 0.9 + 0.1 * jax.random.normal(key, (200,))
        
        test = ParCorr(significance='bootstrap', n_permutations=100)
        result = test.run(X, Y)
        
        assert result.pvalue < 0.05
        assert result.significant
    
    def test_bootstrap_conditional_independence(self):
        """Test bootstrap handles conditional independence correctly."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)
        
        n = 300
        Z = jax.random.normal(keys[0], (n, 1))  # Confounder
        X = 0.8 * Z[:, 0] + 0.2 * jax.random.normal(keys[1], (n,))
        Y = 0.8 * Z[:, 0] + 0.2 * jax.random.normal(keys[2], (n,))
        
        test = ParCorr(significance='bootstrap', n_permutations=100)
        
        # Without conditioning: should detect spurious correlation
        result_uncond = test.run(X, Y)
        assert result_uncond.significant
        
        # With conditioning on Z: should show conditional independence
        result_cond = test.run(X, Y, Z)
        # The conditional p-value should be higher
        assert result_cond.pvalue > result_uncond.pvalue


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""
    
    def test_perfect_correlation_clipped(self):
        """Test that perfect correlation is handled correctly."""
        X = jnp.array([1., 2., 3., 4., 5.])
        Y = X * 2 + 1  # Perfect linear relationship
        
        test = ParCorr()
        result = test.run(X, Y)
        
        # Statistic should be exactly 1 (or very close)
        assert abs(result.statistic) > 0.999
        # P-value should be very small
        assert result.pvalue < 0.01
    
    def test_constant_variable_handled(self):
        """Test that constant variables don't cause errors."""
        X = jnp.array([1., 2., 3., 4., 5.])
        Y = jnp.ones(5) * 3.0  # Constant
        
        test = ParCorr()
        result = test.run(X, Y)
        
        # Correlation with constant should be 0
        assert abs(result.statistic) < 0.01
    
    def test_cmi_small_k_relative_to_n(self):
        """Test CMI with k close to sample size."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (20,))
        Y = jax.random.normal(jax.random.PRNGKey(1), (20,))
        
        # k=15 is close to n=20
        test = CMIKnn(k=15)
        stat = test.compute_statistic(X, Y)
        
        assert stat >= 0
        assert jnp.isfinite(stat)
    
    def test_parcorr_insufficient_df(self):
        """Test ParCorr returns pval=1.0 when degrees of freedom is insufficient."""
        X = jnp.array([1., 2., 3., 4., 5.])
        Y = jnp.array([2., 4., 5., 4., 5.])
        Z = jnp.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])
        
        test = ParCorr()
        result = test.run(X, Y, Z)
        
        # With n=5 and n_cond=2, df=5-2-3=0 which is insufficient
        # Should return conservative pvalue=1.0
        assert result.pvalue == 1.0
    
    def test_cmi_insufficient_samples_for_k(self):
        """Test CMI returns pval=1.0 when n_samples < 2*k+1."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (15,))
        Y = jax.random.normal(jax.random.PRNGKey(43), (15,))
        
        # k=10, min_samples=2*10+1=21, but we only have 15 samples
        test = CMIKnn(k=10, significance='analytic')
        result = test.run(X, Y)
        
        # Should return conservative pvalue=1.0
        assert result.pvalue == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
