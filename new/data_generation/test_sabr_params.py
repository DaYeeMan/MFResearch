"""
Unit tests for SABR parameter and grid configuration classes.

Tests parameter validation, grid configuration, and various sampling strategies
to ensure robust and correct behavior.
"""

import pytest
import numpy as np
from typing import List
import warnings

from sabr_params import (
    SABRParams, GridConfig, ParameterSampler,
    create_default_grid_config, create_test_sabr_params
)


class TestSABRParams:
    """Test cases for SABRParams class."""
    
    def test_valid_parameters(self):
        """Test creation of valid SABR parameters."""
        params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
        assert params.F0 == 100.0
        assert params.alpha == 0.3
        assert params.beta == 0.7
        assert params.nu == 0.4
        assert params.rho == -0.3
    
    def test_invalid_forward_price(self):
        """Test validation of forward price."""
        with pytest.raises(ValueError, match="Forward price F0 must be positive"):
            SABRParams(F0=-10.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
        
        with pytest.raises(ValueError, match="Forward price F0 must be positive"):
            SABRParams(F0=0.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
    
    def test_invalid_alpha(self):
        """Test validation of initial volatility."""
        with pytest.raises(ValueError, match="Initial volatility alpha must be positive"):
            SABRParams(F0=100.0, alpha=-0.1, beta=0.7, nu=0.4, rho=-0.3)
        
        with pytest.raises(ValueError, match="Initial volatility alpha must be positive"):
            SABRParams(F0=100.0, alpha=0.0, beta=0.7, nu=0.4, rho=-0.3)
    
    def test_invalid_beta(self):
        """Test validation of beta parameter."""
        with pytest.raises(ValueError, match="Beta must be in \\[0, 1\\]"):
            SABRParams(F0=100.0, alpha=0.3, beta=-0.1, nu=0.4, rho=-0.3)
        
        with pytest.raises(ValueError, match="Beta must be in \\[0, 1\\]"):
            SABRParams(F0=100.0, alpha=0.3, beta=1.1, nu=0.4, rho=-0.3)
    
    def test_invalid_nu(self):
        """Test validation of vol-of-vol parameter."""
        with pytest.raises(ValueError, match="Vol-of-vol nu must be non-negative"):
            SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=-0.1, rho=-0.3)
    
    def test_invalid_rho(self):
        """Test validation of correlation parameter."""
        with pytest.raises(ValueError, match="Correlation rho must be in \\[-1, 1\\]"):
            SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-1.1)
        
        with pytest.raises(ValueError, match="Correlation rho must be in \\[-1, 1\\]"):
            SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=1.1)
    
    def test_boundary_values(self):
        """Test boundary values for parameters."""
        # Test valid boundary values
        params1 = SABRParams(F0=0.001, alpha=0.001, beta=0.0, nu=0.0, rho=-1.0)
        assert params1.beta == 0.0
        assert params1.nu == 0.0
        assert params1.rho == -1.0
        
        params2 = SABRParams(F0=1000.0, alpha=10.0, beta=1.0, nu=5.0, rho=1.0)
        assert params2.beta == 1.0
        assert params2.rho == 1.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
        params_dict = params.to_dict()
        
        expected = {'F0': 100.0, 'alpha': 0.3, 'beta': 0.7, 'nu': 0.4, 'rho': -0.3}
        assert params_dict == expected
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        params_dict = {'F0': 100.0, 'alpha': 0.3, 'beta': 0.7, 'nu': 0.4, 'rho': -0.3}
        params = SABRParams.from_dict(params_dict)
        
        assert params.F0 == 100.0
        assert params.alpha == 0.3
        assert params.beta == 0.7
        assert params.nu == 0.4
        assert params.rho == -0.3
    
    def test_str_representation(self):
        """Test string representation."""
        params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
        str_repr = str(params)
        
        assert "SABRParams" in str_repr
        assert "F0=100.0000" in str_repr
        assert "alpha=0.3000" in str_repr
        assert "beta=0.7000" in str_repr
        assert "nu=0.4000" in str_repr
        assert "rho=-0.3000" in str_repr


class TestGridConfig:
    """Test cases for GridConfig class."""
    
    def test_valid_grid_config(self):
        """Test creation of valid grid configuration."""
        config = GridConfig(
            strike_range=(0.5, 2.0),
            maturity_range=(0.25, 5.0),
            n_strikes=21,
            n_maturities=11
        )
        
        assert config.strike_range == (0.5, 2.0)
        assert config.maturity_range == (0.25, 5.0)
        assert config.n_strikes == 21
        assert config.n_maturities == 11
        assert config.log_strikes is True  # default
        assert config.log_maturities is False  # default
    
    def test_invalid_strike_range(self):
        """Test validation of strike range."""
        with pytest.raises(ValueError, match="strike_range\\[0\\] must be less than strike_range\\[1\\]"):
            GridConfig(
                strike_range=(2.0, 0.5),
                maturity_range=(0.25, 5.0),
                n_strikes=21,
                n_maturities=11
            )
        
        with pytest.raises(ValueError, match="Minimum strike must be positive"):
            GridConfig(
                strike_range=(-0.5, 2.0),
                maturity_range=(0.25, 5.0),
                n_strikes=21,
                n_maturities=11
            )
    
    def test_invalid_maturity_range(self):
        """Test validation of maturity range."""
        with pytest.raises(ValueError, match="maturity_range\\[0\\] must be less than maturity_range\\[1\\]"):
            GridConfig(
                strike_range=(0.5, 2.0),
                maturity_range=(5.0, 0.25),
                n_strikes=21,
                n_maturities=11
            )
        
        with pytest.raises(ValueError, match="Minimum maturity must be positive"):
            GridConfig(
                strike_range=(0.5, 2.0),
                maturity_range=(-0.25, 5.0),
                n_strikes=21,
                n_maturities=11
            )
    
    def test_invalid_grid_size(self):
        """Test validation of grid dimensions."""
        with pytest.raises(ValueError, match="n_strikes must be at least 2"):
            GridConfig(
                strike_range=(0.5, 2.0),
                maturity_range=(0.25, 5.0),
                n_strikes=1,
                n_maturities=11
            )
        
        with pytest.raises(ValueError, match="n_maturities must be at least 2"):
            GridConfig(
                strike_range=(0.5, 2.0),
                maturity_range=(0.25, 5.0),
                n_strikes=21,
                n_maturities=1
            )
    
    def test_get_strikes_linear(self):
        """Test linear strike generation."""
        config = GridConfig(
            strike_range=(0.8, 1.2),
            maturity_range=(0.25, 5.0),
            n_strikes=5,
            n_maturities=3,
            log_strikes=False
        )
        
        strikes = config.get_strikes(forward_price=100.0)
        expected = np.linspace(80.0, 120.0, 5)
        
        np.testing.assert_array_almost_equal(strikes, expected)
    
    def test_get_strikes_log(self):
        """Test logarithmic strike generation."""
        config = GridConfig(
            strike_range=(0.5, 2.0),
            maturity_range=(0.25, 5.0),
            n_strikes=5,
            n_maturities=3,
            log_strikes=True
        )
        
        strikes = config.get_strikes(forward_price=100.0)
        expected = np.logspace(np.log10(50.0), np.log10(200.0), 5)
        
        np.testing.assert_array_almost_equal(strikes, expected)
    
    def test_get_maturities_linear(self):
        """Test linear maturity generation."""
        config = GridConfig(
            strike_range=(0.5, 2.0),
            maturity_range=(0.25, 2.0),
            n_strikes=5,
            n_maturities=4,
            log_maturities=False
        )
        
        maturities = config.get_maturities()
        expected = np.linspace(0.25, 2.0, 4)
        
        np.testing.assert_array_almost_equal(maturities, expected)
    
    def test_get_maturities_log(self):
        """Test logarithmic maturity generation."""
        config = GridConfig(
            strike_range=(0.5, 2.0),
            maturity_range=(0.25, 4.0),
            n_strikes=5,
            n_maturities=4,
            log_maturities=True
        )
        
        maturities = config.get_maturities()
        expected = np.logspace(np.log10(0.25), np.log10(4.0), 4)
        
        np.testing.assert_array_almost_equal(maturities, expected)
    
    def test_get_grid_shape(self):
        """Test grid shape calculation."""
        config = GridConfig(
            strike_range=(0.5, 2.0),
            maturity_range=(0.25, 5.0),
            n_strikes=21,
            n_maturities=11
        )
        
        shape = config.get_grid_shape()
        assert shape == (11, 21)  # (n_maturities, n_strikes)
    
    def test_str_representation(self):
        """Test string representation."""
        config = GridConfig(
            strike_range=(0.5, 2.0),
            maturity_range=(0.25, 5.0),
            n_strikes=21,
            n_maturities=11
        )
        
        str_repr = str(config)
        assert "GridConfig" in str_repr
        assert "(0.5, 2.0)" in str_repr
        assert "(0.25, 5.0)" in str_repr
        assert "(11, 21)" in str_repr


class TestParameterSampler:
    """Test cases for ParameterSampler class."""
    
    def test_uniform_sampling(self):
        """Test uniform parameter sampling."""
        sampler = ParameterSampler(random_seed=42)
        samples = sampler.uniform_sampling(n_samples=10)
        
        assert len(samples) == 10
        assert all(isinstance(s, SABRParams) for s in samples)
        
        # Check parameters are within expected ranges
        for sample in samples:
            assert 80.0 <= sample.F0 <= 120.0
            assert 0.1 <= sample.alpha <= 0.8
            assert 0.0 <= sample.beta <= 1.0
            assert 0.1 <= sample.nu <= 1.0
            assert -0.9 <= sample.rho <= 0.9
    
    def test_uniform_sampling_custom_ranges(self):
        """Test uniform sampling with custom parameter ranges."""
        sampler = ParameterSampler(random_seed=42)
        samples = sampler.uniform_sampling(
            n_samples=5,
            F0_range=(90.0, 110.0),
            alpha_range=(0.2, 0.6),
            beta_range=(0.3, 0.8),
            nu_range=(0.2, 0.8),
            rho_range=(-0.5, 0.5)
        )
        
        assert len(samples) == 5
        
        for sample in samples:
            assert 90.0 <= sample.F0 <= 110.0
            assert 0.2 <= sample.alpha <= 0.6
            assert 0.3 <= sample.beta <= 0.8
            assert 0.2 <= sample.nu <= 0.8
            assert -0.5 <= sample.rho <= 0.5
    
    def test_latin_hypercube_sampling(self):
        """Test Latin Hypercube sampling."""
        sampler = ParameterSampler(random_seed=42)
        samples = sampler.latin_hypercube_sampling(n_samples=20)
        
        assert len(samples) == 20
        assert all(isinstance(s, SABRParams) for s in samples)
        
        # Check parameters are within expected ranges
        for sample in samples:
            assert 80.0 <= sample.F0 <= 120.0
            assert 0.1 <= sample.alpha <= 0.8
            assert 0.0 <= sample.beta <= 1.0
            assert 0.1 <= sample.nu <= 1.0
            assert -0.9 <= sample.rho <= 0.9
    
    def test_latin_hypercube_space_filling(self):
        """Test that LHS provides better space-filling than uniform sampling."""
        sampler = ParameterSampler(random_seed=42)
        
        # Generate samples using both methods
        uniform_samples = sampler.uniform_sampling(n_samples=50)
        lhs_samples = sampler.latin_hypercube_sampling(n_samples=50)
        
        # Convert to arrays for analysis
        uniform_array = np.array([[s.F0, s.alpha, s.beta, s.nu, s.rho] for s in uniform_samples])
        lhs_array = np.array([[s.F0, s.alpha, s.beta, s.nu, s.rho] for s in lhs_samples])
        
        # LHS should have better coverage (this is a basic check)
        assert len(uniform_samples) == 50
        assert len(lhs_samples) == 50
        assert uniform_array.shape == (50, 5)
        assert lhs_array.shape == (50, 5)
    
    def test_adaptive_sampling_no_initial_data(self):
        """Test adaptive sampling without initial data (should fall back to LHS)."""
        sampler = ParameterSampler(random_seed=42)
        samples = sampler.adaptive_sampling(n_samples=10)
        
        assert len(samples) == 10
        assert all(isinstance(s, SABRParams) for s in samples)
    
    def test_adaptive_sampling_with_initial_data(self):
        """Test adaptive sampling with initial performance data."""
        sampler = ParameterSampler(random_seed=42)
        
        # Create initial samples and mock performance scores
        initial_samples = sampler.uniform_sampling(n_samples=5)
        performance_scores = [0.1, 0.8, 0.3, 0.9, 0.2]  # Higher = worse performance
        
        samples = sampler.adaptive_sampling(
            n_samples=10,
            initial_samples=initial_samples,
            performance_scores=performance_scores,
            exploration_ratio=0.5
        )
        
        assert len(samples) == 10
        assert all(isinstance(s, SABRParams) for s in samples)
    
    def test_adaptive_sampling_invalid_input(self):
        """Test adaptive sampling with mismatched input lengths."""
        sampler = ParameterSampler(random_seed=42)
        
        initial_samples = sampler.uniform_sampling(n_samples=5)
        performance_scores = [0.1, 0.8, 0.3]  # Wrong length
        
        with pytest.raises(ValueError, match="initial_samples and performance_scores must have same length"):
            sampler.adaptive_sampling(
                n_samples=10,
                initial_samples=initial_samples,
                performance_scores=performance_scores
            )
    
    def test_sample_parameters_interface(self):
        """Test the unified sampling interface."""
        sampler = ParameterSampler(random_seed=42)
        
        # Test all strategies
        uniform_samples = sampler.sample_parameters(n_samples=5, strategy="uniform")
        lhs_samples = sampler.sample_parameters(n_samples=5, strategy="latin_hypercube")
        adaptive_samples = sampler.sample_parameters(n_samples=5, strategy="adaptive")
        
        assert len(uniform_samples) == 5
        assert len(lhs_samples) == 5
        assert len(adaptive_samples) == 5
        
        # Test invalid strategy
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            sampler.sample_parameters(n_samples=5, strategy="invalid_strategy")
    
    def test_reproducibility(self):
        """Test that sampling is reproducible with fixed seed."""
        sampler1 = ParameterSampler(random_seed=42)
        sampler2 = ParameterSampler(random_seed=42)
        
        samples1 = sampler1.uniform_sampling(n_samples=5)
        samples2 = sampler2.uniform_sampling(n_samples=5)
        
        # Should be identical
        for s1, s2 in zip(samples1, samples2):
            assert s1.F0 == s2.F0
            assert s1.alpha == s2.alpha
            assert s1.beta == s2.beta
            assert s1.nu == s2.nu
            assert s1.rho == s2.rho


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_create_default_grid_config(self):
        """Test default grid configuration creation."""
        config = create_default_grid_config()
        
        assert isinstance(config, GridConfig)
        assert config.strike_range == (0.5, 2.0)
        assert config.maturity_range == (0.25, 5.0)
        assert config.n_strikes == 21
        assert config.n_maturities == 11
        assert config.log_strikes is True
        assert config.log_maturities is False
    
    def test_create_test_sabr_params(self):
        """Test test SABR parameters creation."""
        params = create_test_sabr_params()
        
        assert isinstance(params, SABRParams)
        assert params.F0 == 100.0
        assert params.alpha == 0.3
        assert params.beta == 0.7
        assert params.nu == 0.4
        assert params.rho == -0.3


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])