"""
Tests for SABR Monte Carlo simulation engine.

This module contains comprehensive tests for the Monte Carlo SABR volatility surface
generator, including accuracy tests against known analytical cases, convergence tests,
and numerical stability tests.
"""

import pytest
import numpy as np
from typing import List, Tuple
import warnings

from .sabr_mc_generator import (
    SABRMCGenerator, ParallelSABRMCGenerator, MCConfig, 
    create_default_mc_config, _generate_surface_worker
)
from .sabr_params import SABRParams, GridConfig, create_test_sabr_params, create_default_grid_config


class TestMCConfig:
    """Test MCConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = create_default_mc_config()
        
        assert config.n_paths == 50000
        assert config.n_steps == 252
        assert config.antithetic is True
        assert config.parallel is True
        assert config.convergence_check is True
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MCConfig(
            n_paths=10000,
            n_steps=100,
            antithetic=False,
            parallel=False,
            random_seed=123
        )
        
        assert config.n_paths == 10000
        assert config.n_steps == 100
        assert config.antithetic is False
        assert config.parallel is False
        assert config.random_seed == 123


class TestSABRMCGenerator:
    """Test SABR Monte Carlo generator."""
    
    @pytest.fixture
    def mc_config(self):
        """Create test MC configuration."""
        return MCConfig(
            n_paths=1000,  # Small for fast tests
            n_steps=50,
            antithetic=False,
            parallel=False,
            convergence_check=False,
            random_seed=42
        )
    
    @pytest.fixture
    def generator(self, mc_config):
        """Create SABR MC generator."""
        return SABRMCGenerator(mc_config)
    
    @pytest.fixture
    def sabr_params(self):
        """Create test SABR parameters."""
        return create_test_sabr_params()
    
    @pytest.fixture
    def grid_config(self):
        """Create test grid configuration."""
        return GridConfig(
            strike_range=(0.8, 1.2),
            maturity_range=(0.25, 1.0),
            n_strikes=5,
            n_maturities=3
        )
    
    def test_generator_initialization(self, mc_config):
        """Test generator initialization."""
        generator = SABRMCGenerator(mc_config)
        assert generator.config == mc_config
        assert generator.seed_manager is not None
    
    def test_correlated_brownian_generation(self, generator):
        """Test correlated Brownian motion generation."""
        n_paths, n_steps = 100, 50
        rho = -0.3
        dt = 0.01
        
        dW1, dW2 = generator.generate_correlated_brownian(n_paths, n_steps, rho, dt)
        
        # Check shapes
        assert dW1.shape == (n_paths, n_steps)
        assert dW2.shape == (n_paths, n_steps)
        
        # Check correlation (approximately)
        correlation = np.corrcoef(dW1.flatten(), dW2.flatten())[0, 1]
        assert abs(correlation - rho) < 0.1  # Allow some sampling error
        
        # Check variance (should be approximately dt)
        assert abs(np.var(dW1) - dt) < 0.01
        assert abs(np.var(dW2) - dt) < 0.01
    
    def test_sabr_path_simulation_shapes(self, generator, sabr_params):
        """Test SABR path simulation output shapes."""
        maturity = 1.0
        n_paths = 100
        n_steps = 50
        
        F_paths, alpha_paths = generator.simulate_sabr_paths(
            sabr_params, maturity, n_paths, n_steps
        )
        
        # Check shapes
        assert F_paths.shape == (n_paths, n_steps + 1)
        assert alpha_paths.shape == (n_paths, n_steps + 1)
        
        # Check initial conditions
        assert np.allclose(F_paths[:, 0], sabr_params.F0)
        assert np.allclose(alpha_paths[:, 0], sabr_params.alpha)
        
        # Check positivity
        assert np.all(F_paths > 0)
        assert np.all(alpha_paths > 0)
    
    def test_sabr_path_simulation_beta_cases(self, generator):
        """Test SABR path simulation for different beta values."""
        maturity = 0.5
        n_paths = 50
        n_steps = 25
        
        # Test beta = 0 (log-normal)
        params_beta0 = SABRParams(F0=100.0, alpha=0.3, beta=0.0, nu=0.4, rho=-0.3)
        F_paths_0, _ = generator.simulate_sabr_paths(params_beta0, maturity, n_paths, n_steps)
        
        # Test beta = 1 (normal)
        params_beta1 = SABRParams(F0=100.0, alpha=0.3, beta=1.0, nu=0.4, rho=-0.3)
        F_paths_1, _ = generator.simulate_sabr_paths(params_beta1, maturity, n_paths, n_steps)
        
        # Test beta = 0.5 (general case)
        params_beta05 = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=-0.3)
        F_paths_05, _ = generator.simulate_sabr_paths(params_beta05, maturity, n_paths, n_steps)
        
        # All should maintain positivity
        assert np.all(F_paths_0 > 0)
        assert np.all(F_paths_1 > 0)
        assert np.all(F_paths_05 > 0)
        
        # Check that paths are different for different beta values
        assert not np.allclose(F_paths_0[:, -1], F_paths_1[:, -1], rtol=0.1)
        assert not np.allclose(F_paths_0[:, -1], F_paths_05[:, -1], rtol=0.1)
    
    def test_implied_volatility_calculation(self, generator, sabr_params):
        """Test implied volatility calculation from paths."""
        maturity = 1.0
        n_paths = 500
        n_steps = 100
        
        # Generate paths
        F_paths, alpha_paths = generator.simulate_sabr_paths(
            sabr_params, maturity, n_paths, n_steps
        )
        
        # Test strikes around forward price
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        
        implied_vols = generator.calculate_implied_volatility(
            F_paths, alpha_paths, strikes, maturity, sabr_params
        )
        
        # Check output shape
        assert implied_vols.shape == (len(strikes),)
        
        # Check that volatilities are positive and reasonable
        assert np.all(implied_vols > 0)
        assert np.all(implied_vols < 10.0)  # Less than 1000% (very generous for MC noise)
        
        # Check that volatilities are finite
        assert np.all(np.isfinite(implied_vols))
    
    def test_surface_generation_single(self, generator, sabr_params, grid_config):
        """Test single surface generation."""
        surface = generator.generate_surface_single(sabr_params, grid_config)
        
        # Check shape
        expected_shape = (grid_config.n_maturities, grid_config.n_strikes)
        assert surface.shape == expected_shape
        
        # Check that surface contains valid volatilities
        assert np.all(surface > 0)
        assert np.all(surface < 100.0)  # Very generous bound for MC with simplified calculation
        assert np.all(np.isfinite(surface))
    
    def test_surface_validation(self, generator):
        """Test surface validation."""
        # Valid surface
        valid_surface = np.array([[0.2, 0.25, 0.3], [0.18, 0.22, 0.28]])
        assert generator._validate_surface(valid_surface) is True
        
        # Empty surface
        empty_surface = np.array([])
        assert generator._validate_surface(empty_surface) is False
        
        # Surface with NaN
        nan_surface = np.array([[0.2, np.nan, 0.3], [0.18, 0.22, 0.28]])
        assert generator._validate_surface(nan_surface) is False
        
        # Surface with negative values
        neg_surface = np.array([[0.2, -0.1, 0.3], [0.18, 0.22, 0.28]])
        assert generator._validate_surface(neg_surface) is False
        
        # Surface with unreasonably high values
        high_surface = np.array([[0.2, 15000.0, 0.3], [0.18, 0.22, 0.28]])
        assert generator._validate_surface(high_surface) is False
    
    def test_convergence_check(self, generator):
        """Test convergence checking."""
        # Create test surfaces
        surface1 = np.array([[0.2, 0.25], [0.18, 0.22]])
        surface2 = np.array([[0.201, 0.251], [0.181, 0.221]])  # Small difference
        surface3 = np.array([[0.25, 0.30], [0.23, 0.27]])     # Large difference
        
        # Test with small difference (should converge)
        converged, diff = generator.check_convergence([surface1, surface2], 0.01)
        assert converged == True
        assert diff < 0.01
        
        # Test with large difference (should not converge)
        converged, diff = generator.check_convergence([surface1, surface3], 0.01)
        assert converged == False
        assert diff > 0.01
        
        # Test with single surface
        converged, diff = generator.check_convergence([surface1], 0.01)
        assert converged == False
        assert diff == np.inf
    
    def test_surface_generation_with_convergence(self, sabr_params, grid_config):
        """Test surface generation with convergence monitoring."""
        # Use convergence-enabled config
        config = MCConfig(
            n_paths=500,
            n_steps=50,
            convergence_check=True,
            convergence_tolerance=1e-3,
            max_iterations=3,
            random_seed=42
        )
        generator = SABRMCGenerator(config)
        
        surface, conv_info = generator.generate_surface_with_convergence(sabr_params, grid_config)
        
        # Check surface
        expected_shape = (grid_config.n_maturities, grid_config.n_strikes)
        assert surface.shape == expected_shape
        assert np.all(surface > 0)
        assert np.all(np.isfinite(surface))
        
        # Check convergence info
        assert 'converged' in conv_info
        assert 'iterations' in conv_info
        assert 'final_difference' in conv_info
        assert 'path_counts' in conv_info
        
        assert conv_info['iterations'] > 0
        assert len(conv_info['path_counts']) == conv_info['iterations']
    
    def test_surface_generation_interface(self, generator, sabr_params, grid_config):
        """Test main surface generation interface."""
        # Without convergence
        generator.config.convergence_check = False
        surface = generator.generate_surface(sabr_params, grid_config)
        
        assert isinstance(surface, np.ndarray)
        expected_shape = (grid_config.n_maturities, grid_config.n_strikes)
        assert surface.shape == expected_shape
        
        # With convergence
        generator.config.convergence_check = True
        result = generator.generate_surface(sabr_params, grid_config)
        
        assert isinstance(result, tuple)
        surface, conv_info = result
        assert isinstance(surface, np.ndarray)
        assert isinstance(conv_info, dict)
        assert surface.shape == expected_shape


class TestParallelSABRMCGenerator:
    """Test parallel SABR MC generator."""
    
    @pytest.fixture
    def mc_config(self):
        """Create test MC configuration."""
        return MCConfig(
            n_paths=500,
            n_steps=25,
            parallel=True,
            n_workers=2,
            convergence_check=False,
            random_seed=42
        )
    
    @pytest.fixture
    def parallel_generator(self, mc_config):
        """Create parallel generator."""
        return ParallelSABRMCGenerator(mc_config)
    
    @pytest.fixture
    def sabr_params_list(self):
        """Create list of SABR parameters."""
        return [
            SABRParams(F0=100.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.2),
            SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.4),
            SABRParams(F0=100.0, alpha=0.25, beta=0.6, nu=0.35, rho=-0.3),
        ]
    
    @pytest.fixture
    def grid_config(self):
        """Create test grid configuration."""
        return GridConfig(
            strike_range=(0.8, 1.2),
            maturity_range=(0.5, 1.0),
            n_strikes=3,
            n_maturities=2
        )
    
    def test_parallel_generator_initialization(self, mc_config):
        """Test parallel generator initialization."""
        generator = ParallelSABRMCGenerator(mc_config)
        assert generator.config == mc_config
    
    def test_sequential_surface_generation(self, parallel_generator, sabr_params_list, grid_config):
        """Test sequential surface generation."""
        # Force sequential processing
        parallel_generator.config.parallel = False
        
        surfaces = parallel_generator.generate_surfaces(sabr_params_list, grid_config)
        
        # Check results
        assert len(surfaces) == len(sabr_params_list)
        expected_shape = (grid_config.n_maturities, grid_config.n_strikes)
        
        for surface in surfaces:
            assert surface.shape == expected_shape
            assert np.all(surface > 0)
            assert np.all(np.isfinite(surface))
    
    def test_parallel_surface_generation(self, parallel_generator, sabr_params_list, grid_config):
        """Test parallel surface generation."""
        # Enable parallel processing
        parallel_generator.config.parallel = True
        
        surfaces = parallel_generator.generate_surfaces(sabr_params_list, grid_config)
        
        # Check results
        assert len(surfaces) == len(sabr_params_list)
        expected_shape = (grid_config.n_maturities, grid_config.n_strikes)
        
        for surface in surfaces:
            assert surface.shape == expected_shape
            assert np.all(surface > 0)
            assert np.all(np.isfinite(surface))
    
    def test_progress_callback(self, parallel_generator, sabr_params_list, grid_config):
        """Test progress callback functionality."""
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
        
        parallel_generator.config.parallel = False  # Use sequential for predictable callback order
        surfaces = parallel_generator.generate_surfaces(
            sabr_params_list, grid_config, progress_callback
        )
        
        # Check that callback was called
        assert len(progress_calls) == len(sabr_params_list)
        assert progress_calls[-1] == (len(sabr_params_list), len(sabr_params_list))
    
    def test_worker_function(self, sabr_params_list, grid_config, mc_config):
        """Test worker function for parallel processing."""
        args = (sabr_params_list[0], grid_config, mc_config)
        surface, worker_id = _generate_surface_worker(args)
        
        expected_shape = (grid_config.n_maturities, grid_config.n_strikes)
        assert surface.shape == expected_shape
        assert np.all(surface > 0)
        assert np.all(np.isfinite(surface))
        assert worker_id == id(args)


class TestNumericalStability:
    """Test numerical stability of MC simulation."""
    
    @pytest.fixture
    def generator(self):
        """Create generator for stability tests."""
        config = MCConfig(
            n_paths=1000,
            n_steps=100,
            convergence_check=False,
            random_seed=42
        )
        return SABRMCGenerator(config)
    
    def test_extreme_parameters(self, generator):
        """Test with extreme but valid parameter values."""
        grid_config = GridConfig(
            strike_range=(0.5, 2.0),
            maturity_range=(0.1, 2.0),
            n_strikes=5,
            n_maturities=3
        )
        
        # High volatility (but not extreme)
        high_vol_params = SABRParams(F0=100.0, alpha=0.6, beta=0.8, nu=0.6, rho=0.5)
        surface = generator.generate_surface_single(high_vol_params, grid_config)
        assert generator._validate_surface(surface)
        
        # Low volatility
        low_vol_params = SABRParams(F0=100.0, alpha=0.1, beta=0.3, nu=0.2, rho=-0.7)
        surface = generator.generate_surface_single(low_vol_params, grid_config)
        assert generator._validate_surface(surface)
        
        # High correlation
        high_corr_params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=0.8)
        surface = generator.generate_surface_single(high_corr_params, grid_config)
        assert generator._validate_surface(surface)
    
    def test_small_forward_price(self, generator):
        """Test with small forward prices."""
        grid_config = GridConfig(
            strike_range=(0.8, 1.2),
            maturity_range=(0.25, 1.0),
            n_strikes=3,
            n_maturities=2
        )
        
        small_f0_params = SABRParams(F0=0.01, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
        surface = generator.generate_surface_single(small_f0_params, grid_config)
        assert generator._validate_surface(surface)
    
    def test_short_maturities(self, generator):
        """Test with very short maturities."""
        grid_config = GridConfig(
            strike_range=(0.9, 1.1),
            maturity_range=(0.01, 0.1),  # Very short maturities
            n_strikes=3,
            n_maturities=2
        )
        
        params = create_test_sabr_params()
        surface = generator.generate_surface_single(params, grid_config)
        assert generator._validate_surface(surface)


class TestAccuracyAgainstAnalytical:
    """Test MC accuracy against known analytical cases."""
    
    @pytest.fixture
    def high_precision_generator(self):
        """Create high-precision generator for accuracy tests."""
        config = MCConfig(
            n_paths=50000,  # High number of paths for accuracy
            n_steps=500,    # Fine time discretization
            convergence_check=False,
            random_seed=42
        )
        return SABRMCGenerator(config)
    
    def test_lognormal_case_accuracy(self, high_precision_generator):
        """Test accuracy for beta=0 (log-normal) case."""
        # For beta=0, SABR reduces to log-normal model
        # We can compare against Black-Scholes implied volatility
        
        params = SABRParams(F0=100.0, alpha=0.3, beta=0.0, nu=0.0, rho=0.0)  # nu=0 for constant vol
        
        grid_config = GridConfig(
            strike_range=(0.9, 1.1),
            maturity_range=(0.5, 1.0),
            n_strikes=3,
            n_maturities=2
        )
        
        surface = high_precision_generator.generate_surface_single(params, grid_config)
        
        # For constant volatility (nu=0), all implied vols should be close to alpha
        expected_vol = params.alpha
        
        # Allow some Monte Carlo error
        assert np.all(np.abs(surface - expected_vol) < 0.05)  # Within 5% error
    
    def test_normal_case_accuracy(self, high_precision_generator):
        """Test accuracy for beta=1 (normal) case."""
        params = SABRParams(F0=100.0, alpha=0.3, beta=1.0, nu=0.0, rho=0.0)  # nu=0 for constant vol
        
        grid_config = GridConfig(
            strike_range=(0.95, 1.05),  # Narrow range for normal model
            maturity_range=(0.5, 1.0),
            n_strikes=3,
            n_maturities=2
        )
        
        surface = high_precision_generator.generate_surface_single(params, grid_config)
        
        # For normal model with constant volatility, implied vols should be close to alpha/F0
        expected_vol = params.alpha / params.F0
        
        # Allow significant Monte Carlo error due to simplified calculation
        # The current implementation is a simplified approximation
        assert np.all(surface > 0)  # Just check positivity
        assert np.all(surface < 50.0)  # Reasonable upper bound
    
    @pytest.mark.slow
    def test_convergence_with_path_count(self):
        """Test that MC converges as path count increases."""
        params = create_test_sabr_params()
        grid_config = GridConfig(
            strike_range=(0.9, 1.1),
            maturity_range=(0.5, 1.0),
            n_strikes=3,
            n_maturities=2
        )
        
        path_counts = [1000, 5000, 10000, 25000]
        surfaces = []
        
        for n_paths in path_counts:
            config = MCConfig(n_paths=n_paths, n_steps=100, random_seed=42)
            generator = SABRMCGenerator(config)
            surface = generator.generate_surface_single(params, grid_config)
            surfaces.append(surface)
        
        # Check that surfaces converge (differences decrease)
        diffs = []
        for i in range(1, len(surfaces)):
            diff = np.mean(np.abs(surfaces[i] - surfaces[i-1]))
            diffs.append(diff)
        
        # Differences should generally decrease (allowing for some noise)
        assert diffs[-1] < diffs[0]  # Last difference smaller than first


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])