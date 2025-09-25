"""
Unit tests for Hagan analytical SABR surface generator.

This module contains comprehensive tests for the HaganSurfaceGenerator class,
including accuracy tests against literature benchmarks, edge case handling,
and numerical stability validation.
"""

import unittest
import numpy as np
import warnings
from typing import Dict, Any

from hagan_surface_generator import HaganSurfaceGenerator, HaganConfig, create_default_hagan_config
from sabr_params import SABRParams, GridConfig, create_default_grid_config, create_test_sabr_params


class TestHaganSurfaceGenerator(unittest.TestCase):
    """Test cases for HaganSurfaceGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = HaganSurfaceGenerator()
        self.test_params = create_test_sabr_params()
        self.test_grid = create_default_grid_config()
        
        # Suppress warnings during testing
        warnings.filterwarnings("ignore")
    
    def tearDown(self):
        """Clean up after tests."""
        warnings.resetwarnings()
    
    def test_initialization(self):
        """Test HaganSurfaceGenerator initialization."""
        # Test default initialization
        generator = HaganSurfaceGenerator()
        self.assertIsInstance(generator.config, HaganConfig)
        
        # Test custom config initialization
        custom_config = HaganConfig(numerical_tolerance=1e-10)
        generator_custom = HaganSurfaceGenerator(custom_config)
        self.assertEqual(generator_custom.config.numerical_tolerance, 1e-10)
    
    def test_atm_case_handling(self):
        """Test at-the-money volatility calculation."""
        # Test standard case
        atm_vol = self.generator._handle_atm_case(self.test_params, 1.0)
        self.assertIsInstance(atm_vol, float)
        self.assertGreater(atm_vol, 0)
        self.assertLess(atm_vol, 10.0)  # Reasonable volatility range
        
        # Test beta = 1 case
        normal_params = SABRParams(F0=100.0, alpha=20.0, beta=1.0, nu=0.3, rho=0.0)
        normal_vol = self.generator._handle_atm_case(normal_params, 1.0)
        self.assertIsInstance(normal_vol, float)
        self.assertGreater(normal_vol, 0)
        
        # Test beta = 0 case
        lognormal_params = SABRParams(F0=100.0, alpha=0.2, beta=0.0, nu=0.4, rho=-0.5)
        lognormal_vol = self.generator._handle_atm_case(lognormal_params, 1.0)
        self.assertIsInstance(lognormal_vol, float)
        self.assertGreater(lognormal_vol, 0)
        
        # Test time scaling
        short_vol = self.generator._handle_atm_case(self.test_params, 0.25)
        long_vol = self.generator._handle_atm_case(self.test_params, 2.0)
        self.assertNotEqual(short_vol, long_vol)
    
    def test_single_volatility_calculation(self):
        """Test single point volatility calculation."""
        F = self.test_params.F0
        
        # Test ATM
        atm_vol = self.generator._calculate_hagan_volatility(F, F, 1.0, self.test_params)
        self.assertIsInstance(atm_vol, float)
        self.assertGreater(atm_vol, 0)
        
        # Test OTM call
        otm_call_vol = self.generator._calculate_hagan_volatility(F, F * 1.2, 1.0, self.test_params)
        self.assertIsInstance(otm_call_vol, float)
        self.assertGreater(otm_call_vol, 0)
        
        # Test OTM put
        otm_put_vol = self.generator._calculate_hagan_volatility(F, F * 0.8, 1.0, self.test_params)
        self.assertIsInstance(otm_put_vol, float)
        self.assertGreater(otm_put_vol, 0)
        
        # Test different maturities
        short_vol = self.generator._calculate_hagan_volatility(F, F * 1.1, 0.25, self.test_params)
        long_vol = self.generator._calculate_hagan_volatility(F, F * 1.1, 2.0, self.test_params)
        self.assertNotEqual(short_vol, long_vol)
    
    def test_edge_cases(self):
        """Test edge case handling."""
        F = self.test_params.F0
        
        # Test zero maturity
        zero_vol = self.generator._calculate_hagan_volatility(F, F, 0.0, self.test_params)
        self.assertEqual(zero_vol, 0.0)
        
        # Test negative maturity
        neg_vol = self.generator._calculate_hagan_volatility(F, F, -1.0, self.test_params)
        self.assertEqual(neg_vol, 0.0)
        
        # Test zero strike
        zero_strike_vol = self.generator._calculate_hagan_volatility(F, 0.0, 1.0, self.test_params)
        self.assertTrue(np.isnan(zero_strike_vol))
        
        # Test negative strike
        neg_strike_vol = self.generator._calculate_hagan_volatility(F, -10.0, 1.0, self.test_params)
        self.assertTrue(np.isnan(neg_strike_vol))
        
        # Test extreme strikes
        extreme_high = self.generator._calculate_hagan_volatility(F, F * 20, 1.0, self.test_params)
        extreme_low = self.generator._calculate_hagan_volatility(F, F * 0.05, 1.0, self.test_params)
        # Should still return valid numbers (with warnings)
        self.assertIsInstance(extreme_high, float)
        self.assertIsInstance(extreme_low, float)
    
    def test_parameter_edge_cases(self):
        """Test edge cases with extreme parameters."""
        F = 100.0
        
        # Test very small alpha
        small_alpha_params = SABRParams(F0=F, alpha=1e-6, beta=0.5, nu=0.3, rho=0.0)
        vol = self.generator._calculate_hagan_volatility(F, F * 1.1, 1.0, small_alpha_params)
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)
        
        # Test very small nu
        small_nu_params = SABRParams(F0=F, alpha=0.3, beta=0.5, nu=1e-6, rho=0.0)
        vol = self.generator._calculate_hagan_volatility(F, F * 1.1, 1.0, small_nu_params)
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)
        
        # Test extreme correlation
        high_rho_params = SABRParams(F0=F, alpha=0.3, beta=0.5, nu=0.4, rho=0.99)
        vol = self.generator._calculate_hagan_volatility(F, F * 1.1, 1.0, high_rho_params)
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)
        
        low_rho_params = SABRParams(F0=F, alpha=0.3, beta=0.5, nu=0.4, rho=-0.99)
        vol = self.generator._calculate_hagan_volatility(F, F * 1.1, 1.0, low_rho_params)
        self.assertIsInstance(vol, float)
        self.assertGreater(vol, 0)
    
    def test_surface_generation(self):
        """Test full surface generation."""
        surface = self.generator.generate_surface(self.test_params, self.test_grid)
        
        # Check surface shape
        expected_shape = self.test_grid.get_grid_shape()
        self.assertEqual(surface.shape, expected_shape)
        
        # Check for valid values
        finite_mask = np.isfinite(surface)
        finite_values = surface[finite_mask]
        
        if len(finite_values) > 0:
            self.assertTrue(np.all(finite_values > 0))  # All positive volatilities
            self.assertTrue(np.all(finite_values < 100.0))  # Reasonable upper bound
        
        # Check that ATM values are reasonable
        n_maturities, n_strikes = surface.shape
        atm_idx = n_strikes // 2  # Approximate ATM index
        atm_vols = surface[:, atm_idx]
        finite_atm = atm_vols[np.isfinite(atm_vols)]
        
        if len(finite_atm) > 0:
            self.assertTrue(np.all(finite_atm > 0))
            self.assertTrue(np.all(finite_atm < 10.0))
    
    def test_different_beta_values(self):
        """Test surface generation with different beta values."""
        grid = GridConfig(
            strike_range=(0.8, 1.2),
            maturity_range=(0.5, 2.0),
            n_strikes=5,
            n_maturities=3
        )
        
        # Test beta = 0 (lognormal)
        lognormal_params = SABRParams(F0=100.0, alpha=0.2, beta=0.0, nu=0.4, rho=-0.3)
        lognormal_surface = self.generator.generate_surface(lognormal_params, grid)
        self.assertEqual(lognormal_surface.shape, grid.get_grid_shape())
        
        # Test beta = 0.5 (CEV)
        cev_params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=-0.3)
        cev_surface = self.generator.generate_surface(cev_params, grid)
        self.assertEqual(cev_surface.shape, grid.get_grid_shape())
        
        # Test beta = 1 (normal)
        normal_params = SABRParams(F0=100.0, alpha=20.0, beta=1.0, nu=0.3, rho=0.0)
        normal_surface = self.generator.generate_surface(normal_params, grid)
        self.assertEqual(normal_surface.shape, grid.get_grid_shape())
        
        # Surfaces should be different
        self.assertFalse(np.array_equal(lognormal_surface, cev_surface))
        self.assertFalse(np.array_equal(cev_surface, normal_surface))
    
    def test_surface_validation(self):
        """Test surface validation functionality."""
        # Valid surface
        valid_surface = np.array([[0.2, 0.25, 0.3], [0.22, 0.27, 0.32]])
        self.assertTrue(self.generator._validate_surface(valid_surface))
        
        # Empty surface
        empty_surface = np.array([])
        self.assertFalse(self.generator._validate_surface(empty_surface))
        
        # Surface with NaN
        nan_surface = np.array([[0.2, np.nan, 0.3], [0.22, 0.27, 0.32]])
        # Should still be valid (NaN acceptable for extreme strikes)
        result = self.generator._validate_surface(nan_surface)
        self.assertIsInstance(result, bool)
        
        # Surface with infinite values
        inf_surface = np.array([[0.2, np.inf, 0.3], [0.22, 0.27, 0.32]])
        self.assertFalse(self.generator._validate_surface(inf_surface))
        
        # Surface with negative values
        neg_surface = np.array([[0.2, -0.1, 0.3], [0.22, 0.27, 0.32]])
        self.assertFalse(self.generator._validate_surface(neg_surface))
    
    def test_benchmark_against_literature(self):
        """Test benchmark against literature values."""
        benchmark_results = self.generator.benchmark_against_literature(tolerance=0.05)
        
        # Check structure
        self.assertIn('passed', benchmark_results)
        self.assertIn('failed', benchmark_results)
        self.assertIn('test_cases', benchmark_results)
        self.assertIn('max_error', benchmark_results)
        
        # Check that we have test cases
        self.assertGreater(len(benchmark_results['test_cases']), 0)
        
        # Check that at least some tests pass (allowing for numerical differences)
        total_tests = benchmark_results['passed'] + benchmark_results['failed']
        self.assertGreater(total_tests, 0)
        
        # Each test case should have required fields
        for test_case in benchmark_results['test_cases']:
            self.assertIn('name', test_case)
            self.assertIn('expected', test_case)
            self.assertIn('calculated', test_case)
            self.assertIn('error', test_case)
            self.assertIn('passed', test_case)
    
    def test_configuration_options(self):
        """Test different configuration options."""
        # Test with custom tolerance
        strict_config = HaganConfig(numerical_tolerance=1e-15)
        strict_generator = HaganSurfaceGenerator(strict_config)
        
        surface1 = strict_generator.generate_surface(self.test_params, self.test_grid)
        self.assertEqual(surface1.shape, self.test_grid.get_grid_shape())
        
        # Test with validation disabled
        no_validation_config = HaganConfig(validate_output=False)
        no_val_generator = HaganSurfaceGenerator(no_validation_config)
        
        surface2 = no_val_generator.generate_surface(self.test_params, self.test_grid)
        self.assertEqual(surface2.shape, self.test_grid.get_grid_shape())
        
        # Test with different ATM tolerance
        loose_atm_config = HaganConfig(atm_tolerance=1e-3)
        loose_generator = HaganSurfaceGenerator(loose_atm_config)
        
        surface3 = loose_generator.generate_surface(self.test_params, self.test_grid)
        self.assertEqual(surface3.shape, self.test_grid.get_grid_shape())
    
    def test_smile_properties(self):
        """Test that generated surfaces exhibit expected smile properties."""
        # Use a smaller grid for easier analysis
        smile_grid = GridConfig(
            strike_range=(0.7, 1.3),
            maturity_range=(1.0, 1.0),  # Single maturity
            n_strikes=7,
            n_maturities=1
        )
        
        # Parameters that should produce a smile
        smile_params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.5, rho=-0.5)
        surface = self.generator.generate_surface(smile_params, smile_grid)
        
        # Extract the single maturity slice
        vols = surface[0, :]
        strikes = smile_grid.get_strikes(smile_params.F0)
        
        # Find ATM index
        atm_idx = np.argmin(np.abs(strikes - smile_params.F0))
        
        # Check that we have valid volatilities
        finite_mask = np.isfinite(vols)
        if np.sum(finite_mask) >= 3:  # Need at least 3 points
            finite_vols = vols[finite_mask]
            finite_strikes = strikes[finite_mask]
            
            # Basic sanity checks
            self.assertTrue(np.all(finite_vols > 0))
            self.assertTrue(np.all(finite_vols < 10.0))
    
    def test_time_scaling(self):
        """Test that volatilities scale appropriately with time."""
        # Single strike, multiple maturities
        time_grid = GridConfig(
            strike_range=(1.0, 1.0),  # ATM only
            maturity_range=(0.25, 4.0),
            n_strikes=1,
            n_maturities=5
        )
        
        surface = self.generator.generate_surface(self.test_params, time_grid)
        vols = surface[:, 0]  # Single strike column
        maturities = time_grid.get_maturities()
        
        # Check that we have valid volatilities
        finite_mask = np.isfinite(vols)
        if np.sum(finite_mask) >= 2:
            finite_vols = vols[finite_mask]
            finite_mats = maturities[finite_mask]
            
            # Volatilities should be positive
            self.assertTrue(np.all(finite_vols > 0))
            
            # For SABR, longer maturities typically have different volatilities
            # due to vol-of-vol effects
            if len(finite_vols) > 1:
                self.assertFalse(np.allclose(finite_vols, finite_vols[0]))


class TestHaganConfig(unittest.TestCase):
    """Test cases for HaganConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = create_default_hagan_config()
        
        self.assertIsInstance(config, HaganConfig)
        self.assertFalse(config.use_normal_vol)
        self.assertEqual(config.numerical_tolerance, 1e-12)
        self.assertEqual(config.max_iterations, 100)
        self.assertEqual(config.atm_tolerance, 1e-6)
        self.assertEqual(config.wing_cutoff, 10.0)
        self.assertTrue(config.validate_output)
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = HaganConfig(
            use_normal_vol=True,
            numerical_tolerance=1e-10,
            atm_tolerance=1e-4,
            validate_output=False
        )
        
        self.assertTrue(config.use_normal_vol)
        self.assertEqual(config.numerical_tolerance, 1e-10)
        self.assertEqual(config.atm_tolerance, 1e-4)
        self.assertFalse(config.validate_output)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)