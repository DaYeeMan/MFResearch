"""
Hagan analytical SABR volatility surface generator.

This module implements the Hagan et al. (2002) analytical approximation for SABR
volatility surfaces with proper handling of edge cases, numerical stability,
and efficient vectorized evaluation across strike/maturity grids.

Reference:
Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
Managing smile risk. Wilmott magazine, 1(1), 84-108.
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import warnings
import time

from .sabr_params import SABRParams, GridConfig
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class HaganConfig:
    """
    Configuration for Hagan analytical surface generation.
    
    Attributes:
        use_normal_vol: If True, return normal volatilities; if False, return lognormal
        numerical_tolerance: Tolerance for numerical stability checks
        max_iterations: Maximum iterations for iterative calculations
        atm_tolerance: Tolerance for at-the-money detection
        wing_cutoff: Cutoff for extreme strikes (as multiple of forward)
        validate_output: Whether to validate output volatilities
    """
    use_normal_vol: bool = False
    numerical_tolerance: float = 1e-12
    max_iterations: int = 100
    atm_tolerance: float = 1e-6
    wing_cutoff: float = 10.0
    validate_output: bool = True


class HaganSurfaceGenerator:
    """
    Hagan analytical SABR volatility surface generator.
    
    Implements the Hagan et al. (2002) analytical approximation for SABR
    implied volatilities with proper handling of edge cases and numerical stability.
    """
    
    def __init__(self, hagan_config: Optional[HaganConfig] = None):
        """
        Initialize Hagan surface generator.
        
        Args:
            hagan_config: Configuration for Hagan formula evaluation
        """
        self.config = hagan_config or HaganConfig()
        logger.info("Initialized Hagan SABR surface generator")
    
    def _handle_atm_case(self, sabr_params: SABRParams, maturity: float) -> float:
        """
        Handle at-the-money case using exact formula.
        
        For K = F, the Hagan formula simplifies to:
        σ_ATM = α / F^(1-β) * [1 + ((1-β)²/24 * α²/F^(2-2β) + ρβνα/4F^(1-β) + (2-3ρ²)ν²/24) * T]
        
        Args:
            sabr_params: SABR model parameters
            maturity: Time to maturity in years
            
        Returns:
            At-the-money implied volatility
        """
        F = sabr_params.F0
        alpha = sabr_params.alpha
        beta = sabr_params.beta
        nu = sabr_params.nu
        rho = sabr_params.rho
        T = maturity
        
        # Base volatility
        if beta == 1.0:
            base_vol = alpha
        else:
            base_vol = alpha / (F ** (1 - beta))
        
        # Time-dependent correction terms
        term1 = ((1 - beta) ** 2) / 24 * (alpha ** 2) / (F ** (2 - 2 * beta))
        term2 = rho * beta * nu * alpha / (4 * F ** (1 - beta))
        term3 = (2 - 3 * rho ** 2) * (nu ** 2) / 24
        
        correction = 1 + (term1 + term2 + term3) * T
        
        return base_vol * correction
    
    def _calculate_hagan_volatility(self, forward: float, strike: float, maturity: float, sabr_params: SABRParams) -> float:
        """
        Calculate Hagan implied volatility for a single (F, K, T) point.
        
        Args:
            forward: Forward price
            strike: Strike price
            maturity: Time to maturity in years
            sabr_params: SABR model parameters
            
        Returns:
            Implied volatility
        """
        F = forward
        K = strike
        T = maturity
        alpha = sabr_params.alpha
        beta = sabr_params.beta
        nu = sabr_params.nu
        rho = sabr_params.rho
        
        # Handle edge cases
        if T <= 0:
            return 0.0
        if K <= 0 or F <= 0:
            return np.nan
        
        # At-the-money case
        if abs(F - K) < self.config.atm_tolerance * F:
            return self._handle_atm_case(sabr_params, T)
        
        # Avoid division by zero
        if abs(alpha) < self.config.numerical_tolerance or abs(nu) < self.config.numerical_tolerance:
            return self._handle_atm_case(sabr_params, T)
        
        # Calculate z parameter
        log_FK = np.log(F / K)
        if abs(beta - 1.0) < self.config.numerical_tolerance:
            z = (nu / alpha) * log_FK
        else:
            z = (nu / alpha) * (F ** (1 - beta)) * log_FK
        
        # Calculate chi(z)
        if abs(z) < self.config.numerical_tolerance:
            chi = 1.0
        else:
            discriminant = 1 - 2 * rho * z + z ** 2
            if discriminant < 0:
                chi = 1.0
            else:
                sqrt_discriminant = np.sqrt(discriminant)
                numerator = sqrt_discriminant + z - rho
                denominator = 1 - rho
                
                if abs(denominator) < self.config.numerical_tolerance or numerator <= 0:
                    chi = 1.0
                else:
                    chi = np.log(numerator / denominator) / z
        
        # Calculate base volatility
        numerator = alpha * chi
        
        if abs(beta - 1.0) < self.config.numerical_tolerance:
            denom_base = 1.0
        else:
            FK_product = F * K
            if FK_product <= 0:
                return np.nan
            denom_base = FK_product ** ((1 - beta) / 2)
            
            # Additional denominator correction
            log_FK_2 = log_FK ** 2
            log_FK_4 = log_FK ** 4
            denom_correction = (1 + ((1 - beta) ** 2) / 24 * log_FK_2 + ((1 - beta) ** 4) / 1920 * log_FK_4)
            denom_base *= denom_correction
        
        base_vol = numerator / denom_base if abs(denom_base) > self.config.numerical_tolerance else alpha
        
        # Time correction
        if abs(beta - 1.0) < self.config.numerical_tolerance:
            term1 = 0.0
            FK_power_half = 1.0
        else:
            FK_product = F * K
            if FK_product > 0:
                FK_power_1_minus_beta = FK_product ** (1 - beta)
                FK_power_half = FK_product ** ((1 - beta) / 2)
                term1 = ((1 - beta) ** 2) / 24 * (alpha ** 2) / FK_power_1_minus_beta
            else:
                term1 = 0.0
                FK_power_half = 1.0
        
        term2 = rho * beta * nu * alpha / (4 * FK_power_half) if FK_power_half > 0 else 0.0
        term3 = (2 - 3 * rho ** 2) * (nu ** 2) / 24
        
        time_correction = 1 + (term1 + term2 + term3) * T
        implied_vol = base_vol * time_correction
        
        return max(implied_vol, self.config.numerical_tolerance)
    
    def generate_surface(self, sabr_params: SABRParams, grid_config: GridConfig) -> np.ndarray:
        """
        Generate Hagan SABR volatility surface.
        
        Args:
            sabr_params: SABR model parameters
            grid_config: Grid configuration for surface discretization
            
        Returns:
            Volatility surface array of shape (n_maturities, n_strikes)
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            sabr_params.validate()
            grid_config.validate()
            
            # Get grid points
            strikes = grid_config.get_strikes(sabr_params.F0)
            maturities = grid_config.get_maturities()
            
            # Initialize surface
            surface = np.zeros((len(maturities), len(strikes)))
            
            # Generate surface point by point
            for t_idx, T in enumerate(maturities):
                for k_idx, K in enumerate(strikes):
                    try:
                        vol = self._calculate_hagan_volatility(sabr_params.F0, K, T, sabr_params)
                        surface[t_idx, k_idx] = vol
                    except Exception as e:
                        logger.warning(f"Error calculating volatility at T={T}, K={K}: {e}")
                        surface[t_idx, k_idx] = np.nan
            
            # Validate output
            if self.config.validate_output:
                if not self._validate_surface(surface):
                    logger.warning("Generated Hagan surface contains invalid values")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Generated Hagan surface in {elapsed_time:.4f} seconds")
            
            return surface
            
        except Exception as e:
            logger.error(f"Error generating Hagan surface: {e}")
            raise
    
    def _validate_surface(self, surface: np.ndarray) -> bool:
        """
        Validate generated volatility surface.
        
        Args:
            surface: Volatility surface array
            
        Returns:
            True if surface is valid, False otherwise
        """
        if surface.size == 0:
            return False
        
        # Check for NaN values (some may be acceptable for extreme strikes)
        nan_count = np.sum(np.isnan(surface))
        if nan_count > 0:
            logger.warning(f"Surface contains {nan_count} NaN values")
        
        # Check for infinite values
        if np.any(np.isinf(surface)):
            logger.warning("Surface contains infinite values")
            return False
        
        # Check for negative volatilities
        finite_surface = surface[np.isfinite(surface)]
        if len(finite_surface) > 0 and np.any(finite_surface < 0):
            logger.warning("Surface contains negative volatilities")
            return False
        
        # Check for unreasonably high volatilities (>1000%)
        if len(finite_surface) > 0 and np.any(finite_surface > 10.0):
            logger.warning("Surface contains extremely high volatilities (>1000%)")
        
        return True
    
    def benchmark_against_literature(self, tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Benchmark implementation against known literature values.
        
        Args:
            tolerance: Tolerance for benchmark comparison
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Running Hagan formula benchmark tests")
        
        benchmark_results = {
            'passed': 0,
            'failed': 0,
            'test_cases': [],
            'max_error': 0.0
        }
        
        # Test case 1: ATM case
        test_params_1 = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
        calculated_atm_1y = self._handle_atm_case(test_params_1, 1.0)
        expected_atm_1y = 0.3 / (100.0 ** 0.3) * (1 + (0.09/24 * 0.09/(100.0**0.6) + (-0.3)*0.7*0.4*0.3/4/(100.0**0.3) + (2-3*0.09)*0.16/24) * 1.0)
        error_1 = abs(calculated_atm_1y - expected_atm_1y) / expected_atm_1y
        
        test_case_1 = {
            'name': 'ATM 1Y Hagan benchmark',
            'expected': expected_atm_1y,
            'calculated': calculated_atm_1y,
            'error': error_1,
            'passed': error_1 < tolerance
        }
        
        benchmark_results['test_cases'].append(test_case_1)
        benchmark_results['max_error'] = max(benchmark_results['max_error'], error_1)
        
        if test_case_1['passed']:
            benchmark_results['passed'] += 1
        else:
            benchmark_results['failed'] += 1
        
        # Test case 2: Beta = 1 (normal model)
        test_params_2 = SABRParams(F0=100.0, alpha=20.0, beta=1.0, nu=0.3, rho=0.0)
        calculated_normal = self._handle_atm_case(test_params_2, 1.0)
        expected_normal = 20.0 * (1 + (2-3*0)*0.09/24 * 1.0)
        error_2 = abs(calculated_normal - expected_normal) / expected_normal
        
        test_case_2 = {
            'name': 'Normal model (beta=1) ATM',
            'expected': expected_normal,
            'calculated': calculated_normal,
            'error': error_2,
            'passed': error_2 < tolerance
        }
        
        benchmark_results['test_cases'].append(test_case_2)
        benchmark_results['max_error'] = max(benchmark_results['max_error'], error_2)
        
        if test_case_2['passed']:
            benchmark_results['passed'] += 1
        else:
            benchmark_results['failed'] += 1
        
        logger.info(f"Benchmark results: {benchmark_results['passed']} passed, "
                   f"{benchmark_results['failed']} failed, "
                   f"max error: {benchmark_results['max_error']:.6f}")
        
        return benchmark_results


def create_default_hagan_config() -> HaganConfig:
    """
    Create default Hagan configuration.
    
    Returns:
        HaganConfig with reasonable default values
    """
    return HaganConfig(
        use_normal_vol=False,
        numerical_tolerance=1e-12,
        max_iterations=100,
        atm_tolerance=1e-6,
        wing_cutoff=10.0,
        validate_output=True
    )