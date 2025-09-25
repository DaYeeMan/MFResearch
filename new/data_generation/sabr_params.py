"""
SABR parameter and grid configuration classes with validation and sampling strategies.

This module provides the core data structures for SABR volatility surface modeling,
including parameter validation, grid configuration, and various sampling strategies
for parameter space exploration.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Dict, Any
from scipy.stats import qmc
import warnings


@dataclass
class SABRParams:
    """
    SABR model parameters with validation.
    
    The SABR model is defined by the SDE:
    dF_t = α_t F_t^β dW_1(t)
    dα_t = ν α_t dW_2(t)
    
    where dW_1(t) dW_2(t) = ρ dt
    
    Attributes:
        F0: Forward price (must be positive)
        alpha: Initial volatility (must be positive)
        beta: Elasticity parameter (must be in [0, 1])
        nu: Vol-of-vol parameter (must be non-negative)
        rho: Correlation parameter (must be in [-1, 1])
    """
    F0: float
    alpha: float
    beta: float
    nu: float
    rho: float
    
    def __post_init__(self):
        """Validate SABR parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate SABR parameters are within acceptable ranges.
        
        Raises:
            ValueError: If any parameter is outside acceptable bounds
        """
        if self.F0 <= 0:
            raise ValueError(f"Forward price F0 must be positive, got {self.F0}")
        
        if self.alpha <= 0:
            raise ValueError(f"Initial volatility alpha must be positive, got {self.alpha}")
        
        if not (0 <= self.beta <= 1):
            raise ValueError(f"Beta must be in [0, 1], got {self.beta}")
        
        if self.nu < 0:
            raise ValueError(f"Vol-of-vol nu must be non-negative, got {self.nu}")
        
        if not (-1 <= self.rho <= 1):
            raise ValueError(f"Correlation rho must be in [-1, 1], got {self.rho}")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return {
            'F0': self.F0,
            'alpha': self.alpha,
            'beta': self.beta,
            'nu': self.nu,
            'rho': self.rho
        }
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, float]) -> 'SABRParams':
        """Create SABRParams from dictionary."""
        return cls(**params_dict)
    
    def __str__(self) -> str:
        """String representation of SABR parameters."""
        return (f"SABRParams(F0={self.F0:.4f}, alpha={self.alpha:.4f}, "
                f"beta={self.beta:.4f}, nu={self.nu:.4f}, rho={self.rho:.4f})")


@dataclass
class GridConfig:
    """
    Configuration for volatility surface discretization grid.
    
    Defines the strike and maturity ranges and discretization for generating
    volatility surfaces.
    
    Attributes:
        strike_range: (min_strike, max_strike) as multiples of forward price
        maturity_range: (min_maturity, max_maturity) in years
        n_strikes: Number of strike points
        n_maturities: Number of maturity points
        log_strikes: Whether to use log-spaced strikes (default: True)
        log_maturities: Whether to use log-spaced maturities (default: False)
    """
    strike_range: Tuple[float, float]
    maturity_range: Tuple[float, float]
    n_strikes: int
    n_maturities: int
    log_strikes: bool = True
    log_maturities: bool = False
    
    def __post_init__(self):
        """Validate grid configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate grid configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if len(self.strike_range) != 2:
            raise ValueError("strike_range must be a tuple of length 2")
        
        if len(self.maturity_range) != 2:
            raise ValueError("maturity_range must be a tuple of length 2")
        
        if self.strike_range[0] >= self.strike_range[1]:
            raise ValueError("strike_range[0] must be less than strike_range[1]")
        
        if self.maturity_range[0] >= self.maturity_range[1]:
            raise ValueError("maturity_range[0] must be less than maturity_range[1]")
        
        if self.strike_range[0] <= 0:
            raise ValueError("Minimum strike must be positive")
        
        if self.maturity_range[0] <= 0:
            raise ValueError("Minimum maturity must be positive")
        
        if self.n_strikes < 2:
            raise ValueError("n_strikes must be at least 2")
        
        if self.n_maturities < 2:
            raise ValueError("n_maturities must be at least 2")
    
    def get_strikes(self, forward_price: float) -> np.ndarray:
        """
        Generate strike array based on configuration.
        
        Args:
            forward_price: Forward price to scale strikes
            
        Returns:
            Array of strike values
        """
        min_strike = self.strike_range[0] * forward_price
        max_strike = self.strike_range[1] * forward_price
        
        if self.log_strikes:
            return np.logspace(np.log10(min_strike), np.log10(max_strike), self.n_strikes)
        else:
            return np.linspace(min_strike, max_strike, self.n_strikes)
    
    def get_maturities(self) -> np.ndarray:
        """
        Generate maturity array based on configuration.
        
        Returns:
            Array of maturity values in years
        """
        if self.log_maturities:
            return np.logspace(np.log10(self.maturity_range[0]), 
                             np.log10(self.maturity_range[1]), 
                             self.n_maturities)
        else:
            return np.linspace(self.maturity_range[0], self.maturity_range[1], self.n_maturities)
    
    def get_grid_shape(self) -> Tuple[int, int]:
        """Get the shape of the resulting surface grid (n_maturities, n_strikes)."""
        return (self.n_maturities, self.n_strikes)
    
    def __str__(self) -> str:
        """String representation of grid configuration."""
        return (f"GridConfig(strikes={self.strike_range}, maturities={self.maturity_range}, "
                f"grid_size=({self.n_maturities}, {self.n_strikes}))")


class ParameterSampler:
    """
    Parameter sampling strategies for SABR parameter space exploration.
    
    Supports multiple sampling strategies including uniform random sampling,
    Latin hypercube sampling, and adaptive sampling based on model performance.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize parameter sampler.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
    
    def uniform_sampling(self, 
                        n_samples: int,
                        F0_range: Tuple[float, float] = (80.0, 120.0),
                        alpha_range: Tuple[float, float] = (0.1, 0.8),
                        beta_range: Tuple[float, float] = (0.0, 1.0),
                        nu_range: Tuple[float, float] = (0.1, 1.0),
                        rho_range: Tuple[float, float] = (-0.9, 0.9)) -> List[SABRParams]:
        """
        Generate parameter samples using uniform random sampling.
        
        Args:
            n_samples: Number of parameter sets to generate
            F0_range: Range for forward price
            alpha_range: Range for initial volatility
            beta_range: Range for beta parameter
            nu_range: Range for vol-of-vol parameter
            rho_range: Range for correlation parameter
            
        Returns:
            List of SABRParams objects
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        samples = []
        
        for _ in range(n_samples):
            F0 = np.random.uniform(*F0_range)
            alpha = np.random.uniform(*alpha_range)
            beta = np.random.uniform(*beta_range)
            nu = np.random.uniform(*nu_range)
            rho = np.random.uniform(*rho_range)
            
            try:
                params = SABRParams(F0=F0, alpha=alpha, beta=beta, nu=nu, rho=rho)
                samples.append(params)
            except ValueError as e:
                warnings.warn(f"Invalid parameter combination generated: {e}")
                continue
        
        return samples
    
    def latin_hypercube_sampling(self,
                                n_samples: int,
                                F0_range: Tuple[float, float] = (80.0, 120.0),
                                alpha_range: Tuple[float, float] = (0.1, 0.8),
                                beta_range: Tuple[float, float] = (0.0, 1.0),
                                nu_range: Tuple[float, float] = (0.1, 1.0),
                                rho_range: Tuple[float, float] = (-0.9, 0.9)) -> List[SABRParams]:
        """
        Generate parameter samples using Latin Hypercube Sampling (LHS).
        
        LHS provides better space-filling properties than uniform random sampling.
        
        Args:
            n_samples: Number of parameter sets to generate
            F0_range: Range for forward price
            alpha_range: Range for initial volatility
            beta_range: Range for beta parameter
            nu_range: Range for vol-of-vol parameter
            rho_range: Range for correlation parameter
            
        Returns:
            List of SABRParams objects
        """
        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=5, seed=self.random_seed)
        
        # Generate samples in [0, 1]^5
        unit_samples = sampler.random(n=n_samples)
        
        # Scale to parameter ranges
        samples = []
        ranges = [F0_range, alpha_range, beta_range, nu_range, rho_range]
        
        for i in range(n_samples):
            scaled_params = []
            for j, (min_val, max_val) in enumerate(ranges):
                scaled_val = min_val + unit_samples[i, j] * (max_val - min_val)
                scaled_params.append(scaled_val)
            
            try:
                params = SABRParams(F0=scaled_params[0], alpha=scaled_params[1], 
                                  beta=scaled_params[2], nu=scaled_params[3], 
                                  rho=scaled_params[4])
                samples.append(params)
            except ValueError as e:
                warnings.warn(f"Invalid parameter combination generated: {e}")
                continue
        
        return samples
    
    def adaptive_sampling(self,
                         n_samples: int,
                         initial_samples: Optional[List[SABRParams]] = None,
                         performance_scores: Optional[List[float]] = None,
                         exploration_ratio: float = 0.3,
                         F0_range: Tuple[float, float] = (80.0, 120.0),
                         alpha_range: Tuple[float, float] = (0.1, 0.8),
                         beta_range: Tuple[float, float] = (0.0, 1.0),
                         nu_range: Tuple[float, float] = (0.1, 1.0),
                         rho_range: Tuple[float, float] = (-0.9, 0.9)) -> List[SABRParams]:
        """
        Generate parameter samples using adaptive sampling strategy.
        
        Adaptive sampling focuses on regions where the model performs poorly
        while maintaining some exploration of the parameter space.
        
        Args:
            n_samples: Number of parameter sets to generate
            initial_samples: Previous parameter samples (for exploitation)
            performance_scores: Performance scores for initial samples (lower is worse)
            exploration_ratio: Fraction of samples for exploration vs exploitation
            F0_range: Range for forward price
            alpha_range: Range for initial volatility
            beta_range: Range for beta parameter
            nu_range: Range for vol-of-vol parameter
            rho_range: Range for correlation parameter
            
        Returns:
            List of SABRParams objects
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        if initial_samples is None or performance_scores is None:
            # No previous data, fall back to Latin Hypercube sampling
            return self.latin_hypercube_sampling(n_samples, F0_range, alpha_range, 
                                               beta_range, nu_range, rho_range)
        
        if len(initial_samples) != len(performance_scores):
            raise ValueError("initial_samples and performance_scores must have same length")
        
        n_exploration = int(n_samples * exploration_ratio)
        n_exploitation = n_samples - n_exploration
        
        samples = []
        
        # Exploration: Random sampling in parameter space
        exploration_samples = self.uniform_sampling(n_exploration, F0_range, alpha_range,
                                                   beta_range, nu_range, rho_range)
        samples.extend(exploration_samples)
        
        # Exploitation: Sample around worst-performing regions
        if n_exploitation > 0:
            # Find worst performing samples (highest error scores)
            worst_indices = np.argsort(performance_scores)[-n_exploitation:]
            
            for idx in worst_indices:
                base_params = initial_samples[idx]
                
                # Add noise around the worst-performing parameter
                noise_scale = 0.1  # 10% noise
                
                F0_noise = np.random.normal(0, noise_scale * base_params.F0)
                alpha_noise = np.random.normal(0, noise_scale * base_params.alpha)
                beta_noise = np.random.normal(0, noise_scale * abs(base_params.beta - 0.5))
                nu_noise = np.random.normal(0, noise_scale * base_params.nu)
                rho_noise = np.random.normal(0, noise_scale * abs(base_params.rho))
                
                # Apply noise with bounds checking
                new_F0 = np.clip(base_params.F0 + F0_noise, *F0_range)
                new_alpha = np.clip(base_params.alpha + alpha_noise, *alpha_range)
                new_beta = np.clip(base_params.beta + beta_noise, *beta_range)
                new_nu = np.clip(base_params.nu + nu_noise, *nu_range)
                new_rho = np.clip(base_params.rho + rho_noise, *rho_range)
                
                try:
                    new_params = SABRParams(F0=new_F0, alpha=new_alpha, beta=new_beta,
                                          nu=new_nu, rho=new_rho)
                    samples.append(new_params)
                except ValueError as e:
                    warnings.warn(f"Invalid parameter combination generated: {e}")
                    # Fall back to uniform sampling for this sample
                    fallback = self.uniform_sampling(1, F0_range, alpha_range,
                                                   beta_range, nu_range, rho_range)
                    if fallback:
                        samples.append(fallback[0])
        
        return samples
    
    def sample_parameters(self,
                         n_samples: int,
                         strategy: str = "latin_hypercube",
                         **kwargs) -> List[SABRParams]:
        """
        Generate parameter samples using specified strategy.
        
        Args:
            n_samples: Number of parameter sets to generate
            strategy: Sampling strategy ("uniform", "latin_hypercube", "adaptive")
            **kwargs: Additional arguments for specific sampling strategies
            
        Returns:
            List of SABRParams objects
            
        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy == "uniform":
            return self.uniform_sampling(n_samples, **kwargs)
        elif strategy == "latin_hypercube":
            return self.latin_hypercube_sampling(n_samples, **kwargs)
        elif strategy == "adaptive":
            return self.adaptive_sampling(n_samples, **kwargs)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}. "
                           f"Available strategies: uniform, latin_hypercube, adaptive")


def create_default_grid_config() -> GridConfig:
    """
    Create a default grid configuration for SABR surface generation.
    
    Returns:
        GridConfig with reasonable default values
    """
    return GridConfig(
        strike_range=(0.5, 2.0),  # 50% to 200% of forward price
        maturity_range=(0.25, 5.0),  # 3 months to 5 years
        n_strikes=21,
        n_maturities=11,
        log_strikes=True,
        log_maturities=False
    )


def create_test_sabr_params() -> SABRParams:
    """
    Create test SABR parameters for unit testing and examples.
    
    Returns:
        SABRParams with typical market values
    """
    return SABRParams(
        F0=100.0,
        alpha=0.3,
        beta=0.7,
        nu=0.4,
        rho=-0.3
    )