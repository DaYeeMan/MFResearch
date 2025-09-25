"""
Demonstration script for SABR parameter and grid configuration classes.

This script shows how to use the SABRParams, GridConfig, and ParameterSampler
classes for SABR volatility surface modeling.
"""

import numpy as np
import matplotlib.pyplot as plt
from sabr_params import (
    SABRParams, GridConfig, ParameterSampler,
    create_default_grid_config, create_test_sabr_params
)


def demo_sabr_params():
    """Demonstrate SABRParams class functionality."""
    print("=== SABR Parameters Demo ===")
    
    # Create valid SABR parameters
    params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
    print(f"Valid SABR params: {params}")
    
    # Convert to/from dictionary
    params_dict = params.to_dict()
    print(f"As dictionary: {params_dict}")
    
    params_from_dict = SABRParams.from_dict(params_dict)
    print(f"From dictionary: {params_from_dict}")
    
    # Try invalid parameters
    try:
        invalid_params = SABRParams(F0=-10.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
    except ValueError as e:
        print(f"Validation caught invalid F0: {e}")
    
    print()


def demo_grid_config():
    """Demonstrate GridConfig class functionality."""
    print("=== Grid Configuration Demo ===")
    
    # Create grid configuration
    config = GridConfig(
        strike_range=(0.5, 2.0),
        maturity_range=(0.25, 5.0),
        n_strikes=11,
        n_maturities=6,
        log_strikes=True,
        log_maturities=False
    )
    print(f"Grid config: {config}")
    
    # Generate strikes and maturities
    forward_price = 100.0
    strikes = config.get_strikes(forward_price)
    maturities = config.get_maturities()
    
    print(f"Strikes: {strikes}")
    print(f"Maturities: {maturities}")
    print(f"Grid shape: {config.get_grid_shape()}")
    
    print()


def demo_parameter_sampling():
    """Demonstrate ParameterSampler class functionality."""
    print("=== Parameter Sampling Demo ===")
    
    sampler = ParameterSampler(random_seed=42)
    
    # Uniform sampling
    print("Uniform sampling:")
    uniform_samples = sampler.uniform_sampling(n_samples=3)
    for i, sample in enumerate(uniform_samples):
        print(f"  Sample {i+1}: {sample}")
    
    print()
    
    # Latin Hypercube sampling
    print("Latin Hypercube sampling:")
    lhs_samples = sampler.latin_hypercube_sampling(n_samples=3)
    for i, sample in enumerate(lhs_samples):
        print(f"  Sample {i+1}: {sample}")
    
    print()
    
    # Adaptive sampling (without initial data - falls back to LHS)
    print("Adaptive sampling (no initial data):")
    adaptive_samples = sampler.adaptive_sampling(n_samples=3)
    for i, sample in enumerate(adaptive_samples):
        print(f"  Sample {i+1}: {sample}")
    
    print()
    
    # Adaptive sampling with initial data
    print("Adaptive sampling (with initial data):")
    initial_samples = uniform_samples
    performance_scores = [0.1, 0.8, 0.3]  # Higher = worse performance
    
    adaptive_samples_with_data = sampler.adaptive_sampling(
        n_samples=5,
        initial_samples=initial_samples,
        performance_scores=performance_scores,
        exploration_ratio=0.4
    )
    for i, sample in enumerate(adaptive_samples_with_data):
        print(f"  Sample {i+1}: {sample}")
    
    print()


def demo_utility_functions():
    """Demonstrate utility functions."""
    print("=== Utility Functions Demo ===")
    
    # Default configurations
    default_grid = create_default_grid_config()
    print(f"Default grid config: {default_grid}")
    
    test_params = create_test_sabr_params()
    print(f"Test SABR params: {test_params}")
    
    print()


def visualize_parameter_sampling():
    """Visualize different sampling strategies."""
    print("=== Parameter Sampling Visualization ===")
    
    sampler = ParameterSampler(random_seed=42)
    n_samples = 50
    
    # Generate samples using different strategies
    uniform_samples = sampler.uniform_sampling(n_samples=n_samples)
    lhs_samples = sampler.latin_hypercube_sampling(n_samples=n_samples)
    
    # Extract F0 and alpha for visualization
    uniform_F0 = [s.F0 for s in uniform_samples]
    uniform_alpha = [s.alpha for s in uniform_samples]
    
    lhs_F0 = [s.F0 for s in lhs_samples]
    lhs_alpha = [s.alpha for s in lhs_samples]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Uniform sampling
    ax1.scatter(uniform_F0, uniform_alpha, alpha=0.7, s=50)
    ax1.set_xlabel('F0 (Forward Price)')
    ax1.set_ylabel('Alpha (Initial Volatility)')
    ax1.set_title('Uniform Random Sampling')
    ax1.grid(True, alpha=0.3)
    
    # Latin Hypercube sampling
    ax2.scatter(lhs_F0, lhs_alpha, alpha=0.7, s=50, color='red')
    ax2.set_xlabel('F0 (Forward Price)')
    ax2.set_ylabel('Alpha (Initial Volatility)')
    ax2.set_title('Latin Hypercube Sampling')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_sampling_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved parameter sampling comparison plot as 'parameter_sampling_comparison.png'")
    
    print()


if __name__ == "__main__":
    """Run all demonstrations."""
    print("SABR Parameter and Grid Configuration Demo")
    print("=" * 50)
    print()
    
    demo_sabr_params()
    demo_grid_config()
    demo_parameter_sampling()
    demo_utility_functions()
    
    # Only create visualization if matplotlib is available
    try:
        visualize_parameter_sampling()
    except ImportError:
        print("Matplotlib not available, skipping visualization demo")
    
    print("Demo completed successfully!")