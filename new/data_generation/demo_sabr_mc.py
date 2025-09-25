"""
Demo script for SABR Monte Carlo simulation engine.

This script demonstrates the usage of the Monte Carlo SABR volatility surface
generator with various parameter configurations and shows basic functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sabr_params import SABRParams, GridConfig, create_test_sabr_params, create_default_grid_config

# Import with absolute imports by temporarily modifying the module
import importlib.util
spec = importlib.util.spec_from_file_location("sabr_mc_generator", "sabr_mc_generator.py")
sabr_mc_module = importlib.util.module_from_spec(spec)

# Fix the relative imports in the module
import sabr_params
sabr_mc_module.sabr_params = sabr_params
sys.modules['sabr_mc_generator.sabr_params'] = sabr_params

# Import logging utils
sys.path.append('../utils')
import logging_utils
import reproducibility
sabr_mc_module.logging_utils = logging_utils
sabr_mc_module.reproducibility = reproducibility

spec.loader.exec_module(sabr_mc_module)

SABRMCGenerator = sabr_mc_module.SABRMCGenerator
ParallelSABRMCGenerator = sabr_mc_module.ParallelSABRMCGenerator
MCConfig = sabr_mc_module.MCConfig
create_default_mc_config = sabr_mc_module.create_default_mc_config


def demo_basic_surface_generation():
    """Demonstrate basic surface generation."""
    print("=== Basic SABR MC Surface Generation Demo ===")
    
    # Create configuration
    mc_config = MCConfig(
        n_paths=10000,
        n_steps=100,
        convergence_check=False,
        random_seed=42
    )
    
    # Create generator
    generator = SABRMCGenerator(mc_config)
    
    # Create SABR parameters
    sabr_params = create_test_sabr_params()
    print(f"SABR Parameters: {sabr_params}")
    
    # Create grid configuration
    grid_config = GridConfig(
        strike_range=(0.8, 1.2),
        maturity_range=(0.25, 2.0),
        n_strikes=7,
        n_maturities=5
    )
    print(f"Grid Configuration: {grid_config}")
    
    # Generate surface
    start_time = time.time()
    surface = generator.generate_surface(sabr_params, grid_config)
    elapsed_time = time.time() - start_time
    
    print(f"Surface generated in {elapsed_time:.2f} seconds")
    print(f"Surface shape: {surface.shape}")
    print(f"Surface statistics:")
    print(f"  Min volatility: {np.min(surface):.4f}")
    print(f"  Max volatility: {np.max(surface):.4f}")
    print(f"  Mean volatility: {np.mean(surface):.4f}")
    print(f"  Std volatility: {np.std(surface):.4f}")
    
    return surface, sabr_params, grid_config


def demo_convergence_monitoring():
    """Demonstrate convergence monitoring."""
    print("\n=== Convergence Monitoring Demo ===")
    
    # Create configuration with convergence monitoring
    mc_config = MCConfig(
        n_paths=5000,  # Start with fewer paths
        n_steps=50,
        convergence_check=True,
        convergence_tolerance=1e-3,
        max_iterations=4,
        random_seed=42
    )
    
    generator = SABRMCGenerator(mc_config)
    sabr_params = create_test_sabr_params()
    
    # Simple grid for faster convergence
    grid_config = GridConfig(
        strike_range=(0.9, 1.1),
        maturity_range=(0.5, 1.0),
        n_strikes=3,
        n_maturities=2
    )
    
    # Generate surface with convergence monitoring
    start_time = time.time()
    surface, conv_info = generator.generate_surface(sabr_params, grid_config)
    elapsed_time = time.time() - start_time
    
    print(f"Surface generated in {elapsed_time:.2f} seconds")
    print(f"Convergence info:")
    print(f"  Converged: {conv_info['converged']}")
    print(f"  Iterations: {conv_info['iterations']}")
    print(f"  Final difference: {conv_info['final_difference']:.6f}")
    print(f"  Path counts: {conv_info['path_counts']}")


def demo_parallel_generation():
    """Demonstrate parallel surface generation."""
    print("\n=== Parallel Surface Generation Demo ===")
    
    # Create multiple parameter sets
    param_sets = [
        SABRParams(F0=100.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.2),
        SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.4),
        SABRParams(F0=100.0, alpha=0.25, beta=0.6, nu=0.35, rho=-0.3),
        SABRParams(F0=100.0, alpha=0.35, beta=0.8, nu=0.45, rho=-0.1),
    ]
    
    print(f"Generating surfaces for {len(param_sets)} parameter sets")
    
    # Configuration for parallel processing
    mc_config = MCConfig(
        n_paths=5000,
        n_steps=50,
        parallel=True,
        n_workers=2,
        convergence_check=False,
        random_seed=42
    )
    
    parallel_generator = ParallelSABRMCGenerator(mc_config)
    
    grid_config = GridConfig(
        strike_range=(0.8, 1.2),
        maturity_range=(0.5, 1.5),
        n_strikes=5,
        n_maturities=3
    )
    
    # Progress callback
    def progress_callback(completed, total):
        print(f"  Progress: {completed}/{total} surfaces completed")
    
    # Generate surfaces
    start_time = time.time()
    surfaces = parallel_generator.generate_surfaces(
        param_sets, grid_config, progress_callback
    )
    elapsed_time = time.time() - start_time
    
    print(f"All surfaces generated in {elapsed_time:.2f} seconds")
    print(f"Generated {len(surfaces)} surfaces")
    
    # Show statistics for each surface
    for i, surface in enumerate(surfaces):
        print(f"  Surface {i+1}: shape={surface.shape}, "
              f"mean_vol={np.mean(surface):.4f}, "
              f"std_vol={np.std(surface):.4f}")


def demo_different_beta_values():
    """Demonstrate surfaces with different beta values."""
    print("\n=== Different Beta Values Demo ===")
    
    mc_config = MCConfig(
        n_paths=8000,
        n_steps=75,
        convergence_check=False,
        random_seed=42
    )
    
    generator = SABRMCGenerator(mc_config)
    
    # Test different beta values
    beta_values = [0.0, 0.5, 1.0]
    base_params = SABRParams(F0=100.0, alpha=0.3, beta=0.5, nu=0.4, rho=-0.3)
    
    grid_config = GridConfig(
        strike_range=(0.8, 1.2),
        maturity_range=(0.5, 1.0),
        n_strikes=5,
        n_maturities=3
    )
    
    surfaces = {}
    
    for beta in beta_values:
        print(f"Generating surface for beta = {beta}")
        params = SABRParams(
            F0=base_params.F0,
            alpha=base_params.alpha,
            beta=beta,
            nu=base_params.nu,
            rho=base_params.rho
        )
        
        start_time = time.time()
        surface = generator.generate_surface(params, grid_config)
        elapsed_time = time.time() - start_time
        
        surfaces[beta] = surface
        
        print(f"  Beta {beta}: generated in {elapsed_time:.2f}s, "
              f"mean_vol={np.mean(surface):.4f}")
    
    return surfaces


def plot_volatility_smile(surface, sabr_params, grid_config, maturity_idx=0):
    """Plot volatility smile for a given maturity."""
    try:
        strikes = grid_config.get_strikes(sabr_params.F0)
        maturities = grid_config.get_maturities()
        
        plt.figure(figsize=(10, 6))
        
        # Plot smile for the specified maturity
        maturity = maturities[maturity_idx]
        vols = surface[maturity_idx, :]
        
        plt.plot(strikes, vols, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.title(f'SABR MC Volatility Smile (T = {maturity:.2f} years)')
        plt.grid(True, alpha=0.3)
        
        # Add parameter info
        param_text = (f'F0={sabr_params.F0}, α={sabr_params.alpha:.2f}, '
                     f'β={sabr_params.beta:.2f}, ν={sabr_params.nu:.2f}, '
                     f'ρ={sabr_params.rho:.2f}')
        plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")


def main():
    """Run all demos."""
    print("SABR Monte Carlo Simulation Engine Demo")
    print("=" * 50)
    
    # Basic surface generation
    surface, sabr_params, grid_config = demo_basic_surface_generation()
    
    # Convergence monitoring
    demo_convergence_monitoring()
    
    # Parallel generation
    demo_parallel_generation()
    
    # Different beta values
    beta_surfaces = demo_different_beta_values()
    
    # Try to plot if matplotlib is available
    try:
        plot_volatility_smile(surface, sabr_params, grid_config)
    except Exception as e:
        print(f"\nSkipping plot due to: {e}")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()