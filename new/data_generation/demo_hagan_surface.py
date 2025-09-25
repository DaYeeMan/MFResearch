"""
Demonstration script for Hagan analytical SABR surface generator.

This script shows how to use the HaganSurfaceGenerator class to generate
SABR volatility surfaces using the Hagan et al. (2002) analytical approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any

from hagan_surface_generator import HaganSurfaceGenerator, HaganConfig, create_default_hagan_config
from sabr_params import SABRParams, GridConfig, create_default_grid_config, create_test_sabr_params


def demo_basic_surface_generation():
    """Demonstrate basic Hagan surface generation."""
    print("=== Basic Hagan Surface Generation ===")
    
    # Create generator and parameters
    generator = HaganSurfaceGenerator()
    params = create_test_sabr_params()
    grid = create_default_grid_config()
    
    print(f"SABR Parameters: {params}")
    print(f"Grid Configuration: {grid}")
    
    # Generate surface
    start_time = time.time()
    surface = generator.generate_surface(params, grid)
    elapsed_time = time.time() - start_time
    
    print(f"Generated surface shape: {surface.shape}")
    print(f"Generation time: {elapsed_time:.4f} seconds")
    
    # Display some statistics
    finite_surface = surface[np.isfinite(surface)]
    if len(finite_surface) > 0:
        print(f"Valid volatilities: {len(finite_surface)}/{surface.size}")
        print(f"Min volatility: {np.min(finite_surface):.4f}")
        print(f"Max volatility: {np.max(finite_surface):.4f}")
        print(f"Mean volatility: {np.mean(finite_surface):.4f}")
    
    print()
    return surface, params, grid


def demo_different_beta_values():
    """Demonstrate surface generation with different beta values."""
    print("=== Different Beta Values ===")
    
    generator = HaganSurfaceGenerator()
    grid = GridConfig(
        strike_range=(0.7, 1.3),
        maturity_range=(0.5, 2.0),
        n_strikes=11,
        n_maturities=5
    )
    
    beta_values = [0.0, 0.3, 0.7, 1.0]
    surfaces = {}
    
    for beta in beta_values:
        if beta == 1.0:
            # Normal model - use higher alpha
            params = SABRParams(F0=100.0, alpha=25.0, beta=beta, nu=0.4, rho=-0.3)
        else:
            params = SABRParams(F0=100.0, alpha=0.3, beta=beta, nu=0.4, rho=-0.3)
        
        surface = generator.generate_surface(params, grid)
        surfaces[beta] = surface
        
        finite_surface = surface[np.isfinite(surface)]
        print(f"Beta = {beta}: {len(finite_surface)} valid points, "
              f"mean vol = {np.mean(finite_surface):.4f}")
    
    print()
    return surfaces, grid


def demo_smile_analysis():
    """Demonstrate volatility smile analysis."""
    print("=== Volatility Smile Analysis ===")
    
    generator = HaganSurfaceGenerator()
    
    # Create grid focused on smile analysis
    smile_grid = GridConfig(
        strike_range=(0.6, 1.4),
        maturity_range=(1.0, 1.1),  # Near single maturity
        n_strikes=15,
        n_maturities=2
    )
    
    # Parameters that should produce a pronounced smile
    smile_params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.6, rho=-0.7)
    
    surface = generator.generate_surface(smile_params, smile_grid)
    strikes = smile_grid.get_strikes(smile_params.F0)
    vols = surface[0, :]  # Single maturity slice
    
    print(f"Smile parameters: {smile_params}")
    print(f"Strike range: {strikes[0]:.2f} to {strikes[-1]:.2f}")
    
    # Find ATM
    atm_idx = np.argmin(np.abs(strikes - smile_params.F0))
    atm_vol = vols[atm_idx]
    
    print(f"ATM volatility: {atm_vol:.4f}")
    
    # Calculate smile metrics
    finite_mask = np.isfinite(vols)
    if np.sum(finite_mask) > 0:
        finite_vols = vols[finite_mask]
        finite_strikes = strikes[finite_mask]
        
        min_vol = np.min(finite_vols)
        max_vol = np.max(finite_vols)
        vol_range = max_vol - min_vol
        
        print(f"Volatility range: {min_vol:.4f} to {max_vol:.4f} (spread: {vol_range:.4f})")
        
        # Print strike-vol pairs
        print("\nStrike-Volatility pairs:")
        for k, v in zip(finite_strikes, finite_vols):
            moneyness = k / smile_params.F0
            print(f"  K={k:6.2f} (M={moneyness:.3f}): σ={v:.4f}")
    
    print()
    return strikes, vols, smile_params


def demo_term_structure():
    """Demonstrate volatility term structure."""
    print("=== Volatility Term Structure ===")
    
    generator = HaganSurfaceGenerator()
    
    # Create grid focused on term structure
    term_grid = GridConfig(
        strike_range=(1.0, 1.1),  # Near ATM only
        n_strikes=2,
        maturity_range=(0.25, 5.0),
        n_maturities=10
    )
    
    params = create_test_sabr_params()
    surface = generator.generate_surface(params, term_grid)
    
    maturities = term_grid.get_maturities()
    atm_vols = surface[:, 0]  # Single strike column
    
    print(f"Parameters: {params}")
    print("\nATM Term Structure:")
    
    for T, vol in zip(maturities, atm_vols):
        if np.isfinite(vol):
            print(f"  T={T:5.2f}Y: σ={vol:.4f}")
    
    print()
    return maturities, atm_vols


def demo_benchmark_tests():
    """Demonstrate benchmark testing against literature."""
    print("=== Benchmark Tests ===")
    
    generator = HaganSurfaceGenerator()
    results = generator.benchmark_against_literature(tolerance=0.01)
    
    print(f"Benchmark Results:")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Max Error: {results['max_error']:.6f}")
    
    print("\nDetailed Results:")
    for test_case in results['test_cases']:
        status = "PASS" if test_case['passed'] else "FAIL"
        print(f"  {test_case['name']}: {status}")
        print(f"    Expected: {test_case['expected']:.6f}")
        print(f"    Calculated: {test_case['calculated']:.6f}")
        print(f"    Error: {test_case['error']:.6f}")
    
    print()
    return results


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("=== Configuration Options ===")
    
    params = create_test_sabr_params()
    grid = GridConfig(
        strike_range=(0.8, 1.2),
        maturity_range=(1.0, 2.0),
        n_strikes=5,
        n_maturities=3
    )
    
    # Default configuration
    default_generator = HaganSurfaceGenerator()
    default_surface = default_generator.generate_surface(params, grid)
    
    # Strict numerical tolerance
    strict_config = HaganConfig(numerical_tolerance=1e-15)
    strict_generator = HaganSurfaceGenerator(strict_config)
    strict_surface = strict_generator.generate_surface(params, grid)
    
    # Loose ATM tolerance
    loose_config = HaganConfig(atm_tolerance=1e-3)
    loose_generator = HaganSurfaceGenerator(loose_config)
    loose_surface = loose_generator.generate_surface(params, grid)
    
    # No validation
    no_val_config = HaganConfig(validate_output=False)
    no_val_generator = HaganSurfaceGenerator(no_val_config)
    no_val_surface = no_val_generator.generate_surface(params, grid)
    
    print("Configuration comparison:")
    print(f"  Default: {np.sum(np.isfinite(default_surface))} valid points")
    print(f"  Strict tolerance: {np.sum(np.isfinite(strict_surface))} valid points")
    print(f"  Loose ATM: {np.sum(np.isfinite(loose_surface))} valid points")
    print(f"  No validation: {np.sum(np.isfinite(no_val_surface))} valid points")
    
    print()


def visualize_surfaces():
    """Create visualizations of generated surfaces."""
    print("=== Surface Visualization ===")
    
    try:
        # Generate surfaces for different beta values
        surfaces, grid = demo_different_beta_values()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        strikes = grid.get_strikes(100.0)
        maturities = grid.get_maturities()
        
        beta_values = [0.0, 0.3, 0.7, 1.0]
        
        for i, beta in enumerate(beta_values):
            surface = surfaces[beta]
            
            # Create contour plot
            X, Y = np.meshgrid(strikes, maturities)
            
            # Mask invalid values
            masked_surface = np.ma.masked_invalid(surface)
            
            if not masked_surface.mask.all():
                contour = axes[i].contourf(X, Y, masked_surface, levels=20, cmap='viridis')
                axes[i].set_xlabel('Strike')
                axes[i].set_ylabel('Maturity')
                axes[i].set_title(f'SABR Surface (β={beta})')
                plt.colorbar(contour, ax=axes[i], label='Implied Volatility')
            else:
                axes[i].text(0.5, 0.5, 'No valid data', 
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f'SABR Surface (β={beta}) - No Data')
        
        plt.tight_layout()
        plt.savefig('hagan_sabr_surfaces.png', dpi=150, bbox_inches='tight')
        print("Saved surface plots as 'hagan_sabr_surfaces.png'")
        
        # Create smile plot
        strikes_smile, vols_smile, smile_params = demo_smile_analysis()
        
        plt.figure(figsize=(10, 6))
        finite_mask = np.isfinite(vols_smile)
        if np.any(finite_mask):
            plt.plot(strikes_smile[finite_mask], vols_smile[finite_mask], 'b-o', linewidth=2, markersize=6)
            plt.axvline(x=smile_params.F0, color='r', linestyle='--', alpha=0.7, label='ATM')
            plt.xlabel('Strike')
            plt.ylabel('Implied Volatility')
            plt.title('SABR Volatility Smile (Hagan Formula)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig('hagan_volatility_smile.png', dpi=150, bbox_inches='tight')
            print("Saved smile plot as 'hagan_volatility_smile.png'")
        
        # Create term structure plot
        maturities_term, vols_term = demo_term_structure()
        
        plt.figure(figsize=(10, 6))
        finite_mask = np.isfinite(vols_term)
        if np.any(finite_mask):
            plt.plot(maturities_term[finite_mask], vols_term[finite_mask], 'g-s', linewidth=2, markersize=6)
            plt.xlabel('Maturity (Years)')
            plt.ylabel('ATM Implied Volatility')
            plt.title('SABR ATM Volatility Term Structure (Hagan Formula)')
            plt.grid(True, alpha=0.3)
            
            plt.savefig('hagan_term_structure.png', dpi=150, bbox_inches='tight')
            print("Saved term structure plot as 'hagan_term_structure.png'")
        
        print()
        
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        print()


def performance_comparison():
    """Compare performance of different approaches."""
    print("=== Performance Comparison ===")
    
    # Create larger grid for performance testing
    perf_grid = GridConfig(
        strike_range=(0.5, 2.0),
        maturity_range=(0.25, 5.0),
        n_strikes=25,
        n_maturities=15
    )
    
    params = create_test_sabr_params()
    generator = HaganSurfaceGenerator()
    
    # Time surface generation
    n_runs = 5
    times = []
    
    for i in range(n_runs):
        start_time = time.time()
        surface = generator.generate_surface(params, perf_grid)
        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Performance Results ({n_runs} runs):")
    print(f"  Grid size: {perf_grid.get_grid_shape()} = {np.prod(perf_grid.get_grid_shape())} points")
    print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Points per second: {np.prod(perf_grid.get_grid_shape()) / avg_time:.0f}")
    
    # Check accuracy
    finite_count = np.sum(np.isfinite(surface))
    total_count = surface.size
    print(f"  Valid points: {finite_count}/{total_count} ({100*finite_count/total_count:.1f}%)")
    
    print()


if __name__ == "__main__":
    """Run all demonstrations."""
    print("Hagan Analytical SABR Surface Generator Demo")
    print("=" * 50)
    print()
    
    # Run demonstrations
    demo_basic_surface_generation()
    demo_different_beta_values()
    demo_smile_analysis()
    demo_term_structure()
    demo_benchmark_tests()
    demo_configuration_options()
    performance_comparison()
    
    # Create visualizations if matplotlib is available
    try:
        visualize_surfaces()
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
    
    print("Demo completed successfully!")