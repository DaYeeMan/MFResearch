"""
Example script demonstrating 3D surface visualization capabilities.

This script shows how to use the SurfacePlotter class to create various
types of 3D volatility surface visualizations including:
- Basic 3D surface plots
- Error heatmaps
- Surface difference plots
- Multiple surface comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from visualization.surface_plotter import (
    SurfacePlotter,
    SurfaceData,
    SurfacePlotConfig,
    create_surface_data
)
from data_generation.sabr_params import SABRParams, GridConfig


def create_sample_sabr_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    sabr_params: SABRParams,
    forward_price: float,
    model_name: str,
    add_noise: bool = False,
    noise_level: float = 0.01
) -> SurfaceData:
    """
    Create a sample SABR volatility surface for demonstration.
    
    Args:
        strikes: Strike prices
        maturities: Time to maturities
        sabr_params: SABR parameters
        forward_price: Forward price
        model_name: Name for the model
        add_noise: Whether to add random noise
        noise_level: Standard deviation of noise
        
    Returns:
        SurfaceData object
    """
    # Create meshgrid
    K_mesh, T_mesh = np.meshgrid(strikes, maturities)
    moneyness = K_mesh / forward_price
    
    # Simple SABR-like volatility surface model
    # This is a simplified approximation for demonstration
    alpha = sabr_params.alpha
    beta = sabr_params.beta
    nu = sabr_params.nu
    rho = sabr_params.rho
    
    # Base volatility with smile effect
    base_vol = alpha * (moneyness ** (beta - 1))
    
    # Add smile curvature
    smile_effect = 0.1 * (moneyness - 1.0) ** 2
    
    # Add term structure effect
    term_effect = 0.05 * np.sqrt(T_mesh)
    
    # Combine effects
    volatilities = base_vol + smile_effect + term_effect
    
    # Add vol-of-vol effect (simplified)
    vol_of_vol_effect = nu * 0.1 * np.sqrt(T_mesh) * np.abs(moneyness - 1.0)
    volatilities += vol_of_vol_effect
    
    # Add correlation effect (simplified)
    corr_effect = rho * 0.05 * (moneyness - 1.0) * np.sqrt(T_mesh)
    volatilities += corr_effect
    
    # Ensure positive volatilities
    volatilities = np.maximum(volatilities, 0.05)
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, noise_level, volatilities.shape)
        volatilities += noise
        volatilities = np.maximum(volatilities, 0.01)  # Ensure still positive
    
    return create_surface_data(
        strikes=strikes,
        maturities=maturities,
        volatilities=volatilities,
        model_name=model_name,
        sabr_params=sabr_params,
        forward_price=forward_price
    )


def demonstrate_basic_3d_plotting():
    """Demonstrate basic 3D surface plotting."""
    print("=== Basic 3D Surface Plotting ===")
    
    # Set up parameters
    strikes = np.linspace(80, 120, 15)
    maturities = np.linspace(0.25, 3.0, 12)
    forward_price = 100.0
    
    sabr_params = SABRParams(
        F0=forward_price,
        alpha=0.25,
        beta=0.7,
        nu=0.4,
        rho=-0.3
    )
    
    # Create sample surface
    hf_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "High-Fidelity Monte Carlo"
    )
    
    # Initialize plotter
    config = SurfacePlotConfig(
        figure_size=(14, 10),
        colormap='viridis',
        surface_count=60
    )
    plotter = SurfacePlotter(config=config, output_dir="surface_plots")
    
    # Create static 3D surface plot
    print("Creating static 3D surface plot...")
    fig = plotter.plot_3d_surface(
        hf_surface,
        title="SABR Volatility Surface - Monte Carlo Simulation",
        save_path="hf_surface_3d.png",
        show_wireframe=False,
        elevation=30,
        azimuth=45
    )
    
    # Create interactive 3D surface plot
    print("Creating interactive 3D surface plot...")
    fig_interactive = plotter.plot_interactive_3d_surface(
        hf_surface,
        title="Interactive SABR Volatility Surface",
        save_path="hf_surface_interactive.html",
        show_contours=True
    )
    
    print("Basic 3D plots saved to surface_plots/ directory")
    plt.close('all')


def demonstrate_error_heatmaps():
    """Demonstrate error heatmap visualization."""
    print("\n=== Error Heatmap Visualization ===")
    
    # Set up parameters
    strikes = np.linspace(85, 115, 10)
    maturities = np.linspace(0.5, 2.0, 8)
    forward_price = 100.0
    
    sabr_params = SABRParams(
        F0=forward_price,
        alpha=0.22,
        beta=0.6,
        nu=0.35,
        rho=-0.25
    )
    
    # Create reference (true) surface
    true_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "True Surface (HF MC)"
    )
    
    # Create predicted surface with some error
    predicted_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "MDA-CNN Prediction",
        add_noise=True,
        noise_level=0.015
    )
    
    # Initialize plotter
    plotter = SurfacePlotter(output_dir="surface_plots")
    
    # Create different types of error heatmaps
    print("Creating absolute error heatmap...")
    fig1 = plotter.plot_error_heatmap(
        predicted_surface,
        true_surface,
        title="Absolute Prediction Errors",
        error_type='absolute',
        save_path="error_heatmap_absolute.png"
    )
    
    print("Creating relative error heatmap...")
    fig2 = plotter.plot_error_heatmap(
        predicted_surface,
        true_surface,
        title="Relative Prediction Errors",
        error_type='relative',
        save_path="error_heatmap_relative.png"
    )
    
    print("Creating percentage error heatmap...")
    fig3 = plotter.plot_error_heatmap(
        predicted_surface,
        true_surface,
        title="Percentage Prediction Errors",
        error_type='percentage',
        save_path="error_heatmap_percentage.png",
        moneyness_axis=False
    )
    
    print("Error heatmaps saved to surface_plots/ directory")
    plt.close('all')


def demonstrate_surface_differences():
    """Demonstrate surface difference plotting."""
    print("\n=== Surface Difference Visualization ===")
    
    # Set up parameters
    strikes = np.linspace(80, 120, 12)
    maturities = np.linspace(0.25, 2.5, 10)
    forward_price = 100.0
    
    sabr_params = SABRParams(
        F0=forward_price,
        alpha=0.28,
        beta=0.5,
        nu=0.45,
        rho=-0.4
    )
    
    # Create Monte Carlo surface
    mc_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "Monte Carlo"
    )
    
    # Create Hagan analytical surface (with systematic bias)
    hagan_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "Hagan Analytical"
    )
    # Add systematic bias to simulate Hagan approximation error
    hagan_surface.volatilities *= 0.98  # Slight underestimation
    
    # Initialize plotter
    plotter = SurfacePlotter(output_dir="surface_plots")
    
    # Create static surface difference plot
    print("Creating static surface difference plot...")
    fig1 = plotter.plot_surface_difference(
        mc_surface,
        hagan_surface,
        title="MC vs Hagan Surface Difference",
        save_path="surface_difference_static.png",
        interactive=False
    )
    
    # Create interactive surface difference plot
    print("Creating interactive surface difference plot...")
    fig2 = plotter.plot_surface_difference(
        mc_surface,
        hagan_surface,
        title="Interactive MC vs Hagan Difference",
        save_path="surface_difference_interactive.html",
        interactive=True
    )
    
    print("Surface difference plots saved to surface_plots/ directory")
    plt.close('all')


def demonstrate_multiple_surface_comparison():
    """Demonstrate multiple surface comparison."""
    print("\n=== Multiple Surface Comparison ===")
    
    # Set up parameters
    strikes = np.linspace(85, 115, 8)
    maturities = np.linspace(0.5, 2.0, 6)
    forward_price = 100.0
    
    sabr_params = SABRParams(
        F0=forward_price,
        alpha=0.24,
        beta=0.8,
        nu=0.3,
        rho=-0.2
    )
    
    # Create multiple surfaces for comparison
    surfaces = []
    
    # High-fidelity Monte Carlo
    hf_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "HF Monte Carlo"
    )
    surfaces.append(hf_surface)
    
    # Low-fidelity Hagan
    lf_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "LF Hagan"
    )
    lf_surface.volatilities *= 0.97  # Systematic underestimation
    surfaces.append(lf_surface)
    
    # Baseline MLP
    baseline_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "Baseline MLP",
        add_noise=True,
        noise_level=0.02
    )
    surfaces.append(baseline_surface)
    
    # MDA-CNN prediction
    mdacnn_surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "MDA-CNN",
        add_noise=True,
        noise_level=0.008
    )
    surfaces.append(mdacnn_surface)
    
    # Initialize plotter
    plotter = SurfacePlotter(output_dir="surface_plots")
    
    # Create overlay comparison (static)
    print("Creating overlay surface comparison...")
    fig1 = plotter.plot_multiple_surfaces_comparison(
        surfaces,
        title="Multiple Model Surface Comparison - Overlay",
        layout='overlay',
        save_path="multiple_surfaces_overlay.png",
        interactive=False
    )
    
    # Create grid comparison (static)
    print("Creating grid surface comparison...")
    fig2 = plotter.plot_multiple_surfaces_comparison(
        surfaces,
        title="Multiple Model Surface Comparison - Grid",
        layout='grid',
        save_path="multiple_surfaces_grid.png",
        interactive=False
    )
    
    # Create interactive overlay comparison
    print("Creating interactive overlay comparison...")
    fig3 = plotter.plot_multiple_surfaces_comparison(
        surfaces,
        title="Interactive Multiple Surface Comparison",
        layout='overlay',
        save_path="multiple_surfaces_interactive.html",
        interactive=True
    )
    
    print("Multiple surface comparison plots saved to surface_plots/ directory")
    plt.close('all')


def demonstrate_advanced_features():
    """Demonstrate advanced visualization features."""
    print("\n=== Advanced Visualization Features ===")
    
    # Set up parameters for high-resolution surface
    strikes = np.linspace(70, 130, 20)
    maturities = np.linspace(0.1, 5.0, 15)
    forward_price = 100.0
    
    # Create surface with more complex SABR parameters
    sabr_params = SABRParams(
        F0=forward_price,
        alpha=0.3,
        beta=0.4,
        nu=0.6,
        rho=-0.5
    )
    
    surface = create_sample_sabr_surface(
        strikes, maturities, sabr_params, forward_price,
        "High-Resolution SABR Surface"
    )
    
    # Initialize plotter with custom configuration
    config = SurfacePlotConfig(
        figure_size=(16, 12),
        colormap='plasma',
        error_colormap='seismic',
        surface_count=80,
        alpha=0.9
    )
    plotter = SurfacePlotter(config=config, output_dir="surface_plots")
    
    # Create high-resolution surface with wireframe
    print("Creating high-resolution surface with wireframe...")
    fig1 = plotter.plot_3d_surface(
        surface,
        title="High-Resolution SABR Surface with Wireframe",
        save_path="high_res_surface_wireframe.png",
        show_wireframe=True,
        elevation=20,
        azimuth=60
    )
    
    # Create surface using strike axis instead of moneyness
    print("Creating surface with strike axis...")
    fig2 = plotter.plot_3d_surface(
        surface,
        title="SABR Surface - Strike Axis",
        save_path="surface_strike_axis.png",
        moneyness_axis=False,
        elevation=45,
        azimuth=135
    )
    
    # Create interactive surface with custom camera angle
    print("Creating interactive surface with custom view...")
    fig3 = plotter.plot_interactive_3d_surface(
        surface,
        title="Interactive High-Resolution SABR Surface",
        save_path="high_res_interactive.html",
        show_contours=False
    )
    
    print("Advanced visualization features demonstrated")
    plt.close('all')


def main():
    """Run all demonstration examples."""
    print("SABR 3D Surface Visualization Demonstration")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Run demonstrations
        demonstrate_basic_3d_plotting()
        demonstrate_error_heatmaps()
        demonstrate_surface_differences()
        demonstrate_multiple_surface_comparison()
        demonstrate_advanced_features()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("Check the 'surface_plots/' directory for generated visualizations.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()