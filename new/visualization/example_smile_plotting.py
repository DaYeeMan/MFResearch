"""
Example usage of volatility smile visualization tools.

This script demonstrates how to use the SmilePlotter class to create
comprehensive volatility smile visualizations comparing different models.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from visualization.smile_plotter import (
    SmilePlotter, SmileData, SmilePlotConfig, create_smile_data
)
from data_generation.sabr_params import SABRParams, GridConfig
from data_generation.hagan_surface_generator import HaganSurfaceGenerator
from data_generation.sabr_mc_generator import SABRMCGenerator


def generate_sample_data():
    """Generate sample volatility smile data for demonstration."""
    
    # Define SABR parameters
    sabr_params = SABRParams(
        F0=100.0,
        alpha=0.3,
        beta=0.7,
        nu=0.4,
        rho=-0.3
    )
    
    # Define grid configuration
    grid_config = GridConfig(
        strike_range=(0.7, 1.3),  # 70% to 130% of forward
        maturity_range=(1.0, 1.0),  # Single maturity
        n_strikes=15,
        n_maturities=1
    )
    
    # Generate strikes and maturity
    strikes = grid_config.get_strikes(sabr_params.F0)
    maturity = 1.0
    
    print(f"Generating sample data for {len(strikes)} strikes...")
    print(f"Strike range: {strikes[0]:.1f} to {strikes[-1]:.1f}")
    
    # Generate high-fidelity Monte Carlo surface
    print("Generating HF Monte Carlo surface...")
    mc_generator = SABRMCGenerator()
    hf_surface = mc_generator.generate_surface(sabr_params, grid_config, n_paths=50000)
    hf_volatilities = hf_surface[0, :]  # Single maturity
    
    # Generate low-fidelity Hagan surface
    print("Generating LF Hagan surface...")
    hagan_generator = HaganSurfaceGenerator()
    lf_surface = hagan_generator.generate_surface(sabr_params, grid_config)
    lf_volatilities = lf_surface[0, :]  # Single maturity
    
    # Simulate baseline model predictions (MLP with some error)
    print("Simulating baseline model predictions...")
    baseline_error = np.random.normal(0, 0.01, len(strikes))
    baseline_volatilities = hf_volatilities + baseline_error
    
    # Simulate MDA-CNN predictions (better than baseline)
    print("Simulating MDA-CNN predictions...")
    mda_cnn_error = np.random.normal(0, 0.005, len(strikes))
    mda_cnn_volatilities = hf_volatilities + mda_cnn_error
    
    # Create SmileData objects
    smile_data_list = [
        create_smile_data(
            strikes=strikes,
            volatilities=hf_volatilities,
            model_name="HF Monte Carlo",
            maturity=maturity,
            sabr_params=sabr_params,
            forward_price=sabr_params.F0
        ),
        create_smile_data(
            strikes=strikes,
            volatilities=lf_volatilities,
            model_name="LF Hagan",
            maturity=maturity,
            sabr_params=sabr_params,
            forward_price=sabr_params.F0
        ),
        create_smile_data(
            strikes=strikes,
            volatilities=baseline_volatilities,
            model_name="Baseline MLP",
            maturity=maturity,
            sabr_params=sabr_params,
            forward_price=sabr_params.F0
        ),
        create_smile_data(
            strikes=strikes,
            volatilities=mda_cnn_volatilities,
            model_name="MDA-CNN",
            maturity=maturity,
            sabr_params=sabr_params,
            forward_price=sabr_params.F0
        )
    ]
    
    return smile_data_list


def generate_multi_maturity_data():
    """Generate sample data for multiple maturities."""
    
    sabr_params = SABRParams(
        F0=100.0,
        alpha=0.3,
        beta=0.7,
        nu=0.4,
        rho=-0.3
    )
    
    maturities = [0.25, 0.5, 1.0, 2.0]
    smile_data_dict = {}
    
    print(f"Generating multi-maturity data for {len(maturities)} maturities...")
    
    for maturity in maturities:
        print(f"  Processing maturity T = {maturity:.2f}y...")
        
        # Define grid for this maturity
        grid_config = GridConfig(
            strike_range=(0.8, 1.2),
            maturity_range=(maturity, maturity),
            n_strikes=11,
            n_maturities=1
        )
        
        strikes = grid_config.get_strikes(sabr_params.F0)
        
        # Generate surfaces
        mc_generator = SABRMCGenerator()
        hagan_generator = HaganSurfaceGenerator()
        
        hf_surface = mc_generator.generate_surface(sabr_params, grid_config, n_paths=20000)
        lf_surface = hagan_generator.generate_surface(sabr_params, grid_config)
        
        hf_volatilities = hf_surface[0, :]
        lf_volatilities = lf_surface[0, :]
        
        # Simulate model predictions
        mda_cnn_error = np.random.normal(0, 0.003, len(strikes))
        mda_cnn_volatilities = hf_volatilities + mda_cnn_error
        
        smile_data_dict[maturity] = [
            create_smile_data(
                strikes=strikes,
                volatilities=hf_volatilities,
                model_name="HF Monte Carlo",
                maturity=maturity,
                sabr_params=sabr_params,
                forward_price=sabr_params.F0
            ),
            create_smile_data(
                strikes=strikes,
                volatilities=lf_volatilities,
                model_name="LF Hagan",
                maturity=maturity,
                sabr_params=sabr_params,
                forward_price=sabr_params.F0
            ),
            create_smile_data(
                strikes=strikes,
                volatilities=mda_cnn_volatilities,
                model_name="MDA-CNN",
                maturity=maturity,
                sabr_params=sabr_params,
                forward_price=sabr_params.F0
            )
        ]
    
    return smile_data_dict


def generate_confidence_interval_data():
    """Generate sample data with confidence intervals."""
    
    sabr_params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
    
    # Define strikes
    strikes = np.linspace(80, 120, 9)
    maturity = 1.0
    
    # Generate base volatilities (true values)
    grid_config = GridConfig(
        strike_range=(0.8, 1.2),
        maturity_range=(1.0, 1.0),
        n_strikes=len(strikes),
        n_maturities=1
    )
    
    hagan_generator = HaganSurfaceGenerator()
    base_surface = hagan_generator.generate_surface(sabr_params, grid_config)
    base_volatilities = base_surface[0, :]
    
    # Simulate multiple model predictions to create confidence intervals
    n_predictions = 100
    predictions = np.random.normal(
        base_volatilities.reshape(1, -1),
        0.008,  # Standard deviation
        (n_predictions, len(strikes))
    )
    
    # Compute confidence intervals
    mean_predictions = np.mean(predictions, axis=0)
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)
    
    confidence_intervals = {
        'lower': ci_lower,
        'upper': ci_upper
    }
    
    # Create smile data with confidence intervals
    smile_data_with_ci = create_smile_data(
        strikes=strikes,
        volatilities=mean_predictions,
        model_name="Model with CI",
        maturity=maturity,
        sabr_params=sabr_params,
        forward_price=sabr_params.F0,
        confidence_intervals=confidence_intervals
    )
    
    # Also create reference data without CI
    smile_data_reference = create_smile_data(
        strikes=strikes,
        volatilities=base_volatilities,
        model_name="Reference Model",
        maturity=maturity,
        sabr_params=sabr_params,
        forward_price=sabr_params.F0
    )
    
    return [smile_data_reference, smile_data_with_ci]


def main():
    """Main demonstration function."""
    
    print("=== Volatility Smile Visualization Demo ===\n")
    
    # Create output directory
    output_dir = Path("smile_plots_demo")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize plotter with custom configuration
    config = SmilePlotConfig(
        figure_size=(14, 10),
        dpi=150,
        line_width=2.5,
        marker_size=7.0
    )
    
    plotter = SmilePlotter(config=config, output_dir=output_dir)
    
    print("1. Basic Smile Comparison Plot")
    print("-" * 40)
    
    # Generate sample data
    smile_data_list = generate_sample_data()
    
    # Create static comparison plot
    fig1 = plotter.plot_smile_comparison(
        smile_data_list=smile_data_list,
        title="SABR Volatility Smile - Model Comparison",
        save_path="basic_smile_comparison.png",
        show_errors=True,
        moneyness_axis=True
    )
    
    print(f"✓ Static plot saved to: {output_dir}/basic_smile_comparison.png")
    plt.show()
    plt.close(fig1)
    
    print("\n2. Interactive Smile Comparison Plot")
    print("-" * 40)
    
    # Create interactive comparison plot
    fig2 = plotter.plot_interactive_smile_comparison(
        smile_data_list=smile_data_list,
        title="Interactive SABR Volatility Smile Comparison",
        save_path="interactive_smile_comparison.html",
        show_errors=True,
        moneyness_axis=True
    )
    
    print(f"✓ Interactive plot saved to: {output_dir}/interactive_smile_comparison.html")
    
    print("\n3. Multi-Maturity Smile Plots")
    print("-" * 40)
    
    # Generate multi-maturity data
    smile_data_dict = generate_multi_maturity_data()
    
    # Create static multi-maturity plot
    fig3 = plotter.plot_multi_maturity_smiles(
        smile_data_dict=smile_data_dict,
        title="SABR Volatility Smiles - Multiple Maturities",
        save_path="multi_maturity_smiles_static.png",
        interactive=False
    )
    
    print(f"✓ Static multi-maturity plot saved to: {output_dir}/multi_maturity_smiles_static.png")
    plt.show()
    plt.close(fig3)
    
    # Create interactive multi-maturity plot
    fig4 = plotter.plot_multi_maturity_smiles(
        smile_data_dict=smile_data_dict,
        title="Interactive Multi-Maturity SABR Volatility Smiles",
        save_path="multi_maturity_smiles_interactive.html",
        interactive=True
    )
    
    print(f"✓ Interactive multi-maturity plot saved to: {output_dir}/multi_maturity_smiles_interactive.html")
    
    print("\n4. Confidence Interval Visualization")
    print("-" * 40)
    
    # Generate data with confidence intervals
    smile_data_with_ci = generate_confidence_interval_data()
    
    # Create plot with confidence intervals
    fig5 = plotter.plot_smile_comparison(
        smile_data_list=smile_data_with_ci,
        title="Volatility Smile with Confidence Intervals",
        save_path="smile_with_confidence_intervals.png",
        show_confidence_intervals=True,
        show_errors=False
    )
    
    print(f"✓ Confidence interval plot saved to: {output_dir}/smile_with_confidence_intervals.png")
    plt.show()
    plt.close(fig5)
    
    print("\n5. Strike Axis vs Moneyness Axis Comparison")
    print("-" * 40)
    
    # Create plot with strike axis
    fig6 = plotter.plot_smile_comparison(
        smile_data_list=smile_data_list[:2],  # Just HF and LF for clarity
        title="Volatility Smile - Strike Axis",
        save_path="smile_strike_axis.png",
        moneyness_axis=False,
        show_errors=False
    )
    
    print(f"✓ Strike axis plot saved to: {output_dir}/smile_strike_axis.png")
    plt.show()
    plt.close(fig6)
    
    print("\n=== Demo Complete ===")
    print(f"All plots saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file_path in sorted(output_dir.glob("*")):
        print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()