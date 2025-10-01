"""
Simple example of volatility smile visualization tools.

This script demonstrates basic usage without requiring the full data generation pipeline.
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
from data_generation.sabr_params import SABRParams


def create_sample_smile_data():
    """Create sample volatility smile data for demonstration."""
    
    # Define SABR parameters
    sabr_params = SABRParams(
        F0=100.0,
        alpha=0.3,
        beta=0.7,
        nu=0.4,
        rho=-0.3
    )
    
    # Define strikes around the forward price
    strikes = np.array([70, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130])
    maturity = 1.0
    
    # Create realistic volatility smile shapes
    # High-fidelity "true" volatilities (typical smile shape)
    moneyness = strikes / sabr_params.F0
    base_vol = 0.20
    
    # Create smile effect: higher vol for OTM options
    smile_effect = 0.05 * (moneyness - 1.0) ** 2
    hf_volatilities = base_vol + smile_effect + np.random.normal(0, 0.002, len(strikes))
    
    # Low-fidelity approximation (slightly different)
    lf_volatilities = base_vol + 0.8 * smile_effect + np.random.normal(0, 0.003, len(strikes))
    
    # Baseline model (worse approximation)
    baseline_volatilities = base_vol + 0.6 * smile_effect + np.random.normal(0, 0.008, len(strikes))
    
    # MDA-CNN model (better approximation)
    mda_cnn_volatilities = base_vol + 0.95 * smile_effect + np.random.normal(0, 0.004, len(strikes))
    
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


def main():
    """Main demonstration function."""
    
    print("=== Simple Volatility Smile Visualization Demo ===\n")
    
    # Create output directory
    output_dir = Path("simple_smile_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize plotter
    config = SmilePlotConfig(
        figure_size=(12, 8),
        dpi=100,
        line_width=2.0
    )
    
    plotter = SmilePlotter(config=config, output_dir=output_dir)
    
    print("1. Creating sample smile data...")
    smile_data_list = create_sample_smile_data()
    
    print("2. Creating static comparison plot...")
    fig1 = plotter.plot_smile_comparison(
        smile_data_list=smile_data_list,
        title="SABR Volatility Smile - Model Comparison",
        save_path="simple_smile_comparison.png",
        show_errors=True,
        moneyness_axis=True
    )
    
    print(f"✓ Static plot saved to: {output_dir}/simple_smile_comparison.png")
    plt.close(fig1)
    
    print("3. Creating interactive comparison plot...")
    fig2 = plotter.plot_interactive_smile_comparison(
        smile_data_list=smile_data_list,
        title="Interactive SABR Volatility Smile Comparison",
        save_path="simple_interactive_comparison.html",
        show_errors=True,
        moneyness_axis=True
    )
    
    print(f"✓ Interactive plot saved to: {output_dir}/simple_interactive_comparison.html")
    
    print("4. Creating plot with confidence intervals...")
    
    # Add confidence intervals to one model
    smile_with_ci = smile_data_list[3]  # MDA-CNN
    n_strikes = len(smile_with_ci.strikes)
    
    # Simulate confidence intervals
    ci_width = 0.01
    confidence_intervals = {
        'lower': smile_with_ci.volatilities - ci_width,
        'upper': smile_with_ci.volatilities + ci_width
    }
    
    smile_with_ci.confidence_intervals = confidence_intervals
    
    fig3 = plotter.plot_smile_comparison(
        smile_data_list=[smile_data_list[0], smile_with_ci],  # HF and MDA-CNN with CI
        title="Volatility Smile with Confidence Intervals",
        save_path="smile_with_ci.png",
        show_confidence_intervals=True,
        show_errors=False
    )
    
    print(f"✓ Confidence interval plot saved to: {output_dir}/smile_with_ci.png")
    plt.close(fig3)
    
    print("\n=== Demo Complete ===")
    print(f"All plots saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file_path in sorted(output_dir.glob("*")):
        print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()