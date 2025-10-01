"""
Example usage of the comprehensive evaluation system for SABR volatility surface models.

This script demonstrates how to use the evaluation metrics, statistical testing,
and benchmark comparison functionality.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from .metrics import SurfaceEvaluator, RegionBounds
from .surface_evaluator import ComprehensiveEvaluator, ModelPrediction
from .benchmark_comparison import BenchmarkComparator


def create_synthetic_data():
    """Create synthetic SABR volatility surface data for demonstration."""
    np.random.seed(42)
    
    # Grid parameters
    n_strikes = 20
    n_maturities = 10
    strikes = np.linspace(70, 130, n_strikes)
    maturities = np.linspace(0.1, 5.0, n_maturities)
    forward_price = 100.0
    
    # Create strike-maturity grid
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    
    # Synthetic "true" volatility surface (simplified SABR-like)
    moneyness = strike_grid / forward_price
    y_true = 0.2 + 0.1 * np.exp(-maturity_grid) * (moneyness - 1)**2
    
    return y_true, strikes, maturities, forward_price


def create_model_predictions(y_true, strikes, maturities, forward_price, hf_budget=100):
    """Create synthetic model predictions for different models."""
    predictions = []
    
    # Baseline model (simple MLP) - higher error
    baseline_pred = y_true + np.random.normal(0, 0.03, y_true.shape)
    predictions.append(ModelPrediction(
        predictions=baseline_pred,
        model_name="Baseline_MLP",
        hf_budget=hf_budget,
        strikes=strikes,
        maturities=maturities,
        forward_price=forward_price,
        metadata={"model_type": "baseline", "architecture": "MLP"}
    ))
    
    # Residual MLP (without patches) - medium error
    residual_mlp_pred = y_true + np.random.normal(0, 0.02, y_true.shape)
    predictions.append(ModelPrediction(
        predictions=residual_mlp_pred,
        model_name="Residual_MLP",
        hf_budget=hf_budget,
        strikes=strikes,
        maturities=maturities,
        forward_price=forward_price,
        metadata={"model_type": "residual", "architecture": "MLP"}
    ))
    
    # MDA-CNN model - lower error, especially in wings
    mda_cnn_pred = y_true.copy()
    # Add small random error
    mda_cnn_pred += np.random.normal(0, 0.01, y_true.shape)
    # Add slightly more error in ATM region to simulate realistic behavior
    strike_grid, _ = np.meshgrid(strikes, maturities)
    moneyness = strike_grid / forward_price
    atm_mask = (moneyness > 0.95) & (moneyness < 1.05)
    mda_cnn_pred[atm_mask] += np.random.normal(0, 0.005, np.sum(atm_mask))
    
    predictions.append(ModelPrediction(
        predictions=mda_cnn_pred,
        model_name="MDA_CNN",
        hf_budget=hf_budget,
        strikes=strikes,
        maturities=maturities,
        forward_price=forward_price,
        metadata={"model_type": "multi_fidelity", "architecture": "CNN+MLP"}
    ))
    
    return predictions


def demonstrate_basic_evaluation():
    """Demonstrate basic surface evaluation metrics."""
    print("=== Basic Surface Evaluation Demo ===")
    
    # Create synthetic data
    y_true, strikes, maturities, forward_price = create_synthetic_data()
    predictions = create_model_predictions(y_true, strikes, maturities, forward_price)
    
    # Initialize evaluator with custom region bounds
    custom_bounds = RegionBounds(
        atm_lower=0.95, atm_upper=1.05,
        itm_call_upper=0.95, otm_call_lower=1.05
    )
    evaluator = SurfaceEvaluator(custom_bounds)
    
    # Evaluate each model
    for pred in predictions:
        metrics = evaluator.compute_surface_metrics(
            y_true=y_true,
            y_pred=pred.predictions,
            strikes=strikes,
            forward_price=forward_price
        )
        
        print(f"\n{pred.model_name} Results:")
        print(f"  Overall RMSE: {metrics.rmse:.6f}")
        print(f"  Overall MAE:  {metrics.mae:.6f}")
        print(f"  R-squared:    {metrics.r_squared:.6f}")
        print(f"  ATM RMSE:     {metrics.atm_rmse:.6f}")
        print(f"  ITM RMSE:     {metrics.itm_rmse:.6f}")
        print(f"  OTM RMSE:     {metrics.otm_rmse:.6f}")
        print(f"  Mean Bias:    {metrics.mean_bias:.6f}")
        print(f"  Max Error:    {metrics.max_error:.6f}")


def demonstrate_comprehensive_evaluation():
    """Demonstrate comprehensive evaluation with statistical testing."""
    print("\n=== Comprehensive Evaluation Demo ===")
    
    # Create synthetic data
    y_true, strikes, maturities, forward_price = create_synthetic_data()
    predictions = create_model_predictions(y_true, strikes, maturities, forward_price)
    
    # Initialize comprehensive evaluator
    evaluator = ComprehensiveEvaluator(output_dir="evaluation_results")
    
    # Compare all models
    results = evaluator.compare_models(y_true, predictions, baseline_model="Baseline_MLP")
    
    print("\nModel Comparison Results:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  RMSE: {result.metrics.rmse:.6f}")
        print(f"  MAE:  {result.metrics.mae:.6f}")
        
        # Print confidence intervals if available
        if result.confidence_intervals:
            rmse_ci = result.confidence_intervals.get('rmse', {})
            if rmse_ci:
                print(f"  RMSE 95% CI: [{rmse_ci['ci_lower']:.6f}, {rmse_ci['ci_upper']:.6f}]")
        
        # Print statistical test results if available
        if result.statistical_tests:
            for test_name, test_result in result.statistical_tests.items():
                if isinstance(test_result, dict) and 'significant' in test_result:
                    sig_status = "significant" if test_result['significant'] else "not significant"
                    print(f"  {test_name}: {sig_status} (p={test_result['p_value']:.4f})")


def demonstrate_multi_budget_analysis():
    """Demonstrate evaluation across different HF budgets."""
    print("\n=== Multi-Budget Analysis Demo ===")
    
    # Create data for different HF budgets
    budgets = [50, 100, 200, 500]
    y_true_dict = {}
    predictions_dict = {}
    
    for budget in budgets:
        y_true, strikes, maturities, forward_price = create_synthetic_data()
        predictions = create_model_predictions(y_true, strikes, maturities, forward_price, budget)
        
        y_true_dict[budget] = y_true
        predictions_dict[budget] = predictions
    
    # Run multi-budget evaluation
    evaluator = ComprehensiveEvaluator(output_dir="evaluation_results")
    results_df = evaluator.evaluate_across_budgets(y_true_dict, predictions_dict)
    
    # Display results summary
    print("\nPerformance vs HF Budget:")
    summary = results_df.groupby(['model_name', 'hf_budget'])['rmse'].mean().unstack()
    print(summary.round(6))
    
    # Save results
    evaluator.save_results(results_df, "multi_budget_results")
    
    # Generate summary report
    report = evaluator.generate_summary_report(results_df, "evaluation_summary.txt")
    print(f"\nSummary report saved. First few lines:")
    print("\n".join(report.split("\n")[:10]))


def demonstrate_benchmark_comparison():
    """Demonstrate benchmark comparison functionality."""
    print("\n=== Benchmark Comparison Demo ===")
    
    # Create synthetic data
    y_true, strikes, maturities, forward_price = create_synthetic_data()
    predictions = create_model_predictions(y_true, strikes, maturities, forward_price)
    
    # Initialize benchmark comparator
    comparator = BenchmarkComparator(output_dir="evaluation_results")
    
    # Find baseline and other models
    baseline_pred = next(p for p in predictions if p.model_name == "Baseline_MLP")
    other_preds = [p for p in predictions if p.model_name != "Baseline_MLP"]
    
    # Compare models against baseline
    benchmark_results = comparator.compare_multiple_models(
        y_true, other_preds, baseline_pred
    )
    
    print(f"\nBenchmark Results (vs {baseline_pred.model_name}):")
    for result in benchmark_results:
        print(f"\n{result.model_name}:")
        print(f"  RMSE Improvement: {result.rmse_improvement:+.1f}%")
        print(f"  MAE Improvement:  {result.mae_improvement:+.1f}%")
        print(f"  R² Improvement:   {result.r_squared_improvement:+.1f}%")
        print(f"  Significantly Better: {result.is_significantly_better}")
        print(f"  P-value: {result.p_value:.4f}")
        
        # Regional improvements
        if not np.isnan(result.atm_rmse_improvement):
            print(f"  ATM RMSE Improvement: {result.atm_rmse_improvement:+.1f}%")
        if not np.isnan(result.itm_rmse_improvement):
            print(f"  ITM RMSE Improvement: {result.itm_rmse_improvement:+.1f}%")
        if not np.isnan(result.otm_rmse_improvement):
            print(f"  OTM RMSE Improvement: {result.otm_rmse_improvement:+.1f}%")


def demonstrate_cross_budget_benchmark():
    """Demonstrate benchmark comparison across HF budgets."""
    print("\n=== Cross-Budget Benchmark Demo ===")
    
    # Create data for different budgets
    budgets = [100, 200, 500]
    y_true_dict = {}
    predictions_dict = {}
    
    for budget in budgets:
        y_true, strikes, maturities, forward_price = create_synthetic_data()
        predictions = create_model_predictions(y_true, strikes, maturities, forward_price, budget)
        
        y_true_dict[budget] = y_true
        predictions_dict[budget] = predictions
    
    # Run benchmark comparison across budgets
    comparator = BenchmarkComparator(output_dir="evaluation_results")
    benchmark_df = comparator.benchmark_across_budgets(
        y_true_dict, predictions_dict, baseline_name="Baseline_MLP"
    )
    
    # Display summary
    print("\nBenchmark Summary (RMSE Improvement %):")
    pivot_table = benchmark_df.pivot_table(
        values='rmse_improvement', 
        index='model_name', 
        columns='hf_budget'
    )
    print(pivot_table.round(1))
    
    # Generate benchmark report
    report = comparator.generate_benchmark_report(benchmark_df, "benchmark_report.txt")
    print(f"\nBenchmark report generated. Key findings:")
    
    # Extract key statistics
    avg_improvements = benchmark_df.groupby('model_name')['rmse_improvement'].mean()
    sig_improvements = benchmark_df.groupby('model_name')['is_significantly_better'].mean()
    
    for model in avg_improvements.index:
        avg_imp = avg_improvements[model]
        sig_frac = sig_improvements[model]
        print(f"  {model}: {avg_imp:+.1f}% avg improvement, {sig_frac:.1%} significant")


def main():
    """Run all evaluation demonstrations."""
    print("SABR Volatility Surface Evaluation System Demo")
    print("=" * 50)
    
    # Create output directory
    Path("evaluation_results").mkdir(exist_ok=True)
    
    # Run demonstrations
    demonstrate_basic_evaluation()
    demonstrate_comprehensive_evaluation()
    demonstrate_multi_budget_analysis()
    demonstrate_benchmark_comparison()
    demonstrate_cross_budget_benchmark()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check 'evaluation_results/' directory for saved outputs.")


if __name__ == "__main__":
    main()