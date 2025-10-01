"""
Benchmark comparison utilities for SABR volatility surface models.

This module provides standardized comparison against baseline models
and reference implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings

from .metrics import SurfaceEvaluator, SurfaceMetrics, compute_relative_improvement
from .surface_evaluator import ModelPrediction, EvaluationResult, ComprehensiveEvaluator


@dataclass
class BenchmarkResult:
    """Container for benchmark comparison results."""
    model_name: str
    baseline_name: str
    hf_budget: int
    
    # Relative improvements (positive = better than baseline)
    rmse_improvement: float
    mae_improvement: float
    r_squared_improvement: float
    
    # Regional improvements
    atm_rmse_improvement: float
    itm_rmse_improvement: float
    otm_rmse_improvement: float
    
    # Statistical significance
    is_significantly_better: bool
    p_value: float
    
    # Raw metrics for reference
    model_metrics: SurfaceMetrics
    baseline_metrics: SurfaceMetrics


class BenchmarkComparator:
    """
    Standardized benchmark comparison for volatility surface models.
    
    Provides relative improvement calculations and statistical significance testing
    against baseline models.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize benchmark comparator.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.evaluator = ComprehensiveEvaluator(output_dir=output_dir)
        self.output_dir = Path(output_dir) if output_dir else None
    
    def compare_against_baseline(
        self,
        y_true: np.ndarray,
        model_prediction: ModelPrediction,
        baseline_prediction: ModelPrediction,
        mask: Optional[np.ndarray] = None,
        significance_level: float = 0.05
    ) -> BenchmarkResult:
        """
        Compare a model against a baseline with statistical testing.
        
        Args:
            y_true: True volatility values
            model_prediction: Model prediction to evaluate
            baseline_prediction: Baseline model prediction
            mask: Optional mask for valid data points
            significance_level: Significance level for statistical tests
            
        Returns:
            BenchmarkResult with comparison metrics
        """
        # Evaluate both models
        model_result = self.evaluator.evaluate_single_model(y_true, model_prediction, mask)
        baseline_result = self.evaluator.evaluate_single_model(y_true, baseline_prediction, mask)
        
        # Compute relative improvements
        rmse_improvement = compute_relative_improvement(
            baseline_result.metrics.rmse, model_result.metrics.rmse, lower_is_better=True
        )
        mae_improvement = compute_relative_improvement(
            baseline_result.metrics.mae, model_result.metrics.mae, lower_is_better=True
        )
        r_squared_improvement = compute_relative_improvement(
            baseline_result.metrics.r_squared, model_result.metrics.r_squared, lower_is_better=False
        )
        
        # Regional improvements
        atm_rmse_improvement = compute_relative_improvement(
            baseline_result.metrics.atm_rmse, model_result.metrics.atm_rmse, lower_is_better=True
        ) if not np.isnan(baseline_result.metrics.atm_rmse) else np.nan
        
        itm_rmse_improvement = compute_relative_improvement(
            baseline_result.metrics.itm_rmse, model_result.metrics.itm_rmse, lower_is_better=True
        ) if not np.isnan(baseline_result.metrics.itm_rmse) else np.nan
        
        otm_rmse_improvement = compute_relative_improvement(
            baseline_result.metrics.otm_rmse, model_result.metrics.otm_rmse, lower_is_better=True
        ) if not np.isnan(baseline_result.metrics.otm_rmse) else np.nan
        
        # Statistical significance testing
        model_errors = np.abs(y_true.flatten() - model_prediction.predictions.flatten())
        baseline_errors = np.abs(y_true.flatten() - baseline_prediction.predictions.flatten())
        
        if mask is not None:
            mask_flat = mask.flatten()
            model_errors = model_errors[mask_flat]
            baseline_errors = baseline_errors[mask_flat]
        
        # Remove invalid values
        valid_idx = np.isfinite(model_errors) & np.isfinite(baseline_errors)
        model_errors = model_errors[valid_idx]
        baseline_errors = baseline_errors[valid_idx]
        
        # Perform statistical test
        is_significantly_better = False
        p_value = 1.0
        
        if len(model_errors) > 5:
            try:
                test_result = self.evaluator.statistical_tester.paired_t_test(
                    model_errors, baseline_errors, alpha=significance_level
                )
                p_value = test_result['p_value']
                # Model is significantly better if errors are significantly lower
                is_significantly_better = (
                    test_result['significant'] and 
                    test_result['mean_diff'] < 0  # Model errors < baseline errors
                )
            except Exception as e:
                warnings.warn(f"Statistical test failed: {e}")
        
        return BenchmarkResult(
            model_name=model_prediction.model_name,
            baseline_name=baseline_prediction.model_name,
            hf_budget=model_prediction.hf_budget,
            rmse_improvement=rmse_improvement,
            mae_improvement=mae_improvement,
            r_squared_improvement=r_squared_improvement,
            atm_rmse_improvement=atm_rmse_improvement,
            itm_rmse_improvement=itm_rmse_improvement,
            otm_rmse_improvement=otm_rmse_improvement,
            is_significantly_better=is_significantly_better,
            p_value=p_value,
            model_metrics=model_result.metrics,
            baseline_metrics=baseline_result.metrics
        )
    
    def compare_multiple_models(
        self,
        y_true: np.ndarray,
        model_predictions: List[ModelPrediction],
        baseline_prediction: ModelPrediction,
        mask: Optional[np.ndarray] = None
    ) -> List[BenchmarkResult]:
        """
        Compare multiple models against a single baseline.
        
        Args:
            y_true: True volatility values
            model_predictions: List of model predictions to compare
            baseline_prediction: Baseline model prediction
            mask: Optional mask for valid data points
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for model_pred in model_predictions:
            if model_pred.model_name == baseline_prediction.model_name:
                continue  # Skip comparing baseline to itself
            
            result = self.compare_against_baseline(
                y_true, model_pred, baseline_prediction, mask
            )
            results.append(result)
        
        return results
    
    def benchmark_across_budgets(
        self,
        y_true_dict: Dict[int, np.ndarray],
        model_predictions_dict: Dict[int, List[ModelPrediction]],
        baseline_name: str,
        mask_dict: Optional[Dict[int, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Benchmark models across different HF budget sizes.
        
        Args:
            y_true_dict: Dictionary mapping budget -> true values
            model_predictions_dict: Dictionary mapping budget -> list of predictions
            baseline_name: Name of baseline model to compare against
            mask_dict: Optional dictionary mapping budget -> masks
            
        Returns:
            DataFrame with benchmark results across budgets
        """
        all_results = []
        
        for budget in sorted(y_true_dict.keys()):
            y_true = y_true_dict[budget]
            predictions = model_predictions_dict[budget]
            mask = mask_dict.get(budget) if mask_dict else None
            
            # Find baseline prediction
            baseline_pred = None
            model_preds = []
            
            for pred in predictions:
                if pred.model_name == baseline_name:
                    baseline_pred = pred
                else:
                    model_preds.append(pred)
            
            if baseline_pred is None:
                warnings.warn(f"Baseline model '{baseline_name}' not found for budget {budget}")
                continue
            
            # Compare all models against baseline
            budget_results = self.compare_multiple_models(
                y_true, model_preds, baseline_pred, mask
            )
            
            # Convert to DataFrame rows
            for result in budget_results:
                row = {
                    'hf_budget': budget,
                    'model_name': result.model_name,
                    'baseline_name': result.baseline_name,
                    'rmse_improvement': result.rmse_improvement,
                    'mae_improvement': result.mae_improvement,
                    'r_squared_improvement': result.r_squared_improvement,
                    'atm_rmse_improvement': result.atm_rmse_improvement,
                    'itm_rmse_improvement': result.itm_rmse_improvement,
                    'otm_rmse_improvement': result.otm_rmse_improvement,
                    'is_significantly_better': result.is_significantly_better,
                    'p_value': result.p_value,
                    
                    # Raw metrics for reference
                    'model_rmse': result.model_metrics.rmse,
                    'baseline_rmse': result.baseline_metrics.rmse,
                    'model_mae': result.model_metrics.mae,
                    'baseline_mae': result.baseline_metrics.mae,
                    'model_r_squared': result.model_metrics.r_squared,
                    'baseline_r_squared': result.baseline_metrics.r_squared
                }
                
                all_results.append(row)
        
        return pd.DataFrame(all_results)
    
    def generate_benchmark_report(
        self,
        benchmark_df: pd.DataFrame,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive benchmark comparison report.
        
        Args:
            benchmark_df: DataFrame with benchmark results
            output_filename: Optional filename to save report
            
        Returns:
            Benchmark report as string
        """
        report_lines = []
        report_lines.append("SABR Model Benchmark Comparison Report")
        report_lines.append("=" * 45)
        report_lines.append("")
        
        if len(benchmark_df) == 0:
            report_lines.append("No benchmark results to report.")
            return "\n".join(report_lines)
        
        baseline_name = benchmark_df['baseline_name'].iloc[0]
        report_lines.append(f"Baseline Model: {baseline_name}")
        report_lines.append("")
        
        # Overall improvement statistics
        report_lines.append("Overall Improvement Statistics:")
        improvement_stats = benchmark_df.groupby('model_name').agg({
            'rmse_improvement': ['mean', 'std', 'min', 'max'],
            'mae_improvement': ['mean', 'std', 'min', 'max'],
            'is_significantly_better': 'mean'  # Fraction of significant improvements
        }).round(2)
        
        for model in improvement_stats.index:
            report_lines.append(f"\n{model}:")
            rmse_mean = improvement_stats.loc[model, ('rmse_improvement', 'mean')]
            rmse_std = improvement_stats.loc[model, ('rmse_improvement', 'std')]
            mae_mean = improvement_stats.loc[model, ('mae_improvement', 'mean')]
            mae_std = improvement_stats.loc[model, ('mae_improvement', 'std')]
            sig_frac = improvement_stats.loc[model, ('is_significantly_better', 'mean')]
            
            report_lines.append(f"  RMSE Improvement: {rmse_mean:.1f}% ± {rmse_std:.1f}%")
            report_lines.append(f"  MAE Improvement:  {mae_mean:.1f}% ± {mae_std:.1f}%")
            report_lines.append(f"  Significantly Better: {sig_frac:.1%} of cases")
        
        report_lines.append("")
        
        # Performance by HF budget
        report_lines.append("Performance by HF Budget:")
        budget_performance = benchmark_df.groupby(['hf_budget', 'model_name']).agg({
            'rmse_improvement': 'mean',
            'mae_improvement': 'mean',
            'is_significantly_better': 'any'
        }).round(2)
        
        for budget in sorted(benchmark_df['hf_budget'].unique()):
            report_lines.append(f"\nHF Budget: {budget}")
            budget_data = budget_performance.loc[budget]
            
            for model in budget_data.index:
                rmse_imp = budget_data.loc[model, 'rmse_improvement']
                mae_imp = budget_data.loc[model, 'mae_improvement']
                is_sig = budget_data.loc[model, 'is_significantly_better']
                sig_marker = " *" if is_sig else ""
                
                report_lines.append(
                    f"  {model}: RMSE {rmse_imp:+.1f}%, MAE {mae_imp:+.1f}%{sig_marker}"
                )
        
        report_lines.append("")
        report_lines.append("* Statistically significant improvement")
        
        # Regional performance analysis
        regional_cols = ['atm_rmse_improvement', 'itm_rmse_improvement', 'otm_rmse_improvement']
        if all(col in benchmark_df.columns for col in regional_cols):
            report_lines.append("")
            report_lines.append("Regional Performance Analysis (Average RMSE Improvement):")
            
            regional_stats = benchmark_df.groupby('model_name')[regional_cols].mean().round(1)
            
            for model in regional_stats.index:
                report_lines.append(f"\n{model}:")
                report_lines.append(f"  ATM: {regional_stats.loc[model, 'atm_rmse_improvement']:+.1f}%")
                report_lines.append(f"  ITM: {regional_stats.loc[model, 'itm_rmse_improvement']:+.1f}%")
                report_lines.append(f"  OTM: {regional_stats.loc[model, 'otm_rmse_improvement']:+.1f}%")
        
        report_text = "\n".join(report_lines)
        
        # Save report if filename provided
        if output_filename and self.output_dir:
            with open(self.output_dir / output_filename, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_benchmark_results(
        self,
        benchmark_df: pd.DataFrame,
        filename: str = "benchmark_results.csv"
    ):
        """
        Save benchmark results to CSV file.
        
        Args:
            benchmark_df: DataFrame with benchmark results
            filename: Output filename
        """
        if self.output_dir is None:
            raise ValueError("Output directory not specified")
        
        filepath = self.output_dir / filename
        benchmark_df.to_csv(filepath, index=False)


def create_performance_summary(
    benchmark_df: pd.DataFrame,
    metric_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a performance summary table from benchmark results.
    
    Args:
        benchmark_df: DataFrame with benchmark results
        metric_columns: Columns to include in summary
        
    Returns:
        Summary DataFrame with key performance metrics
    """
    if metric_columns is None:
        metric_columns = [
            'rmse_improvement', 'mae_improvement', 'r_squared_improvement',
            'is_significantly_better'
        ]
    
    # Group by model and compute summary statistics
    summary = benchmark_df.groupby('model_name').agg({
        col: ['mean', 'std', 'min', 'max'] if col != 'is_significantly_better' 
        else ['mean', 'sum'] for col in metric_columns
    }).round(3)
    
    # Flatten column names
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    
    # Add count of evaluations
    summary['n_evaluations'] = benchmark_df.groupby('model_name').size()
    
    return summary