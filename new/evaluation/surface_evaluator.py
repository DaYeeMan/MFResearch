"""
Surface-specific evaluation pipeline for SABR volatility surface models.

This module provides comprehensive evaluation capabilities including
multi-budget analysis, model comparison, and detailed reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import warnings

from .metrics import SurfaceEvaluator, SurfaceMetrics, StatisticalTester, RegionBounds


@dataclass
class ModelPrediction:
    """Container for model predictions and metadata."""
    predictions: np.ndarray
    model_name: str
    hf_budget: int
    strikes: np.ndarray
    maturities: np.ndarray
    forward_price: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    hf_budget: int
    metrics: SurfaceMetrics
    statistical_tests: Optional[Dict[str, Any]] = None
    confidence_intervals: Optional[Dict[str, Any]] = None


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation pipeline for volatility surface models.
    
    Supports multi-budget analysis, statistical testing, and detailed reporting.
    """
    
    def __init__(
        self,
        region_bounds: Optional[RegionBounds] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize comprehensive evaluator.
        
        Args:
            region_bounds: Custom region boundaries for analysis
            output_dir: Directory to save evaluation results
        """
        self.surface_evaluator = SurfaceEvaluator(region_bounds)
        self.statistical_tester = StatisticalTester()
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_single_model(
        self,
        y_true: np.ndarray,
        prediction: ModelPrediction,
        mask: Optional[np.ndarray] = None
    ) -> EvaluationResult:
        """
        Evaluate a single model prediction.
        
        Args:
            y_true: True volatility values
            prediction: Model prediction container
            mask: Optional mask for valid data points
            
        Returns:
            EvaluationResult with computed metrics
        """
        metrics = self.surface_evaluator.compute_surface_metrics(
            y_true=y_true,
            y_pred=prediction.predictions,
            strikes=prediction.strikes,
            forward_price=prediction.forward_price,
            mask=mask
        )
        
        # Compute confidence intervals for key metrics
        errors = np.abs(y_true.flatten() - prediction.predictions.flatten())
        if mask is not None:
            errors = errors[mask.flatten()]
        
        # Remove invalid values
        errors = errors[np.isfinite(errors)]
        
        confidence_intervals = {}
        if len(errors) > 10:  # Need sufficient data for bootstrap
            try:
                confidence_intervals['rmse'] = self.statistical_tester.bootstrap_confidence_interval(
                    errors, lambda x: np.sqrt(np.mean(x**2))
                )
                confidence_intervals['mae'] = self.statistical_tester.bootstrap_confidence_interval(
                    errors, np.mean
                )
            except Exception as e:
                warnings.warn(f"Could not compute confidence intervals: {e}")
        
        return EvaluationResult(
            model_name=prediction.model_name,
            hf_budget=prediction.hf_budget,
            metrics=metrics,
            confidence_intervals=confidence_intervals
        )
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: List[ModelPrediction],
        mask: Optional[np.ndarray] = None,
        baseline_model: Optional[str] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple model predictions with statistical testing.
        
        Args:
            y_true: True volatility values
            predictions: List of model predictions to compare
            mask: Optional mask for valid data points
            baseline_model: Name of baseline model for comparison
            
        Returns:
            Dictionary of evaluation results with statistical tests
        """
        results = {}
        
        # Evaluate each model individually
        for pred in predictions:
            results[pred.model_name] = self.evaluate_single_model(y_true, pred, mask)
        
        # Perform pairwise statistical tests
        if len(predictions) > 1:
            self._add_statistical_tests(y_true, predictions, results, mask, baseline_model)
        
        return results
    
    def evaluate_across_budgets(
        self,
        y_true_dict: Dict[int, np.ndarray],
        predictions_dict: Dict[int, List[ModelPrediction]],
        mask_dict: Optional[Dict[int, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Evaluate models across different HF budget sizes.
        
        Args:
            y_true_dict: Dictionary mapping budget -> true values
            predictions_dict: Dictionary mapping budget -> list of predictions
            mask_dict: Optional dictionary mapping budget -> masks
            
        Returns:
            DataFrame with evaluation results across budgets
        """
        all_results = []
        
        for budget in sorted(y_true_dict.keys()):
            y_true = y_true_dict[budget]
            predictions = predictions_dict[budget]
            mask = mask_dict.get(budget) if mask_dict else None
            
            # Compare models for this budget
            budget_results = self.compare_models(y_true, predictions, mask)
            
            # Convert to DataFrame rows
            for model_name, result in budget_results.items():
                row = {
                    'hf_budget': budget,
                    'model_name': model_name,
                    **self._metrics_to_dict(result.metrics)
                }
                
                # Add confidence intervals if available
                if result.confidence_intervals:
                    for metric, ci_info in result.confidence_intervals.items():
                        row[f'{metric}_ci_lower'] = ci_info['ci_lower']
                        row[f'{metric}_ci_upper'] = ci_info['ci_upper']
                
                # Add statistical test results if available
                if result.statistical_tests:
                    for test_name, test_result in result.statistical_tests.items():
                        if isinstance(test_result, dict):
                            for key, value in test_result.items():
                                row[f'{test_name}_{key}'] = value
                
                all_results.append(row)
        
        return pd.DataFrame(all_results)
    
    def _add_statistical_tests(
        self,
        y_true: np.ndarray,
        predictions: List[ModelPrediction],
        results: Dict[str, EvaluationResult],
        mask: Optional[np.ndarray],
        baseline_model: Optional[str]
    ):
        """Add statistical test results to evaluation results."""
        # Prepare error arrays for each model
        model_errors = {}
        
        for pred in predictions:
            errors = np.abs(y_true.flatten() - pred.predictions.flatten())
            if mask is not None:
                errors = errors[mask.flatten()]
            errors = errors[np.isfinite(errors)]
            model_errors[pred.model_name] = errors
        
        # Perform pairwise tests
        model_names = list(model_errors.keys())
        
        for i, model1 in enumerate(model_names):
            if model1 not in results:
                continue
                
            results[model1].statistical_tests = {}
            
            for j, model2 in enumerate(model_names):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                errors1 = model_errors[model1]
                errors2 = model_errors[model2]
                
                # Ensure same length for paired tests
                min_len = min(len(errors1), len(errors2))
                errors1_paired = errors1[:min_len]
                errors2_paired = errors2[:min_len]
                
                if len(errors1_paired) > 5:  # Need sufficient data
                    try:
                        # Paired t-test
                        t_test = self.statistical_tester.paired_t_test(
                            errors1_paired, errors2_paired
                        )
                        results[model1].statistical_tests[f'vs_{model2}_ttest'] = t_test
                        
                        # Wilcoxon test (non-parametric)
                        wilcoxon_test = self.statistical_tester.wilcoxon_test(
                            errors1_paired, errors2_paired
                        )
                        results[model1].statistical_tests[f'vs_{model2}_wilcoxon'] = wilcoxon_test
                        
                    except Exception as e:
                        warnings.warn(f"Statistical test failed for {model1} vs {model2}: {e}")
        
        # Add baseline comparisons if specified
        if baseline_model and baseline_model in model_errors:
            baseline_errors = model_errors[baseline_model]
            
            for model_name in model_names:
                if model_name == baseline_model or model_name not in results:
                    continue
                
                model_errors_current = model_errors[model_name]
                min_len = min(len(baseline_errors), len(model_errors_current))
                
                if min_len > 5:
                    try:
                        # Test against baseline
                        baseline_test = self.statistical_tester.paired_t_test(
                            model_errors_current[:min_len],
                            baseline_errors[:min_len]
                        )
                        
                        if 'statistical_tests' not in results[model_name].__dict__:
                            results[model_name].statistical_tests = {}
                        
                        results[model_name].statistical_tests['vs_baseline'] = baseline_test
                        
                    except Exception as e:
                        warnings.warn(f"Baseline comparison failed for {model_name}: {e}")
    
    def _metrics_to_dict(self, metrics: SurfaceMetrics) -> Dict[str, float]:
        """Convert SurfaceMetrics to dictionary."""
        return asdict(metrics)
    
    def save_results(
        self,
        results: Union[Dict[str, EvaluationResult], pd.DataFrame],
        filename: str
    ):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results to save
            filename: Output filename
        """
        if self.output_dir is None:
            raise ValueError("Output directory not specified")
        
        filepath = self.output_dir / filename
        
        if isinstance(results, pd.DataFrame):
            # Save DataFrame as CSV
            results.to_csv(filepath.with_suffix('.csv'), index=False)
        else:
            # Save dictionary results as JSON
            serializable_results = {}
            for model_name, result in results.items():
                serializable_results[model_name] = {
                    'model_name': result.model_name,
                    'hf_budget': result.hf_budget,
                    'metrics': asdict(result.metrics),
                    'statistical_tests': result.statistical_tests,
                    'confidence_intervals': result.confidence_intervals
                }
            
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
    
    def generate_summary_report(
        self,
        results_df: pd.DataFrame,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Generate a summary report of evaluation results.
        
        Args:
            results_df: DataFrame with evaluation results
            output_filename: Optional filename to save report
            
        Returns:
            Summary report as string
        """
        report_lines = []
        report_lines.append("SABR Volatility Surface Model Evaluation Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("Overall Statistics:")
        report_lines.append(f"Total evaluations: {len(results_df)}")
        report_lines.append(f"Models evaluated: {results_df['model_name'].nunique()}")
        report_lines.append(f"HF budgets tested: {sorted(results_df['hf_budget'].unique())}")
        report_lines.append("")
        
        # Best performing models by budget
        report_lines.append("Best Performing Models by HF Budget (RMSE):")
        for budget in sorted(results_df['hf_budget'].unique()):
            budget_data = results_df[results_df['hf_budget'] == budget]
            best_model = budget_data.loc[budget_data['rmse'].idxmin()]
            report_lines.append(
                f"Budget {budget}: {best_model['model_name']} "
                f"(RMSE: {best_model['rmse']:.6f})"
            )
        report_lines.append("")
        
        # Performance summary by model
        report_lines.append("Performance Summary by Model:")
        model_summary = results_df.groupby('model_name').agg({
            'rmse': ['mean', 'std', 'min'],
            'mae': ['mean', 'std', 'min'],
            'r_squared': ['mean', 'std', 'max']
        }).round(6)
        
        for model in model_summary.index:
            report_lines.append(f"\n{model}:")
            report_lines.append(f"  RMSE: {model_summary.loc[model, ('rmse', 'mean')]:.6f} ± {model_summary.loc[model, ('rmse', 'std')]:.6f}")
            report_lines.append(f"  MAE:  {model_summary.loc[model, ('mae', 'mean')]:.6f} ± {model_summary.loc[model, ('mae', 'std')]:.6f}")
            report_lines.append(f"  R²:   {model_summary.loc[model, ('r_squared', 'mean')]:.6f} ± {model_summary.loc[model, ('r_squared', 'std')]:.6f}")
        
        report_lines.append("")
        
        # Regional performance analysis
        if 'atm_rmse' in results_df.columns:
            report_lines.append("Regional Performance Analysis (Average RMSE):")
            regional_avg = results_df.groupby('model_name')[['atm_rmse', 'itm_rmse', 'otm_rmse']].mean()
            
            for model in regional_avg.index:
                report_lines.append(f"\n{model}:")
                report_lines.append(f"  ATM: {regional_avg.loc[model, 'atm_rmse']:.6f}")
                report_lines.append(f"  ITM: {regional_avg.loc[model, 'itm_rmse']:.6f}")
                report_lines.append(f"  OTM: {regional_avg.loc[model, 'otm_rmse']:.6f}")
        
        report_text = "\n".join(report_lines)
        
        # Save report if filename provided
        if output_filename and self.output_dir:
            with open(self.output_dir / output_filename, 'w') as f:
                f.write(report_text)
        
        return report_text