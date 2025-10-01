"""
Results aggregation and statistical analysis for HF budget experiments.

This module provides comprehensive analysis of experiment results including:
- Performance comparison across HF budgets
- Statistical significance testing
- Model ranking and selection
- Trend analysis and visualization data preparation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import StatisticalTester
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelPerformance:
    """Container for model performance statistics."""
    model_type: str
    hf_budget: int
    mean_mse: float
    std_mse: float
    mean_mae: float
    std_mae: float
    mean_training_time: float
    std_training_time: float
    n_trials: int
    best_mse: float
    worst_mse: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'hf_budget': self.hf_budget,
            'mean_mse': self.mean_mse,
            'std_mse': self.std_mse,
            'mean_mae': self.mean_mae,
            'std_mae': self.std_mae,
            'mean_training_time': self.mean_training_time,
            'std_training_time': self.std_training_time,
            'n_trials': self.n_trials,
            'best_mse': self.best_mse,
            'worst_mse': self.worst_mse
        }


@dataclass
class ComparisonResult:
    """Container for model comparison results."""
    model1: str
    model2: str
    hf_budget: int
    improvement_percent: float
    p_value: float
    is_significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]


class ResultsAggregator:
    """
    Comprehensive results aggregation and analysis for HF budget experiments.
    
    Provides statistical analysis, model comparison, and trend analysis
    across different HF budgets and model architectures.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize results aggregator.
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.statistical_tester = StatisticalTester()
        
        # Load results data
        self.detailed_results = self._load_detailed_results()
        self.model_performances = self._compute_model_performances()
        
        logger.info(f"Results aggregator initialized with {len(self.detailed_results)} experiments")
    
    def _load_detailed_results(self) -> pd.DataFrame:
        """Load detailed experiment results."""
        results_file = self.results_dir / "detailed_results.csv"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        df = pd.read_csv(results_file)
        logger.info(f"Loaded {len(df)} experiment results")
        
        return df
    
    def _compute_model_performances(self) -> List[ModelPerformance]:
        """Compute performance statistics for each model-budget combination."""
        performances = []
        
        # Group by model type and HF budget
        grouped = self.detailed_results.groupby(['model_type', 'hf_budget'])
        
        for (model_type, hf_budget), group in grouped:
            performance = ModelPerformance(
                model_type=model_type,
                hf_budget=hf_budget,
                mean_mse=group['mse'].mean(),
                std_mse=group['mse'].std(),
                mean_mae=group['mae'].mean(),
                std_mae=group['mae'].std(),
                mean_training_time=group['training_time'].mean(),
                std_training_time=group['training_time'].std(),
                n_trials=len(group),
                best_mse=group['mse'].min(),
                worst_mse=group['mse'].max()
            )
            performances.append(performance)
        
        return performances
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of model performances."""
        summary_data = [perf.to_dict() for perf in self.model_performances]
        return pd.DataFrame(summary_data)
    
    def compare_models(
        self,
        model1: str,
        model2: str,
        metric: str = 'mse',
        alpha: float = 0.05
    ) -> List[ComparisonResult]:
        """
        Compare two models across all HF budgets.
        
        Args:
            model1: First model type
            model2: Second model type
            metric: Metric to compare ('mse', 'mae')
            alpha: Significance level
            
        Returns:
            List of comparison results for each HF budget
        """
        comparisons = []
        
        # Get unique HF budgets
        hf_budgets = self.detailed_results['hf_budget'].unique()
        
        for hf_budget in sorted(hf_budgets):
            # Get data for both models at this budget
            model1_data = self.detailed_results[
                (self.detailed_results['model_type'] == model1) &
                (self.detailed_results['hf_budget'] == hf_budget)
            ][metric].values
            
            model2_data = self.detailed_results[
                (self.detailed_results['model_type'] == model2) &
                (self.detailed_results['hf_budget'] == hf_budget)
            ][metric].values
            
            if len(model1_data) == 0 or len(model2_data) == 0:
                logger.warning(f"No data for comparison at HF budget {hf_budget}")
                continue
            
            # Perform statistical test
            test_result = self.statistical_tester.paired_t_test(
                model1_data, model2_data, alpha=alpha
            )
            
            # Compute improvement percentage
            improvement = ((model1_data.mean() - model2_data.mean()) / model1_data.mean()) * 100
            
            # Compute effect size (Cohen's d)
            pooled_std = np.sqrt(((len(model1_data) - 1) * model1_data.var() + 
                                 (len(model2_data) - 1) * model2_data.var()) / 
                                (len(model1_data) + len(model2_data) - 2))
            effect_size = (model1_data.mean() - model2_data.mean()) / pooled_std
            
            # Compute confidence interval for difference
            diff = model1_data.mean() - model2_data.mean()
            se_diff = pooled_std * np.sqrt(1/len(model1_data) + 1/len(model2_data))
            t_critical = stats.t.ppf(1 - alpha/2, len(model1_data) + len(model2_data) - 2)
            ci_lower = diff - t_critical * se_diff
            ci_upper = diff + t_critical * se_diff
            
            comparison = ComparisonResult(
                model1=model1,
                model2=model2,
                hf_budget=hf_budget,
                improvement_percent=improvement,
                p_value=test_result['p_value'],
                is_significant=test_result['significant'],
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper)
            )
            
            comparisons.append(comparison)
        
        return comparisons
    
    def rank_models_by_budget(self, metric: str = 'mse') -> Dict[int, List[Tuple[str, float]]]:
        """
        Rank models by performance for each HF budget.
        
        Args:
            metric: Metric to rank by ('mse', 'mae')
            
        Returns:
            Dictionary mapping HF budget to ranked list of (model_type, score)
        """
        rankings = {}
        
        # Get performance summary
        summary = self.get_performance_summary()
        
        # Group by HF budget
        for hf_budget in summary['hf_budget'].unique():
            budget_data = summary[summary['hf_budget'] == hf_budget]
            
            # Sort by metric (ascending for error metrics)
            metric_col = f'mean_{metric}'
            sorted_data = budget_data.sort_values(metric_col)
            
            # Create ranking
            ranking = [
                (row['model_type'], row[metric_col])
                for _, row in sorted_data.iterrows()
            ]
            
            rankings[hf_budget] = ranking
        
        return rankings
    
    def analyze_budget_efficiency(self, model_type: str, metric: str = 'mse') -> Dict[str, Any]:
        """
        Analyze how model performance scales with HF budget.
        
        Args:
            model_type: Model type to analyze
            metric: Metric to analyze
            
        Returns:
            Dictionary with efficiency analysis results
        """
        # Get data for this model
        model_data = self.detailed_results[
            self.detailed_results['model_type'] == model_type
        ]
        
        if len(model_data) == 0:
            raise ValueError(f"No data found for model type: {model_type}")
        
        # Group by HF budget and compute statistics
        budget_stats = model_data.groupby('hf_budget')[metric].agg(['mean', 'std', 'count'])
        budget_stats = budget_stats.reset_index()
        
        # Fit power law: performance = a * budget^b
        budgets = budget_stats['hf_budget'].values
        performances = budget_stats['mean'].values
        
        # Log-transform for linear regression
        log_budgets = np.log(budgets)
        log_performances = np.log(performances)
        
        # Fit linear regression in log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_budgets, log_performances)
        
        # Convert back to power law parameters
        a = np.exp(intercept)
        b = slope
        
        # Compute efficiency metrics
        efficiency_analysis = {
            'model_type': model_type,
            'power_law_a': a,
            'power_law_b': b,
            'r_squared': r_value**2,
            'p_value': p_value,
            'budget_range': (budgets.min(), budgets.max()),
            'performance_range': (performances.min(), performances.max()),
            'improvement_per_doubling': (2**b - 1) * 100,  # % improvement when doubling budget
            'budget_stats': budget_stats.to_dict('records')
        }
        
        return efficiency_analysis
    
    def find_optimal_budget(
        self,
        model_type: str,
        target_performance: float,
        metric: str = 'mse'
    ) -> Optional[int]:
        """
        Find minimum HF budget needed to achieve target performance.
        
        Args:
            model_type: Model type to analyze
            target_performance: Target performance level
            metric: Metric to use
            
        Returns:
            Minimum HF budget needed, or None if target not achievable
        """
        # Get efficiency analysis
        efficiency = self.analyze_budget_efficiency(model_type, metric)
        
        # Use power law to estimate required budget
        a = efficiency['power_law_a']
        b = efficiency['power_law_b']
        
        # Solve: target_performance = a * budget^b
        # budget = (target_performance / a)^(1/b)
        
        if b >= 0:  # Performance gets worse with more budget (shouldn't happen)
            return None
        
        required_budget = (target_performance / a) ** (1/b)
        
        # Check if this is within the tested range
        budget_range = efficiency['budget_range']
        if required_budget < budget_range[0] or required_budget > budget_range[1]:
            logger.warning(f"Required budget {required_budget:.0f} is outside tested range {budget_range}")
        
        return int(np.ceil(required_budget))
    
    def generate_performance_vs_budget_data(self) -> pd.DataFrame:
        """Generate data for performance vs budget plots."""
        summary = self.get_performance_summary()
        
        # Add confidence intervals
        summary['mse_ci_lower'] = summary['mean_mse'] - 1.96 * summary['std_mse'] / np.sqrt(summary['n_trials'])
        summary['mse_ci_upper'] = summary['mean_mse'] + 1.96 * summary['std_mse'] / np.sqrt(summary['n_trials'])
        summary['mae_ci_lower'] = summary['mean_mae'] - 1.96 * summary['std_mae'] / np.sqrt(summary['n_trials'])
        summary['mae_ci_upper'] = summary['mean_mae'] + 1.96 * summary['std_mae'] / np.sqrt(summary['n_trials'])
        
        return summary
    
    def compute_relative_improvements(self, baseline_model: str = 'direct_mlp') -> pd.DataFrame:
        """
        Compute relative improvements over baseline model.
        
        Args:
            baseline_model: Baseline model for comparison
            
        Returns:
            DataFrame with relative improvements
        """
        summary = self.get_performance_summary()
        improvements = []
        
        # Get baseline performance for each budget
        baseline_data = summary[summary['model_type'] == baseline_model]
        
        for _, baseline_row in baseline_data.iterrows():
            hf_budget = baseline_row['hf_budget']
            baseline_mse = baseline_row['mean_mse']
            baseline_mae = baseline_row['mean_mae']
            
            # Compare all other models at this budget
            budget_data = summary[summary['hf_budget'] == hf_budget]
            
            for _, model_row in budget_data.iterrows():
                if model_row['model_type'] == baseline_model:
                    continue
                
                mse_improvement = ((baseline_mse - model_row['mean_mse']) / baseline_mse) * 100
                mae_improvement = ((baseline_mae - model_row['mean_mae']) / baseline_mae) * 100
                
                improvements.append({
                    'model_type': model_row['model_type'],
                    'hf_budget': hf_budget,
                    'mse_improvement_percent': mse_improvement,
                    'mae_improvement_percent': mae_improvement,
                    'baseline_model': baseline_model
                })
        
        return pd.DataFrame(improvements)
    
    def save_analysis_results(self, output_dir: Optional[str] = None):
        """Save all analysis results to files."""
        if output_dir is None:
            output_dir = self.results_dir / "analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save performance summary
        summary = self.get_performance_summary()
        summary.to_csv(output_dir / "performance_summary.csv", index=False)
        
        # Save model rankings
        rankings = self.rank_models_by_budget()
        with open(output_dir / "model_rankings.json", 'w') as f:
            json.dump(rankings, f, indent=2, default=str)
        
        # Save efficiency analysis for each model
        efficiency_results = {}
        for model_type in summary['model_type'].unique():
            try:
                efficiency = self.analyze_budget_efficiency(model_type)
                efficiency_results[model_type] = efficiency
            except Exception as e:
                logger.warning(f"Could not analyze efficiency for {model_type}: {e}")
        
        with open(output_dir / "efficiency_analysis.json", 'w') as f:
            json.dump(efficiency_results, f, indent=2, default=str)
        
        # Save performance vs budget data
        plot_data = self.generate_performance_vs_budget_data()
        plot_data.to_csv(output_dir / "performance_vs_budget.csv", index=False)
        
        # Save relative improvements
        try:
            improvements = self.compute_relative_improvements()
            improvements.to_csv(output_dir / "relative_improvements.csv", index=False)
        except Exception as e:
            logger.warning(f"Could not compute relative improvements: {e}")
        
        # Save model comparisons
        model_types = summary['model_type'].unique()
        comparison_results = []
        
        for i, model1 in enumerate(model_types):
            for model2 in model_types[i+1:]:
                try:
                    comparisons = self.compare_models(model1, model2)
                    for comp in comparisons:
                        comparison_results.append({
                            'model1': comp.model1,
                            'model2': comp.model2,
                            'hf_budget': comp.hf_budget,
                            'improvement_percent': comp.improvement_percent,
                            'p_value': comp.p_value,
                            'is_significant': comp.is_significant,
                            'effect_size': comp.effect_size,
                            'ci_lower': comp.confidence_interval[0],
                            'ci_upper': comp.confidence_interval[1]
                        })
                except Exception as e:
                    logger.warning(f"Could not compare {model1} vs {model2}: {e}")
        
        if comparison_results:
            comparisons_df = pd.DataFrame(comparison_results)
            comparisons_df.to_csv(output_dir / "model_comparisons.csv", index=False)
        
        logger.info(f"Analysis results saved to {output_dir}")


def create_results_aggregator(results_dir: str) -> ResultsAggregator:
    """
    Factory function to create results aggregator.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        ResultsAggregator instance
    """
    return ResultsAggregator(results_dir)