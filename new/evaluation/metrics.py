"""
Comprehensive evaluation metrics for SABR volatility surface models.

This module provides surface-specific metrics, region-based analysis,
and statistical testing capabilities for model evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class RegionBounds:
    """Define moneyness regions for ATM, ITM, OTM analysis."""
    atm_lower: float = 0.95
    atm_upper: float = 1.05
    itm_call_upper: float = 0.95  # K/F < 0.95 for ITM calls
    otm_call_lower: float = 1.05  # K/F > 1.05 for OTM calls


@dataclass
class SurfaceMetrics:
    """Container for surface evaluation metrics."""
    rmse: float
    mae: float
    mape: float  # Mean Absolute Percentage Error
    max_error: float
    r_squared: float
    
    # Region-specific metrics
    atm_rmse: float
    itm_rmse: float
    otm_rmse: float
    
    atm_mae: float
    itm_mae: float
    otm_mae: float
    
    # Additional statistics
    mean_bias: float
    std_error: float
    n_points: int


class SurfaceEvaluator:
    """
    Comprehensive evaluator for volatility surface predictions.
    
    Provides surface-specific metrics, region-based analysis,
    and statistical significance testing.
    """
    
    def __init__(self, region_bounds: Optional[RegionBounds] = None):
        """
        Initialize evaluator with region definitions.
        
        Args:
            region_bounds: Custom region boundaries for ATM/ITM/OTM analysis
        """
        self.region_bounds = region_bounds or RegionBounds()
    
    def compute_surface_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        strikes: np.ndarray,
        forward_price: float,
        mask: Optional[np.ndarray] = None
    ) -> SurfaceMetrics:
        """
        Compute comprehensive surface evaluation metrics.
        
        Args:
            y_true: True volatility values (flattened or 2D)
            y_pred: Predicted volatility values (same shape as y_true)
            strikes: Strike prices corresponding to predictions
            forward_price: Forward price for moneyness calculation
            mask: Optional mask for valid data points
            
        Returns:
            SurfaceMetrics object with all computed metrics
        """
        # Flatten arrays if needed
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Handle strikes array - need to expand to match surface dimensions
        if strikes.ndim == 1 and y_true.ndim == 2:
            # Expand strikes to match surface grid
            strikes_grid = np.tile(strikes, y_true.shape[0])
        else:
            strikes_grid = strikes.flatten() if strikes.ndim > 1 else strikes
        
        # Ensure strikes array matches flattened surface size
        if len(strikes_grid) != len(y_true_flat):
            # If strikes is 1D and doesn't match, repeat it to match surface size
            n_repeats = len(y_true_flat) // len(strikes_grid)
            if n_repeats > 1:
                strikes_grid = np.tile(strikes_grid, n_repeats)
            # Truncate if still too long
            strikes_grid = strikes_grid[:len(y_true_flat)]
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.flatten()
            y_true_flat = y_true_flat[mask_flat]
            y_pred_flat = y_pred_flat[mask_flat]
            strikes_grid = strikes_grid[mask_flat]
        
        # Remove any NaN or infinite values
        valid_idx = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[valid_idx]
        y_pred_clean = y_pred_flat[valid_idx]
        strikes_clean = strikes_grid[valid_idx]
        
        if len(y_true_clean) == 0:
            raise ValueError("No valid data points for evaluation")
        
        # Compute overall metrics
        rmse = self._compute_rmse(y_true_clean, y_pred_clean)
        mae = self._compute_mae(y_true_clean, y_pred_clean)
        mape = self._compute_mape(y_true_clean, y_pred_clean)
        max_error = np.max(np.abs(y_true_clean - y_pred_clean))
        r_squared = self._compute_r_squared(y_true_clean, y_pred_clean)
        
        # Compute bias and error statistics
        errors = y_pred_clean - y_true_clean
        mean_bias = np.mean(errors)
        std_error = np.std(errors)
        
        # Compute region-specific metrics
        moneyness = strikes_clean / forward_price
        region_metrics = self._compute_region_metrics(
            y_true_clean, y_pred_clean, moneyness
        )
        
        return SurfaceMetrics(
            rmse=rmse,
            mae=mae,
            mape=mape,
            max_error=max_error,
            r_squared=r_squared,
            atm_rmse=region_metrics['atm_rmse'],
            itm_rmse=region_metrics['itm_rmse'],
            otm_rmse=region_metrics['otm_rmse'],
            atm_mae=region_metrics['atm_mae'],
            itm_mae=region_metrics['itm_mae'],
            otm_mae=region_metrics['otm_mae'],
            mean_bias=mean_bias,
            std_error=std_error,
            n_points=len(y_true_clean)
        )
    
    def _compute_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Square Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _compute_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    def _compute_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = np.abs(y_true) > 1e-8
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def _compute_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R-squared coefficient of determination."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def _compute_region_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        moneyness: np.ndarray
    ) -> Dict[str, float]:
        """Compute metrics for ATM, ITM, and OTM regions."""
        # Define region masks
        atm_mask = (
            (moneyness >= self.region_bounds.atm_lower) & 
            (moneyness <= self.region_bounds.atm_upper)
        )
        itm_mask = moneyness < self.region_bounds.itm_call_upper
        otm_mask = moneyness > self.region_bounds.otm_call_lower
        
        metrics = {}
        
        # ATM metrics
        if np.any(atm_mask):
            metrics['atm_rmse'] = self._compute_rmse(y_true[atm_mask], y_pred[atm_mask])
            metrics['atm_mae'] = self._compute_mae(y_true[atm_mask], y_pred[atm_mask])
        else:
            metrics['atm_rmse'] = np.nan
            metrics['atm_mae'] = np.nan
        
        # ITM metrics
        if np.any(itm_mask):
            metrics['itm_rmse'] = self._compute_rmse(y_true[itm_mask], y_pred[itm_mask])
            metrics['itm_mae'] = self._compute_mae(y_true[itm_mask], y_pred[itm_mask])
        else:
            metrics['itm_rmse'] = np.nan
            metrics['itm_mae'] = np.nan
        
        # OTM metrics
        if np.any(otm_mask):
            metrics['otm_rmse'] = self._compute_rmse(y_true[otm_mask], y_pred[otm_mask])
            metrics['otm_mae'] = self._compute_mae(y_true[otm_mask], y_pred[otm_mask])
        else:
            metrics['otm_rmse'] = np.nan
            metrics['otm_mae'] = np.nan
        
        return metrics


class StatisticalTester:
    """Statistical significance testing for model comparisons."""
    
    @staticmethod
    def paired_t_test(
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Union[float, bool]]:
        """
        Perform paired t-test to compare model errors.
        
        Args:
            errors_model1: Absolute errors from model 1
            errors_model2: Absolute errors from model 2
            alpha: Significance level
            
        Returns:
            Dictionary with test statistics and results
        """
        if len(errors_model1) != len(errors_model2):
            raise ValueError("Error arrays must have same length")
        
        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(errors_model1, errors_model2)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < alpha),
            'alpha': alpha,
            'mean_diff': float(np.mean(errors_model1 - errors_model2)),
            'std_diff': float(np.std(errors_model1 - errors_model2))
        }
    
    @staticmethod
    def wilcoxon_test(
        errors_model1: np.ndarray,
        errors_model2: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Union[float, bool]]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).
        
        Args:
            errors_model1: Absolute errors from model 1
            errors_model2: Absolute errors from model 2
            alpha: Significance level
            
        Returns:
            Dictionary with test statistics and results
        """
        if len(errors_model1) != len(errors_model2):
            raise ValueError("Error arrays must have same length")
        
        # Perform Wilcoxon signed-rank test
        try:
            statistic, p_value = stats.wilcoxon(errors_model1, errors_model2)
        except ValueError as e:
            # Handle case where all differences are zero
            return {
                'statistic': np.nan,
                'p_value': 1.0,
                'significant': False,
                'alpha': alpha,
                'error': str(e)
            }
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(p_value < alpha),
            'alpha': alpha
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        errors: np.ndarray,
        metric_func: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Args:
            errors: Error values
            metric_func: Function to compute metric (e.g., np.mean, np.std)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Dictionary with metric value and confidence interval
        """
        np.random.seed(42)  # For reproducibility
        
        bootstrap_metrics = []
        n_samples = len(errors)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            bootstrap_sample = np.random.choice(errors, size=n_samples, replace=True)
            bootstrap_metrics.append(metric_func(bootstrap_sample))
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        
        return {
            'metric_value': metric_func(errors),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'bootstrap_std': np.std(bootstrap_metrics)
        }


def compute_relative_improvement(
    baseline_metric: float,
    improved_metric: float,
    lower_is_better: bool = True
) -> float:
    """
    Compute relative improvement between two metrics.
    
    Args:
        baseline_metric: Baseline model metric value
        improved_metric: Improved model metric value
        lower_is_better: Whether lower values indicate better performance
        
    Returns:
        Relative improvement as percentage
    """
    if baseline_metric == 0:
        return np.inf if improved_metric != 0 else 0.0
    
    if lower_is_better:
        return ((baseline_metric - improved_metric) / baseline_metric) * 100
    else:
        return ((improved_metric - baseline_metric) / baseline_metric) * 100