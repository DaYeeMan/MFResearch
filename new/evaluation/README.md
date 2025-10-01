# SABR Volatility Surface Evaluation Module

This module provides comprehensive evaluation capabilities for SABR volatility surface models, including surface-specific metrics, statistical testing, and benchmark comparison functionality.

## Features

### Core Metrics (`metrics.py`)
- **Surface-specific evaluation metrics**: RMSE, MAE, MAPE, R-squared, max error
- **Region-specific analysis**: ATM, ITM, OTM performance breakdown
- **Statistical testing**: Paired t-tests, Wilcoxon tests, bootstrap confidence intervals
- **Bias and error analysis**: Mean bias, standard deviation of errors

### Comprehensive Evaluation (`surface_evaluator.py`)
- **Multi-model comparison** with statistical significance testing
- **Multi-budget analysis** across different HF data budgets
- **Automated reporting** with summary statistics and visualizations
- **Results persistence** in CSV and JSON formats

### Benchmark Comparison (`benchmark_comparison.py`)
- **Relative improvement calculations** against baseline models
- **Statistical significance testing** for model comparisons
- **Cross-budget benchmarking** to analyze performance scaling
- **Automated benchmark reports** with key performance indicators

## Quick Start

### Basic Usage

```python
from evaluation import SurfaceEvaluator, ModelPrediction

# Create evaluator
evaluator = SurfaceEvaluator()

# Compute metrics for a single model
metrics = evaluator.compute_surface_metrics(
    y_true=true_volatilities,
    y_pred=predicted_volatilities,
    strikes=strike_prices,
    forward_price=100.0
)

print(f"RMSE: {metrics.rmse:.6f}")
print(f"ATM RMSE: {metrics.atm_rmse:.6f}")
print(f"ITM RMSE: {metrics.itm_rmse:.6f}")
print(f"OTM RMSE: {metrics.otm_rmse:.6f}")
```

### Comprehensive Model Comparison

```python
from evaluation import ComprehensiveEvaluator, ModelPrediction

# Create model predictions
predictions = [
    ModelPrediction(
        predictions=model1_output,
        model_name="Baseline_MLP",
        hf_budget=100,
        strikes=strikes,
        maturities=maturities,
        forward_price=100.0
    ),
    ModelPrediction(
        predictions=model2_output,
        model_name="MDA_CNN",
        hf_budget=100,
        strikes=strikes,
        maturities=maturities,
        forward_price=100.0
    )
]

# Compare models with statistical testing
evaluator = ComprehensiveEvaluator(output_dir="results")
results = evaluator.compare_models(y_true, predictions)

# Generate summary report
results_df = evaluator.evaluate_across_budgets(
    {100: y_true}, {100: predictions}
)
report = evaluator.generate_summary_report(results_df)
print(report)
```

### Benchmark Against Baseline

```python
from evaluation import BenchmarkComparator

# Compare against baseline
comparator = BenchmarkComparator(output_dir="benchmarks")
benchmark_result = comparator.compare_against_baseline(
    y_true=true_volatilities,
    model_prediction=mda_cnn_prediction,
    baseline_prediction=baseline_prediction
)

print(f"RMSE Improvement: {benchmark_result.rmse_improvement:+.1f}%")
print(f"Significantly Better: {benchmark_result.is_significantly_better}")
```

### Multi-Budget Analysis

```python
# Analyze performance across HF budgets
budgets = [50, 100, 200, 500]
y_true_dict = {budget: generate_test_data(budget) for budget in budgets}
predictions_dict = {budget: generate_predictions(budget) for budget in budgets}

# Run comprehensive evaluation
results_df = evaluator.evaluate_across_budgets(y_true_dict, predictions_dict)

# Generate benchmark comparison
benchmark_df = comparator.benchmark_across_budgets(
    y_true_dict, predictions_dict, baseline_name="Baseline_MLP"
)

print("Performance vs HF Budget:")
print(results_df.groupby(['model_name', 'hf_budget'])['rmse'].mean().unstack())
```

## Key Classes and Functions

### SurfaceMetrics
Container for evaluation metrics with the following attributes:
- `rmse`, `mae`, `mape`: Overall error metrics
- `atm_rmse`, `itm_rmse`, `otm_rmse`: Region-specific RMSE
- `atm_mae`, `itm_mae`, `otm_mae`: Region-specific MAE
- `r_squared`: Coefficient of determination
- `mean_bias`, `std_error`: Bias and error statistics
- `max_error`: Maximum absolute error
- `n_points`: Number of evaluation points

### ModelPrediction
Container for model predictions with metadata:
- `predictions`: Model output array
- `model_name`: Identifier for the model
- `hf_budget`: Number of HF data points used
- `strikes`, `maturities`: Grid information
- `forward_price`: Forward price for moneyness calculation
- `metadata`: Optional additional information

### RegionBounds
Configurable boundaries for ATM/ITM/OTM analysis:
- `atm_lower`, `atm_upper`: ATM moneyness bounds (default: 0.95-1.05)
- `itm_call_upper`: ITM call boundary (default: 0.95)
- `otm_call_lower`: OTM call boundary (default: 1.05)

## Statistical Testing

The module provides several statistical tests for model comparison:

### Paired T-Test
Tests whether two models have significantly different error distributions:
```python
from evaluation import StatisticalTester

tester = StatisticalTester()
result = tester.paired_t_test(errors_model1, errors_model2)
print(f"Significant difference: {result['significant']}")
print(f"P-value: {result['p_value']:.4f}")
```

### Wilcoxon Signed-Rank Test
Non-parametric alternative to t-test:
```python
result = tester.wilcoxon_test(errors_model1, errors_model2)
```

### Bootstrap Confidence Intervals
Compute confidence intervals for any metric:
```python
ci_result = tester.bootstrap_confidence_interval(
    errors, np.mean, n_bootstrap=1000, confidence_level=0.95
)
print(f"Mean: {ci_result['metric_value']:.6f}")
print(f"95% CI: [{ci_result['ci_lower']:.6f}, {ci_result['ci_upper']:.6f}]")
```

## Output Files

The evaluation system generates several types of output files:

### CSV Files
- `multi_budget_results.csv`: Detailed results across HF budgets
- `benchmark_results.csv`: Benchmark comparison results

### JSON Files
- `evaluation_results.json`: Complete evaluation results with metadata
- Model comparison results with statistical test outcomes

### Text Reports
- `evaluation_summary.txt`: Human-readable summary report
- `benchmark_report.txt`: Benchmark comparison report

## Example Output

### Evaluation Summary
```
SABR Volatility Surface Model Evaluation Report
==================================================

Overall Statistics:
Total evaluations: 12
Models evaluated: 3
HF budgets tested: [50, 100, 200, 500]

Best Performing Models by HF Budget (RMSE):
Budget 50: MDA_CNN (RMSE: 0.012345)
Budget 100: MDA_CNN (RMSE: 0.010234)
Budget 200: MDA_CNN (RMSE: 0.008567)
Budget 500: MDA_CNN (RMSE: 0.007123)

Performance Summary by Model:
MDA_CNN:
  RMSE: 0.009567 ± 0.002134
  MAE:  0.007234 ± 0.001567
  R²:   0.892345 ± 0.045678
```

### Benchmark Report
```
SABR Model Benchmark Comparison Report
=============================================

Baseline Model: Baseline_MLP

Overall Improvement Statistics:
MDA_CNN:
  RMSE Improvement: 65.4% ± 8.2%
  MAE Improvement:  62.1% ± 7.9%
  Significantly Better: 100.0% of cases

Residual_MLP:
  RMSE Improvement: 29.2% ± 5.1%
  MAE Improvement:  26.8% ± 4.7%
  Significantly Better: 87.5% of cases
```

## Testing

Run the comprehensive test suite:
```bash
cd new
python -m pytest evaluation/test_evaluation_metrics.py -v
```

The tests cover:
- Basic metric calculations with perfect, noisy, and biased predictions
- Region-specific metric computation
- Statistical significance testing
- Multi-model comparison workflows
- Benchmark comparison functionality
- File I/O and report generation

## Requirements

The evaluation module requires:
- `numpy` for numerical computations
- `pandas` for data manipulation and analysis
- `scipy` for statistical testing
- `pytest` for running tests (development only)

## Integration with Training Pipeline

The evaluation module integrates seamlessly with the training pipeline:

```python
# After training models
from evaluation import ComprehensiveEvaluator
from training import load_test_data

# Load test data
test_data = load_test_data()
y_true = test_data['true_surfaces']

# Create predictions for each model
predictions = []
for model_name, model in trained_models.items():
    pred = model.predict(test_data['inputs'])
    predictions.append(ModelPrediction(
        predictions=pred,
        model_name=model_name,
        hf_budget=test_data['hf_budget'],
        strikes=test_data['strikes'],
        maturities=test_data['maturities'],
        forward_price=test_data['forward_price']
    ))

# Comprehensive evaluation
evaluator = ComprehensiveEvaluator(output_dir="final_evaluation")
results = evaluator.compare_models(y_true, predictions)
results_df = evaluator.evaluate_across_budgets({100: y_true}, {100: predictions})

# Generate final report
report = evaluator.generate_summary_report(results_df, "final_report.txt")
```

This evaluation system provides all the tools needed to comprehensively assess SABR volatility surface model performance according to the requirements specified in the project design.