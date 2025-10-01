# HF Budget Analysis Experiment Orchestrator

This module provides a comprehensive framework for analyzing the performance of SABR MDA-CNN models across different high-fidelity (HF) data budget sizes. The orchestrator automates the entire experimental pipeline from data generation to statistical analysis.

## Overview

The experiment orchestrator addresses the key research question: **How does model performance scale with the amount of expensive high-fidelity Monte Carlo data?**

### Key Features

- **Multi-Budget Analysis**: Test performance across different HF budget sizes (e.g., 50, 100, 200, 500, 1000 points)
- **Multi-Architecture Comparison**: Compare MDA-CNN against baseline models (Direct MLP, Residual MLP, CNN-only)
- **Automated Hyperparameter Tuning**: Optimize hyperparameters for each model and budget combination
- **Statistical Analysis**: Perform rigorous statistical testing and significance analysis
- **Reproducibility**: Ensure consistent results across runs with proper seed management
- **Parallel Execution**: Support for parallel experiment execution to reduce runtime

## Components

### 1. ExperimentOrchestrator (`experiment_orchestrator.py`)

The main orchestrator class that manages the complete experimental pipeline.

```python
from training.experiment_orchestrator import create_experiment_orchestrator

orchestrator = create_experiment_orchestrator(
    base_config=config,
    output_dir="results/hf_budget_analysis",
    n_parallel_jobs=4,
    random_seed=42
)

results_df = orchestrator.run_hf_budget_analysis(
    hf_budgets=[50, 100, 200, 500],
    model_types=["mda_cnn", "residual_mlp", "direct_mlp"],
    n_hyperparameter_trials=10,
    n_random_seeds=3
)
```

### 2. HyperparameterTuning (`hyperparameter_tuning.py`)

Automated hyperparameter optimization with multiple strategies:

- **Random Search**: Fast and effective for most cases
- **Grid Search**: Exhaustive search over discrete parameter grids
- **Bayesian Optimization**: Advanced optimization using Optuna (optional)

```python
from training.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(base_config, data_loaders)
best_params = tuner.tune_hyperparameters(
    method="random",
    n_trials=20
)
```

### 3. ResultsAggregator (`results_aggregator.py`)

Comprehensive analysis and aggregation of experimental results:

```python
from evaluation.results_aggregator import create_results_aggregator

aggregator = create_results_aggregator("results/hf_budget_analysis")
summary = aggregator.get_performance_summary()
rankings = aggregator.rank_models_by_budget()
comparisons = aggregator.compare_models("mda_cnn", "residual_mlp")
```

## Usage Examples

### Quick Start Example

```python
# Run a simple HF budget analysis
python new/training/example_hf_budget_analysis.py
```

### Full Analysis Pipeline

```python
# Run comprehensive analysis with custom parameters
python new/training/run_hf_budget_experiments.py \
    --hf-budgets 50 100 200 500 1000 \
    --model-types mda_cnn residual_mlp direct_mlp cnn_only \
    --hyperparameter-trials 15 \
    --random-seeds 5 \
    --parallel-jobs 4
```

### Analysis Only (on existing results)

```python
# Analyze existing results without running new experiments
python new/training/run_hf_budget_experiments.py \
    --skip-experiments \
    --output-dir results/existing_experiment
```

## Configuration

### Experiment Configuration

The orchestrator uses the standard `ExperimentConfig` class with additional parameters:

```yaml
# config.yaml
name: "hf_budget_analysis"
data_gen_config:
  n_parameter_sets: 1000
  mc_paths: 100000
  hf_budget: 200  # Will be overridden for each budget test
training_config:
  epochs: 200
  batch_size: 64
  learning_rate: 3e-4
model_config:
  patch_size: [9, 9]
  cnn_filters: [32, 64, 128]
  mlp_hidden_dims: [64, 64]
```

### Hyperparameter Space

Define the search space for hyperparameter optimization:

```python
from training.experiment_orchestrator import HyperparameterSpace

hyperparameter_space = HyperparameterSpace(
    learning_rates=[1e-4, 3e-4, 1e-3, 3e-3],
    batch_sizes=[32, 64, 128],
    dropout_rates=[0.1, 0.2, 0.3],
    cnn_filters=[
        [32, 64],
        [32, 64, 128],
        [64, 128, 256]
    ],
    mlp_hidden_dims=[
        [64, 64],
        [128, 64],
        [128, 128, 64]
    ]
)
```

## Output Structure

The orchestrator generates a comprehensive set of outputs:

```
results/hf_budget_analysis/
├── detailed_results.csv           # Raw experiment results
├── summary_statistics.csv         # Aggregated performance statistics
├── statistical_analysis.json      # Statistical test results
├── performance_vs_budget.csv      # Data for plotting performance curves
├── logs/                          # Experiment logs
├── analysis/                      # Detailed analysis results
│   ├── performance_summary.csv
│   ├── model_rankings.json
│   ├── efficiency_analysis.json
│   ├── relative_improvements.csv
│   └── model_comparisons.csv
└── exp_XXXX/                     # Individual experiment directories
    ├── config.json
    ├── models/
    ├── checkpoints/
    └── logs/
```

## Key Metrics and Analysis

### Performance Metrics

- **MSE/RMSE**: Root Mean Square Error for volatility predictions
- **MAE**: Mean Absolute Error
- **Region-specific metrics**: ATM, ITM, OTM performance analysis
- **Training time**: Computational efficiency metrics

### Statistical Analysis

- **Paired t-tests**: Compare model performance with statistical significance
- **Effect size**: Cohen's d for practical significance
- **Confidence intervals**: Bootstrap confidence intervals for metrics
- **Power law fitting**: Analyze performance scaling with budget size

### Efficiency Analysis

- **Budget efficiency**: How performance scales with HF data amount
- **Optimal budget**: Minimum budget needed for target performance
- **Improvement curves**: Performance vs budget relationships
- **Cost-benefit analysis**: Training time vs performance trade-offs

## Advanced Features

### Parallel Execution

The orchestrator supports parallel execution of experiments:

```python
orchestrator = ExperimentOrchestrator(
    base_config=config,
    output_dir="results",
    n_parallel_jobs=8  # Run 8 experiments in parallel
)
```

### Custom Objective Functions

For hyperparameter tuning, you can define custom objective functions:

```python
def custom_objective(hyperparameters):
    # Custom logic for evaluating hyperparameters
    # Return metric to minimize
    return validation_loss

tuner = HyperparameterTuner(config, data_loaders)
tuner.objective_function = custom_objective
```

### Experiment Resumption

The orchestrator can resume interrupted experiments by checking existing results and skipping completed experiments.

## Best Practices

### 1. Start Small

Begin with a small-scale experiment to validate the setup:

```python
# Small-scale test
hf_budgets = [50, 100]
model_types = ["mda_cnn", "residual_mlp"]
n_hyperparameter_trials = 3
n_random_seeds = 2
```

### 2. Use Appropriate Resources

- **CPU-bound**: Data generation and some model training
- **GPU-accelerated**: Model training (if available)
- **Memory requirements**: Scale with batch size and model complexity

### 3. Monitor Progress

- Check logs regularly for errors or performance issues
- Use smaller epoch counts for hyperparameter tuning
- Validate results on a subset before full-scale experiments

### 4. Statistical Robustness

- Use multiple random seeds (3-5) for statistical robustness
- Ensure sufficient hyperparameter trials (10-20)
- Consider the multiple testing problem when interpreting p-values

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model complexity
2. **Slow Execution**: Use parallel jobs or reduce experiment scope
3. **Convergence Issues**: Check learning rates and model architecture
4. **Data Generation Errors**: Verify SABR parameter ranges and grid configuration

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check individual experiment logs in the `exp_XXXX/logs/` directories.

## Testing

Run the test suite to verify functionality:

```bash
python new/training/test_experiment_orchestrator.py
```

## Dependencies

- **Core**: TensorFlow, NumPy, Pandas, SciPy
- **Optional**: Optuna (for Bayesian optimization)
- **Visualization**: Matplotlib, Seaborn (for plotting results)

## Performance Considerations

### Computational Requirements

- **Data Generation**: CPU-intensive, benefits from parallel processing
- **Model Training**: GPU-accelerated when available
- **Memory Usage**: Scales with batch size and surface resolution

### Optimization Tips

1. **Use appropriate HF budgets**: Start with smaller budgets for initial exploration
2. **Optimize hyperparameter search**: Use random search for initial exploration, Bayesian for refinement
3. **Leverage parallel execution**: Use multiple cores/GPUs when available
4. **Cache data**: Reuse generated data across experiments when possible

## Future Extensions

- **Multi-objective optimization**: Optimize for both accuracy and computational efficiency
- **Adaptive budget allocation**: Dynamically adjust HF budget based on performance
- **Online learning**: Update models as new HF data becomes available
- **Ensemble methods**: Combine multiple models for improved performance