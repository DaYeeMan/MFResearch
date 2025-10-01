# Performance Analysis and Reporting System

This module implements comprehensive performance analysis and reporting for SABR volatility surface models as specified in Task 15. The system provides all required analysis components with automated report generation.

## Overview

The performance analysis system consists of three main components:

1. **PerformanceAnalyzer** - Core analysis engine
2. **ComprehensiveEvaluationPipeline** - Complete evaluation orchestrator  
3. **Example Scripts** - Demonstration and testing utilities

## Key Features Implemented

### ✅ Performance vs HF Budget Analysis Plots
- Static matplotlib plots with error bars and confidence intervals
- Interactive plotly visualizations with hover details
- Multiple metrics (MSE, MAE, training time, relative improvement)
- Power law efficiency analysis for budget optimization
- Optimal budget recommendations for target performance levels

### ✅ Residual Distribution Analysis Before/After ML Correction
- Histogram comparisons between baseline and ML models
- Statistical significance testing (t-tests, Wilcoxon tests)
- Box plots showing distribution characteristics
- Q-Q plots for normality assessment
- Improvement quantification with effect sizes

### ✅ Training Convergence Visualization and Analysis
- Training and validation loss convergence curves
- Convergence speed analysis (epochs to convergence)
- Learning rate effect analysis
- Model stability assessment
- Interactive convergence exploration

### ✅ Automated Report Generation with Key Metrics and Plots
- Comprehensive text reports with executive summary
- HTML reports with embedded visualizations
- Statistical analysis summaries
- Model comparison matrices
- Technical appendices with methodology details

### ✅ Comprehensive Evaluation Pipeline
- Complete end-to-end evaluation orchestration
- Configurable analysis components
- Execution logging and error handling
- Results aggregation and serialization
- Pipeline state management

## File Structure

```
new/evaluation/
├── performance_analyzer.py          # Core performance analysis engine
├── comprehensive_pipeline.py        # Complete evaluation pipeline
├── example_performance_analysis.py  # Performance analyzer examples
├── example_comprehensive_pipeline.py # Pipeline examples
├── test_task15_implementation.py    # Implementation tests
└── README_performance_analysis.md   # This documentation
```

## Quick Start

### Basic Performance Analysis

```python
from evaluation.performance_analyzer import create_performance_analyzer

# Create analyzer
analyzer = create_performance_analyzer("path/to/results")

# Run comprehensive analysis
results = analyzer.run_comprehensive_analysis()

# Results include:
# - Performance vs budget analysis
# - Residual distribution analysis  
# - Training convergence analysis
# - Model comparison analysis
# - Comprehensive report
```

### Complete Evaluation Pipeline

```python
from evaluation.comprehensive_pipeline import create_comprehensive_pipeline

# Create pipeline
pipeline = create_comprehensive_pipeline("path/to/results")

# Run complete evaluation
results = pipeline.run_complete_evaluation()

# Generates all analysis components plus:
# - Executive summary
# - Technical appendix
# - HTML reports
# - Summary metrics JSON
```

## Analysis Components

### 1. Performance vs HF Budget Analysis

**Plots Generated:**
- MSE vs HF Budget (log scale with error bars)
- MAE vs HF Budget (log scale with error bars)  
- Training Time vs HF Budget
- Relative Improvement vs HF Budget

**Analysis Provided:**
- Power law efficiency curves: `performance = a * budget^b`
- Optimal budget recommendations for target performance
- Statistical confidence intervals
- Model ranking by budget efficiency

**Key Metrics:**
- Best/worst performance by model
- Improvement ratios across budget range
- Budget efficiency exponents
- R-squared values for power law fits

### 2. Residual Distribution Analysis

**Visualizations:**
- Histogram overlays (baseline vs ML models)
- Box plots by model type
- Q-Q plots for normality testing
- Residual patterns by HF budget

**Statistical Tests:**
- Paired t-tests for mean differences
- Wilcoxon signed-rank tests (non-parametric)
- Effect size calculations (Cohen's d)
- Bootstrap confidence intervals

**Improvement Metrics:**
- Percentage improvement in RMSE/MAE
- Statistical significance (p-values)
- Distribution shape analysis (skewness, kurtosis)
- Regional performance breakdown

### 3. Training Convergence Analysis

**Convergence Metrics:**
- Average epochs to convergence by model
- Final validation loss distributions
- Training stability (loss variance)
- Learning rate sensitivity analysis

**Visualizations:**
- Training/validation loss curves with confidence bands
- Convergence speed box plots
- Final performance distributions
- Learning rate effect analysis

**Insights Generated:**
- Fastest converging model identification
- Most stable training configurations
- Convergence efficiency rankings
- Hyperparameter sensitivity analysis

### 4. Model Comparison Analysis

**Comparison Methods:**
- Pairwise statistical testing across all models
- Model ranking by HF budget
- Relative improvement calculations
- Statistical significance assessment

**Outputs:**
- Model comparison matrices (heatmaps)
- Best model identification by budget
- Significance testing results
- Effect size quantification

### 5. Automated Report Generation

**Report Sections:**
- Executive Summary with key findings
- Detailed analysis results
- Statistical test summaries
- Technical methodology appendix
- Conclusions and recommendations

**Output Formats:**
- Comprehensive text report
- HTML report with embedded plots
- JSON summary metrics
- Executive summary document

## Configuration Options

### PerformanceAnalysisConfig

```python
config = PerformanceAnalysisConfig(
    figure_size=(12, 8),           # Plot dimensions
    dpi=100,                       # Plot resolution
    save_plots=True,               # Save static plots
    plot_format='png',             # Plot file format
    interactive_plots=True,        # Generate interactive plots
    confidence_level=0.95          # Statistical confidence level
)
```

### PipelineConfig

```python
config = PipelineConfig(
    run_performance_analysis=True,  # Enable performance analysis
    run_residual_analysis=True,     # Enable residual analysis
    run_convergence_analysis=True,  # Enable convergence analysis
    run_model_comparison=True,      # Enable model comparison
    run_visualization=True,         # Enable visualization
    save_plots=True,               # Save generated plots
    generate_report=True,          # Generate comprehensive report
    create_summary=True,           # Create executive summary
    plot_formats=['png', 'pdf'],   # Output plot formats
    interactive_plots=True,        # Generate interactive plots
    include_detailed_analysis=True, # Include detailed sections
    include_statistical_tests=True  # Include statistical testing
)
```

## Example Usage

### Running Complete Analysis

```python
# Run complete pipeline example
python evaluation/example_comprehensive_pipeline.py

# Run performance analysis only
python evaluation/example_performance_analysis.py

# Test implementation
python evaluation/test_task15_implementation.py
```

### Custom Analysis Configuration

```python
from evaluation.performance_analyzer import PerformanceAnalyzer, PerformanceAnalysisConfig

# Custom configuration
config = PerformanceAnalysisConfig(
    save_plots=True,
    plot_format='pdf',
    interactive_plots=False,
    confidence_level=0.99
)

# Create analyzer with custom config
analyzer = PerformanceAnalyzer(
    results_dir="path/to/results",
    output_dir="path/to/output", 
    config=config
)

# Run specific analysis components
budget_analysis = analyzer.analyze_performance_vs_budget()
residual_analysis = analyzer.analyze_residual_distributions()
convergence_analysis = analyzer.analyze_training_convergence()
```

## Output Files Generated

### Analysis Results
- `performance_analysis_results.json` - Complete analysis results
- `performance_vs_budget.csv` - Performance data for plotting
- `model_comparisons.csv` - Statistical comparison results
- `summary_metrics.json` - Key performance metrics

### Reports
- `comprehensive_performance_report.txt` - Main text report
- `comprehensive_evaluation_report.html` - Interactive HTML report
- `executive_summary.txt` - Executive summary document
- `pipeline_execution_log.json` - Execution trace log

### Visualizations
- `performance_vs_budget.png/pdf` - Performance plots
- `residual_distributions.png/pdf` - Residual analysis plots
- `training_convergence.png/pdf` - Convergence analysis plots
- `model_comparison_matrix.png/pdf` - Comparison heatmaps
- `*_interactive.html` - Interactive plotly visualizations

## Integration with Existing System

The performance analysis system integrates seamlessly with existing components:

- **ResultsAggregator** - Loads and processes experiment results
- **SurfaceEvaluator** - Provides surface-specific metrics
- **BenchmarkComparator** - Handles model comparisons
- **SmilePlotter/SurfacePlotter** - Generates visualizations
- **StatisticalTester** - Performs significance testing

## Requirements Satisfied

This implementation fully satisfies all requirements from Task 15:

✅ **Create performance vs HF budget analysis plots**
- Multiple plot types with statistical analysis
- Interactive and static visualizations
- Efficiency analysis and optimization

✅ **Implement residual distribution analysis before/after ML correction**  
- Comprehensive statistical testing
- Distribution comparison visualizations
- Improvement quantification

✅ **Add training convergence visualization and analysis**
- Convergence curve analysis
- Speed and stability metrics
- Hyperparameter sensitivity

✅ **Create automated report generation with key metrics and plots**
- Multi-format report generation
- Executive summaries
- Technical appendices

✅ **Write comprehensive evaluation pipeline that generates all analysis**
- Complete end-to-end orchestration
- Configurable components
- Error handling and logging

## Testing

The implementation includes comprehensive testing:

```bash
# Run all tests
python evaluation/test_task15_implementation.py

# Expected output: All tests pass
# ✓ Performance analyzer import successful
# ✓ Comprehensive pipeline import successful  
# ✓ Example scripts import successful
# ✓ Basic functionality test successful
```

## Performance Considerations

- **Memory Efficient**: Processes results in chunks for large datasets
- **Configurable**: Can disable expensive components (interactive plots, detailed analysis)
- **Parallel Ready**: Designed for future parallel processing integration
- **Scalable**: Handles varying numbers of models and HF budgets

## Future Enhancements

Potential areas for extension:
- GPU acceleration for large-scale analysis
- Real-time analysis dashboard
- Advanced statistical modeling (Bayesian analysis)
- Integration with MLflow/Weights & Biases
- Automated hyperparameter optimization recommendations

## Conclusion

This performance analysis and reporting system provides a comprehensive solution for evaluating SABR volatility surface models. It delivers all required analysis components with professional-quality visualizations and reports, enabling thorough assessment of model performance across different HF budgets and configurations.

The modular design allows for easy extension and customization while maintaining robust error handling and comprehensive documentation. The system is ready for production use and provides the foundation for ongoing model evaluation and optimization.