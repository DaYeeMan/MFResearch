"""
Example usage of the performance analysis system for SABR volatility surface models.

This script demonstrates how to use the comprehensive performance analyzer
to generate all required analysis and reports.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.performance_analyzer import PerformanceAnalyzer, create_performance_analyzer
from utils.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def create_synthetic_results_data(results_dir: str):
    """Create synthetic experiment results for demonstration."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic detailed results
    np.random.seed(42)
    
    # Define experiment parameters
    model_types = ['direct_mlp', 'residual_mlp', 'mda_cnn']
    hf_budgets = [50, 100, 200, 500]
    n_trials_per_config = 5
    
    # Generate synthetic results
    results_data = []
    experiment_id = 0
    
    for model_type in model_types:
        for hf_budget in hf_budgets:
            for trial in range(n_trials_per_config):
                # Simulate performance that improves with budget and varies by model
                base_mse = {
                    'direct_mlp': 0.01,
                    'residual_mlp': 0.007,
                    'mda_cnn': 0.005
                }[model_type]
                
                # MSE improves with budget (power law relationship)
                budget_factor = (hf_budget / 100) ** (-0.3)
                mse = base_mse * budget_factor * (1 + np.random.normal(0, 0.1))
                mse = max(mse, 0.0001)  # Ensure positive
                
                mae = np.sqrt(mse) * (0.8 + np.random.normal(0, 0.1))
                mae = max(mae, 0.001)  # Ensure positive
                
                # Training time increases with model complexity and budget
                complexity_factor = {
                    'direct_mlp': 1.0,
                    'residual_mlp': 1.5,
                    'mda_cnn': 2.5
                }[model_type]
                
                training_time = complexity_factor * (hf_budget / 50) * (30 + np.random.normal(0, 5))
                training_time = max(training_time, 10)  # Minimum 10 seconds
                
                # Add some hyperparameters
                learning_rate = np.random.choice([1e-4, 3e-4, 1e-3])
                batch_size = np.random.choice([32, 64, 128])
                dropout_rate = np.random.choice([0.1, 0.2, 0.3])
                
                results_data.append({
                    'experiment_id': f'exp_{experiment_id:04d}',
                    'model_type': model_type,
                    'hf_budget': hf_budget,
                    'trial_id': trial,
                    'mse': mse,
                    'mae': mae,
                    'loss': mse,  # Assuming loss is same as MSE
                    'training_time': training_time,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'dropout_rate': dropout_rate
                })
                
                experiment_id += 1
    
    # Save detailed results
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_path / "detailed_results.csv", index=False)
    
    # Create some synthetic training logs
    logs_dir = results_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    for i, row in results_df.iterrows():
        exp_dir = results_path / row['experiment_id']
        exp_dir.mkdir(exist_ok=True)
        
        # Create synthetic training history
        n_epochs = np.random.randint(20, 100)
        initial_loss = row['mse'] * 10  # Start with higher loss
        
        # Simulate convergence
        epochs = np.arange(1, n_epochs + 1)
        decay_rate = np.random.uniform(0.05, 0.15)
        noise_level = 0.1
        
        train_losses = initial_loss * np.exp(-decay_rate * epochs) * (1 + np.random.normal(0, noise_level, n_epochs))
        val_losses = train_losses * (1.1 + np.random.normal(0, 0.05, n_epochs))  # Validation slightly higher
        
        # Ensure losses are positive and decreasing on average
        train_losses = np.maximum(train_losses, row['mse'] * 0.5)
        val_losses = np.maximum(val_losses, row['mse'] * 0.8)
        
        training_log = {
            'experiment_id': row['experiment_id'],
            'model_type': row['model_type'],
            'hf_budget': row['hf_budget'],
            'config': {
                'learning_rate': row['learning_rate'],
                'batch_size': row['batch_size'],
                'dropout_rate': row['dropout_rate'],
                'epochs': n_epochs
            },
            'history': {
                'loss': train_losses.tolist(),
                'val_loss': val_losses.tolist()
            },
            'final_metrics': {
                'mse': row['mse'],
                'mae': row['mae'],
                'training_time': row['training_time']
            }
        }
        
        # Save training log
        import json
        with open(exp_dir / "training_log.json", 'w') as f:
            json.dump(training_log, f, indent=2)
    
    logger.info(f"Created synthetic results data in {results_dir}")
    logger.info(f"Generated {len(results_df)} experiment results")
    logger.info(f"Model types: {results_df['model_type'].unique()}")
    logger.info(f"HF budgets: {sorted(results_df['hf_budget'].unique())}")


def run_comprehensive_performance_analysis():
    """Run comprehensive performance analysis on synthetic data."""
    logger.info("Starting comprehensive performance analysis example")
    
    # Create synthetic data
    results_dir = "new/results/performance_analysis_example"
    create_synthetic_results_data(results_dir)
    
    # Create performance analyzer
    analyzer = create_performance_analyzer(
        results_dir=results_dir,
        output_dir=f"{results_dir}/analysis"
    )
    
    # Run comprehensive analysis
    logger.info("Running comprehensive analysis pipeline...")
    analysis_results = analyzer.run_comprehensive_analysis()
    
    # Print summary of results
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Budget analysis summary
    if 'budget_analysis' in analysis_results:
        budget_results = analysis_results['budget_analysis']
        print("\n1. Performance vs HF Budget Analysis:")
        
        if 'summary_statistics' in budget_results:
            stats = budget_results['summary_statistics']
            for model, model_stats in stats.items():
                best_mse = model_stats['best_mse']
                improvement_ratio = model_stats['mse_improvement_ratio']
                print(f"   • {model}: Best MSE = {best_mse:.6f}, Improvement ratio = {improvement_ratio:.1f}x")
        
        if 'plots' in budget_results:
            plots = budget_results['plots']
            print(f"   • Generated {len(plots)} performance plots")
    
    # Residual analysis summary
    if 'residual_analysis' in analysis_results:
        residual_results = analysis_results['residual_analysis']
        print("\n2. Residual Distribution Analysis:")
        
        if 'improvement_statistics' in residual_results:
            stats = residual_results['improvement_statistics']
            if 'improvement_percent' in stats:
                improvement = stats['improvement_percent']
                significance = "Yes" if stats.get('is_significant', False) else "No"
                print(f"   • ML models show {improvement:.1f}% improvement over baseline")
                print(f"   • Statistically significant: {significance}")
        
        if 'plots' in residual_results:
            plots = residual_results['plots']
            print(f"   • Generated {len(plots)} residual analysis plots")
    
    # Convergence analysis summary
    if 'convergence_analysis' in analysis_results:
        convergence_results = analysis_results['convergence_analysis']
        print("\n3. Training Convergence Analysis:")
        
        if 'convergence_statistics' in convergence_results:
            stats = convergence_results['convergence_statistics']
            if 'mean_convergence_epoch' in stats:
                avg_epochs = stats['mean_convergence_epoch']
                print(f"   • Average convergence epoch: {avg_epochs:.1f}")
            
            if 'model_type_stats' in stats:
                model_stats = stats['model_type_stats']
                for model, model_data in model_stats.items():
                    if 'mean_convergence_epoch' in model_data:
                        epochs = model_data['mean_convergence_epoch']
                        print(f"   • {model}: {epochs:.1f} epochs to convergence")
        
        if 'plots' in convergence_results:
            plots = convergence_results['plots']
            print(f"   • Generated {len(plots)} convergence plots")
    
    # Model comparison summary
    if 'comparison_analysis' in analysis_results:
        comparison_results = analysis_results['comparison_analysis']
        print("\n4. Model Comparison Analysis:")
        
        if 'best_models_by_budget' in comparison_results:
            best_models = comparison_results['best_models_by_budget']
            print("   • Best models by HF budget:")
            for budget, model in best_models.items():
                print(f"     - Budget {budget}: {model}")
        
        if 'pairwise_comparisons' in comparison_results:
            comparisons = comparison_results['pairwise_comparisons']
            print(f"   • Performed {len(comparisons)} pairwise model comparisons")
    
    # Report generation
    if 'report' in analysis_results:
        print("\n5. Comprehensive Report:")
        print("   • Generated comprehensive performance analysis report")
        print(f"   • Report saved to: {results_dir}/analysis/comprehensive_performance_report.txt")
    
    print(f"\nAll analysis results saved to: {results_dir}/analysis/")
    print("="*60)
    
    return analysis_results


def run_individual_analysis_examples():
    """Run individual analysis components separately."""
    logger.info("Running individual analysis examples")
    
    results_dir = "new/results/performance_analysis_example"
    
    # Ensure synthetic data exists
    if not Path(results_dir).exists():
        create_synthetic_results_data(results_dir)
    
    # Create analyzer
    analyzer = create_performance_analyzer(results_dir)
    
    print("\n" + "="*50)
    print("INDIVIDUAL ANALYSIS EXAMPLES")
    print("="*50)
    
    # 1. Performance vs Budget Analysis
    print("\n1. Performance vs Budget Analysis:")
    budget_analysis = analyzer.analyze_performance_vs_budget()
    
    if 'efficiency_analysis' in budget_analysis:
        print("   Efficiency Analysis Results:")
        for model, efficiency in budget_analysis['efficiency_analysis'].items():
            if 'power_law_b' in efficiency:
                exponent = efficiency['power_law_b']
                r_squared = efficiency.get('r_squared', 0)
                print(f"   • {model}: Power law exponent = {exponent:.3f} (R² = {r_squared:.3f})")
    
    # 2. Residual Distribution Analysis
    print("\n2. Residual Distribution Analysis:")
    residual_analysis = analyzer.analyze_residual_distributions()
    
    if 'baseline_summary' in residual_analysis and 'ml_summary' in residual_analysis:
        baseline_summary = residual_analysis['baseline_summary']
        ml_summary = residual_analysis['ml_summary']
        
        if baseline_summary and ml_summary:
            baseline_rmse = baseline_summary.get('mean_rmse', 0)
            ml_rmse = ml_summary.get('mean_rmse', 0)
            
            if baseline_rmse > 0 and ml_rmse > 0:
                improvement = ((baseline_rmse - ml_rmse) / baseline_rmse) * 100
                print(f"   • Baseline mean RMSE: {baseline_rmse:.6f}")
                print(f"   • ML models mean RMSE: {ml_rmse:.6f}")
                print(f"   • Improvement: {improvement:.1f}%")
    
    # 3. Training Convergence Analysis
    print("\n3. Training Convergence Analysis:")
    convergence_analysis = analyzer.analyze_training_convergence()
    
    if 'convergence_statistics' in convergence_analysis:
        stats = convergence_analysis['convergence_statistics']
        total_experiments = stats.get('total_experiments', 0)
        print(f"   • Analyzed {total_experiments} training experiments")
        
        if 'mean_convergence_epoch' in stats:
            avg_epochs = stats['mean_convergence_epoch']
            std_epochs = stats.get('std_convergence_epoch', 0)
            print(f"   • Average convergence: {avg_epochs:.1f} ± {std_epochs:.1f} epochs")
    
    # 4. Model Comparison Analysis
    print("\n4. Model Comparison Analysis:")
    comparison_analysis = analyzer.analyze_model_comparisons()
    
    if 'rankings' in comparison_analysis:
        rankings = comparison_analysis['rankings']
        print("   • Model rankings by HF budget:")
        for budget, ranking in rankings.items():
            if ranking:
                best_model, best_score = ranking[0]
                print(f"     - Budget {budget}: {best_model} (MSE: {best_score:.6f})")
    
    print("\n" + "="*50)


def demonstrate_custom_analysis():
    """Demonstrate custom analysis configurations."""
    logger.info("Demonstrating custom analysis configurations")
    
    results_dir = "new/results/performance_analysis_example"
    
    # Create analyzer with custom configuration
    from evaluation.performance_analyzer import PerformanceAnalysisConfig
    
    custom_config = PerformanceAnalysisConfig(
        figure_size=(10, 6),
        dpi=150,
        save_plots=True,
        plot_format='pdf',  # Save as PDF instead of PNG
        interactive_plots=True,
        confidence_level=0.99  # Higher confidence level
    )
    
    analyzer = PerformanceAnalyzer(
        results_dir=results_dir,
        output_dir=f"{results_dir}/custom_analysis",
        config=custom_config
    )
    
    print("\n" + "="*40)
    print("CUSTOM ANALYSIS CONFIGURATION")
    print("="*40)
    
    # Run budget analysis with custom settings
    budget_analysis = analyzer.analyze_performance_vs_budget()
    
    print("Custom analysis completed with:")
    print(f"• Figure size: {custom_config.figure_size}")
    print(f"• DPI: {custom_config.dpi}")
    print(f"• Plot format: {custom_config.plot_format}")
    print(f"• Confidence level: {custom_config.confidence_level}")
    
    if 'plots' in budget_analysis:
        plots = budget_analysis['plots']
        print(f"• Generated plots: {list(plots.keys())}")
    
    print("="*40)


def main():
    """Run all performance analysis examples."""
    print("SABR Volatility Surface Performance Analysis Examples")
    print("=" * 60)
    
    try:
        # 1. Run comprehensive analysis
        analysis_results = run_comprehensive_performance_analysis()
        
        # 2. Run individual analysis examples
        run_individual_analysis_examples()
        
        # 3. Demonstrate custom configurations
        demonstrate_custom_analysis()
        
        print("\n" + "=" * 60)
        print("All performance analysis examples completed successfully!")
        print("Check the output directories for generated plots and reports.")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Performance analysis example failed: {e}")
        raise


if __name__ == "__main__":
    main()