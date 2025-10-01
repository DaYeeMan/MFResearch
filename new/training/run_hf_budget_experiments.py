"""
Main script for running HF budget analysis experiments.

This script orchestrates the complete experimental pipeline:
1. Data generation for different HF budgets
2. Hyperparameter tuning for each model type
3. Model training and evaluation across budgets
4. Statistical analysis and result aggregation
5. Report generation
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ExperimentConfig, HyperparameterBounds
from utils.logging_utils import setup_logging, get_logger
from utils.reproducibility import set_random_seed
from training.experiment_orchestrator import (
    ExperimentOrchestrator, 
    HyperparameterSpace,
    create_experiment_orchestrator
)
from evaluation.results_aggregator import create_results_aggregator

# Set up logging
setup_logging()
logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run HF budget analysis experiments for SABR MDA-CNN"
    )
    
    # Experiment configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="new/configs/default_config.yaml",
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="new/results/hf_budget_analysis",
        help="Output directory for results"
    )
    
    # HF budget settings
    parser.add_argument(
        "--hf-budgets",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500, 1000],
        help="List of HF budget sizes to test"
    )
    
    # Model types to compare
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=["mda_cnn", "residual_mlp", "direct_mlp", "cnn_only"],
        help="List of model types to compare"
    )
    
    # Hyperparameter tuning settings
    parser.add_argument(
        "--hyperparameter-trials",
        type=int,
        default=10,
        help="Number of hyperparameter combinations to try per model"
    )
    
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=3,
        help="Number of random seeds for statistical robustness"
    )
    
    parser.add_argument(
        "--tuning-method",
        type=str,
        choices=["random", "grid", "bayesian"],
        default="random",
        help="Hyperparameter tuning method"
    )
    
    # Execution settings
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for experiments"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility"
    )
    
    # Execution modes
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip running experiments and only analyze existing results"
    )
    
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip analysis and only run experiments"
    )
    
    return parser.parse_args()


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from file."""
    from utils.config import ConfigManager
    
    config_manager = ConfigManager()
    
    if Path(config_path).exists():
        config = config_manager.load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Configuration file {config_path} not found, using default")
        config = config_manager.create_default_config("hf_budget_analysis")
    
    return config


def create_hyperparameter_space() -> HyperparameterSpace:
    """Create hyperparameter search space for experiments."""
    return HyperparameterSpace(
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
            [128, 128],
            [256, 128, 64]
        ]
    )


def run_experiments(
    config: ExperimentConfig,
    hf_budgets: List[int],
    model_types: List[str],
    hyperparameter_trials: int,
    random_seeds: int,
    output_dir: str,
    parallel_jobs: int,
    random_seed: int
) -> str:
    """
    Run the complete experimental pipeline.
    
    Returns:
        Path to results directory
    """
    logger.info("Starting HF budget analysis experiments")
    logger.info(f"HF budgets: {hf_budgets}")
    logger.info(f"Model types: {model_types}")
    logger.info(f"Hyperparameter trials: {hyperparameter_trials}")
    logger.info(f"Random seeds: {random_seeds}")
    
    # Set random seeds
    set_random_seed(random_seed)
    
    # Create experiment orchestrator
    orchestrator = create_experiment_orchestrator(
        base_config=config,
        output_dir=output_dir,
        n_parallel_jobs=parallel_jobs,
        random_seed=random_seed
    )
    
    # Create hyperparameter space
    hyperparameter_space = create_hyperparameter_space()
    
    # Run experiments
    results_df = orchestrator.run_hf_budget_analysis(
        hf_budgets=hf_budgets,
        model_types=model_types,
        hyperparameter_space=hyperparameter_space,
        n_hyperparameter_trials=hyperparameter_trials,
        n_random_seeds=random_seeds
    )
    
    logger.info(f"Experiments completed. Results saved to {output_dir}")
    logger.info(f"Total experiments run: {len(results_df)}")
    
    return output_dir


def analyze_results(results_dir: str) -> Dict[str, Any]:
    """
    Analyze experimental results and generate reports.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        Dictionary with analysis summary
    """
    logger.info("Starting results analysis")
    
    # Create results aggregator
    aggregator = create_results_aggregator(results_dir)
    
    # Generate comprehensive analysis
    aggregator.save_analysis_results()
    
    # Get summary statistics
    summary = aggregator.get_performance_summary()
    
    # Get model rankings
    rankings = aggregator.rank_models_by_budget()
    
    # Analyze efficiency for MDA-CNN
    try:
        mda_cnn_efficiency = aggregator.analyze_budget_efficiency('mda_cnn')
        logger.info(f"MDA-CNN efficiency analysis: R²={mda_cnn_efficiency['r_squared']:.3f}")
    except Exception as e:
        logger.warning(f"Could not analyze MDA-CNN efficiency: {e}")
        mda_cnn_efficiency = None
    
    # Compare MDA-CNN vs baselines
    comparison_summary = {}
    if 'mda_cnn' in summary['model_type'].values:
        for model_type in summary['model_type'].unique():
            if model_type != 'mda_cnn':
                try:
                    comparisons = aggregator.compare_models('mda_cnn', model_type)
                    significant_improvements = sum(1 for c in comparisons if c.is_significant and c.improvement_percent > 0)
                    comparison_summary[model_type] = {
                        'total_comparisons': len(comparisons),
                        'significant_improvements': significant_improvements,
                        'avg_improvement': np.mean([c.improvement_percent for c in comparisons])
                    }
                except Exception as e:
                    logger.warning(f"Could not compare MDA-CNN vs {model_type}: {e}")
    
    analysis_summary = {
        'total_experiments': len(aggregator.detailed_results),
        'model_types': list(summary['model_type'].unique()),
        'hf_budgets': list(summary['hf_budget'].unique()),
        'best_overall_performance': {
            'model': summary.loc[summary['mean_mse'].idxmin(), 'model_type'],
            'hf_budget': summary.loc[summary['mean_mse'].idxmin(), 'hf_budget'],
            'mse': summary['mean_mse'].min()
        },
        'model_rankings': rankings,
        'mda_cnn_efficiency': mda_cnn_efficiency,
        'comparison_summary': comparison_summary
    }
    
    # Save analysis summary
    summary_path = Path(results_dir) / "analysis" / "analysis_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(analysis_summary, f, indent=2, default=str)
    
    logger.info("Results analysis completed")
    logger.info(f"Analysis summary saved to {summary_path}")
    
    return analysis_summary


def print_summary_report(analysis_summary: Dict[str, Any]):
    """Print a summary report of the experimental results."""
    print("\n" + "="*80)
    print("HF BUDGET ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    print(f"\nExperiment Overview:")
    print(f"  Total experiments: {analysis_summary['total_experiments']}")
    print(f"  Model types tested: {', '.join(analysis_summary['model_types'])}")
    print(f"  HF budgets tested: {analysis_summary['hf_budgets']}")
    
    print(f"\nBest Overall Performance:")
    best = analysis_summary['best_overall_performance']
    print(f"  Model: {best['model']}")
    print(f"  HF Budget: {best['hf_budget']}")
    print(f"  MSE: {best['mse']:.6f}")
    
    print(f"\nModel Rankings by HF Budget:")
    for budget, ranking in analysis_summary['model_rankings'].items():
        print(f"  HF Budget {budget}:")
        for i, (model, score) in enumerate(ranking, 1):
            print(f"    {i}. {model}: {score:.6f}")
    
    if analysis_summary['mda_cnn_efficiency']:
        eff = analysis_summary['mda_cnn_efficiency']
        print(f"\nMDA-CNN Efficiency Analysis:")
        print(f"  Power law fit R²: {eff['r_squared']:.3f}")
        print(f"  Improvement per budget doubling: {eff['improvement_per_doubling']:.1f}%")
    
    if analysis_summary['comparison_summary']:
        print(f"\nMDA-CNN vs Baseline Comparisons:")
        for model, comp in analysis_summary['comparison_summary'].items():
            print(f"  vs {model}:")
            print(f"    Significant improvements: {comp['significant_improvements']}/{comp['total_comparisons']}")
            print(f"    Average improvement: {comp['avg_improvement']:.1f}%")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    logger.info("Starting HF budget analysis pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Load configuration
        config = load_experiment_config(args.config)
        
        # Run experiments if not skipped
        if not args.skip_experiments:
            results_dir = run_experiments(
                config=config,
                hf_budgets=args.hf_budgets,
                model_types=args.model_types,
                hyperparameter_trials=args.hyperparameter_trials,
                random_seeds=args.random_seeds,
                output_dir=args.output_dir,
                parallel_jobs=args.parallel_jobs,
                random_seed=args.random_seed
            )
        else:
            results_dir = args.output_dir
            logger.info(f"Skipping experiments, using existing results from {results_dir}")
        
        # Analyze results if not skipped
        if not args.skip_analysis:
            analysis_summary = analyze_results(results_dir)
            print_summary_report(analysis_summary)
        else:
            logger.info("Skipping analysis")
        
        logger.info("HF budget analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()