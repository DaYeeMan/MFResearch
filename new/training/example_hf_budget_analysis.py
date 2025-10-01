"""
Example script demonstrating HF budget analysis.

This script shows how to use the experiment orchestrator to run
a simplified HF budget analysis with a small number of experiments.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ExperimentConfig, ConfigManager
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


def create_example_config() -> ExperimentConfig:
    """Create a simple configuration for demonstration."""
    config_manager = ConfigManager()
    config = config_manager.create_default_config("hf_budget_example")
    
    # Reduce data generation for faster execution
    config.data_gen_config.n_parameter_sets = 100  # Reduced from 1000
    config.data_gen_config.mc_paths = 10000       # Reduced from 100000
    
    # Reduce training epochs for faster execution
    config.training_config.epochs = 20            # Reduced from 200
    config.training_config.early_stopping_patience = 5
    
    return config


def create_simple_hyperparameter_space() -> HyperparameterSpace:
    """Create a simple hyperparameter space for demonstration."""
    return HyperparameterSpace(
        learning_rates=[3e-4, 1e-3],              # Only 2 options
        batch_sizes=[32, 64],                     # Only 2 options
        dropout_rates=[0.1, 0.2],                 # Only 2 options
        cnn_filters=[[32, 64], [64, 128]],        # Only 2 options
        mlp_hidden_dims=[[64, 64], [128, 64]]     # Only 2 options
    )


def run_simple_example():
    """Run a simple HF budget analysis example."""
    logger.info("Starting simple HF budget analysis example")
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create configuration
    config = create_example_config()
    
    # Create experiment orchestrator
    output_dir = "new/results/hf_budget_example"
    orchestrator = create_experiment_orchestrator(
        base_config=config,
        output_dir=output_dir,
        n_parallel_jobs=1,  # Sequential execution for simplicity
        random_seed=42
    )
    
    # Define experiment parameters
    hf_budgets = [50, 100, 200]                   # Only 3 budgets
    model_types = ["mda_cnn", "residual_mlp"]     # Only 2 models
    hyperparameter_space = create_simple_hyperparameter_space()
    n_hyperparameter_trials = 2                   # Only 2 trials
    n_random_seeds = 2                            # Only 2 seeds
    
    logger.info(f"Running experiments with:")
    logger.info(f"  HF budgets: {hf_budgets}")
    logger.info(f"  Model types: {model_types}")
    logger.info(f"  Hyperparameter trials: {n_hyperparameter_trials}")
    logger.info(f"  Random seeds: {n_random_seeds}")
    logger.info(f"  Total experiments: {len(hf_budgets) * len(model_types) * n_hyperparameter_trials * n_random_seeds}")
    
    # Run experiments
    try:
        results_df = orchestrator.run_hf_budget_analysis(
            hf_budgets=hf_budgets,
            model_types=model_types,
            hyperparameter_space=hyperparameter_space,
            n_hyperparameter_trials=n_hyperparameter_trials,
            n_random_seeds=n_random_seeds
        )
        
        logger.info(f"Experiments completed successfully!")
        logger.info(f"Results shape: {results_df.shape}")
        logger.info(f"Results saved to: {output_dir}")
        
        # Analyze results
        logger.info("Analyzing results...")
        aggregator = create_results_aggregator(output_dir)
        
        # Get performance summary
        summary = aggregator.get_performance_summary()
        print("\nPerformance Summary:")
        print(summary[['model_type', 'hf_budget', 'mean_mse', 'mean_mae', 'n_trials']])
        
        # Get model rankings
        rankings = aggregator.rank_models_by_budget()
        print("\nModel Rankings by HF Budget:")
        for budget, ranking in rankings.items():
            print(f"  HF Budget {budget}:")
            for i, (model, score) in enumerate(ranking, 1):
                print(f"    {i}. {model}: {score:.6f}")
        
        # Compare models if both are present
        if len(model_types) >= 2:
            comparisons = aggregator.compare_models(model_types[0], model_types[1])
            print(f"\nModel Comparison ({model_types[0]} vs {model_types[1]}):")
            for comp in comparisons:
                significance = "significant" if comp.is_significant else "not significant"
                print(f"  HF Budget {comp.hf_budget}: {comp.improvement_percent:.1f}% improvement, {significance}")
        
        # Save analysis results
        aggregator.save_analysis_results()
        logger.info("Analysis completed and saved!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


def run_analysis_only_example():
    """Example of running analysis on existing results."""
    logger.info("Running analysis-only example")
    
    results_dir = "new/results/hf_budget_example"
    
    # Check if results exist
    results_file = Path(results_dir) / "detailed_results.csv"
    if not results_file.exists():
        logger.error(f"No results found at {results_file}")
        logger.info("Please run the full example first with run_simple_example()")
        return
    
    # Create aggregator and analyze
    aggregator = create_results_aggregator(results_dir)
    
    # Print basic statistics
    print(f"\nLoaded {len(aggregator.detailed_results)} experiment results")
    print(f"Model types: {aggregator.detailed_results['model_type'].unique()}")
    print(f"HF budgets: {sorted(aggregator.detailed_results['hf_budget'].unique())}")
    
    # Generate and save analysis
    aggregator.save_analysis_results()
    logger.info("Analysis completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HF budget analysis example")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only run analysis on existing results"
    )
    
    args = parser.parse_args()
    
    if args.analysis_only:
        run_analysis_only_example()
    else:
        run_simple_example()