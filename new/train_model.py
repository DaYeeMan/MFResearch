#!/usr/bin/env python3
"""
Main training script for SABR volatility surface MDA-CNN models.

This script provides a command-line interface for training MDA-CNN and baseline
models with comprehensive experiment management and hyperparameter tuning.

Usage:
    python train_model.py --data-dir data/experiment1
    python train_model.py --config configs/training_config.yaml --model mda_cnn
    python train_model.py --hf-budget-analysis --budgets 50 100 200 500
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config import ConfigManager, ExperimentConfig
from utils.logging_utils import setup_logging, get_logger
from utils.reproducibility import set_random_seed
from training.trainer import ModelTrainer
from training.experiment_orchestrator import ExperimentOrchestrator, HyperparameterSpace
from models.mda_cnn import create_mda_cnn_model
from models.baseline_models import create_baseline_model
from preprocessing.data_loader import create_data_loaders

logger = get_logger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train SABR volatility surface models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train MDA-CNN with default settings
  python train_model.py --data-dir data/experiment1

  # Train specific model type
  python train_model.py --data-dir data/experiment1 --model mda_cnn

  # Run HF budget analysis
  python train_model.py --data-dir data/experiment1 --hf-budget-analysis

  # Custom training configuration
  python train_model.py --config configs/custom_training.yaml

  # Hyperparameter tuning
  python train_model.py --data-dir data/experiment1 --tune-hyperparameters

  # Train multiple models for comparison
  python train_model.py --data-dir data/experiment1 --models mda_cnn residual_mlp direct_mlp
        """
    )
    
    # Data and configuration
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        required=True,
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results (default: results/timestamp)'
    )
    
    # Model selection
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['mda_cnn', 'residual_mlp', 'direct_mlp', 'cnn_only'],
        default='mda_cnn',
        help='Model type to train (default: mda_cnn)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['mda_cnn', 'residual_mlp', 'direct_mlp', 'cnn_only'],
        help='Multiple models to train for comparison'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--hf-budget',
        type=int,
        help='HF budget for training data'
    )
    
    # Experiment types
    parser.add_argument(
        '--hf-budget-analysis',
        action='store_true',
        help='Run comprehensive HF budget analysis'
    )
    
    parser.add_argument(
        '--budgets',
        nargs='+',
        type=int,
        default=[50, 100, 200, 500],
        help='HF budgets to test (default: 50 100 200 500)'
    )
    
    parser.add_argument(
        '--tune-hyperparameters',
        action='store_true',
        help='Run hyperparameter tuning'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of hyperparameter tuning trials (default: 20)'
    )
    
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=3,
        help='Number of random seeds for statistical robustness (default: 3)'
    )
    
    # Training options
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint'
    )
    
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )
    
    parser.add_argument(
        '--save-best-only',
        action='store_true',
        help='Save only the best model'
    )
    
    # Execution options
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel training for multiple experiments'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    
    parser.add_argument(
        '--gpu',
        type=str,
        help='GPU device to use (e.g., "0" or "0,1")'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit without training'
    )
    
    return parser


def setup_gpu(gpu_str: Optional[str]):
    """Setup GPU configuration."""
    if gpu_str is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        logger.info(f"Set CUDA_VISIBLE_DEVICES to {gpu_str}")
    
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.info("No GPUs found, using CPU")
    except ImportError:
        logger.warning("TensorFlow not available, cannot configure GPU")


def override_config_from_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Override configuration with command-line arguments."""
    
    # Training parameters
    if args.epochs is not None:
        config.training_config.epochs = args.epochs
    
    if args.batch_size is not None:
        config.training_config.batch_size = args.batch_size
    
    if args.learning_rate is not None:
        config.training_config.learning_rate = args.learning_rate
    
    if args.hf_budget is not None:
        config.data_gen_config.hf_budget = args.hf_budget
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    # Early stopping
    if args.early_stopping:
        config.training_config.early_stopping = True
        config.training_config.early_stopping_patience = args.patience
    
    return config


def print_training_summary(config: ExperimentConfig, args: argparse.Namespace):
    """Print training configuration summary."""
    print("\n" + "="*60)
    print("SABR MODEL TRAINING CONFIGURATION")
    print("="*60)
    
    print(f"\nExperiment: {config.name}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {config.output_dir}")
    
    if args.hf_budget_analysis:
        print(f"\nHF Budget Analysis:")
        print(f"  Budgets: {args.budgets}")
        print(f"  Models: {args.models or [args.model]}")
        print(f"  Trials per config: {args.n_trials}")
        print(f"  Random seeds: {args.n_seeds}")
    else:
        print(f"\nSingle Model Training:")
        print(f"  Model: {args.model}")
        print(f"  HF Budget: {config.data_gen_config.hf_budget}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.training_config.epochs}")
    print(f"  Batch Size: {config.training_config.batch_size}")
    print(f"  Learning Rate: {config.training_config.learning_rate}")
    print(f"  Early Stopping: {config.training_config.early_stopping}")
    
    if args.tune_hyperparameters:
        print(f"\nHyperparameter Tuning:")
        print(f"  Trials: {args.n_trials}")
        print(f"  Parallel Jobs: {args.n_jobs}")
    
    print("="*60)


def train_single_model(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """Train a single model."""
    logger.info(f"Training single model: {args.model}")
    
    # Create data loaders
    logger.info(f"Loading data from {args.data_dir}")
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=config.training_config.batch_size,
        validation_split=config.data_gen_config.validation_split,
        test_split=config.data_gen_config.test_split,
        random_seed=config.data_gen_config.random_seed
    )
    
    # Create model
    logger.info(f"Creating {args.model} model")
    if args.model == 'mda_cnn':
        model = create_mda_cnn_model(
            patch_size=tuple(config.model_config.patch_size),
            cnn_filters=tuple(config.model_config.cnn_filters),
            mlp_hidden_dims=tuple(config.model_config.mlp_hidden_dims),
            fusion_hidden_dims=tuple(config.model_config.fusion_dims),
            dropout_rate=config.model_config.dropout_rate,
            activation=config.model_config.activation
        )
    else:
        model = create_baseline_model(
            model_type=args.model,
            dropout_rate=config.model_config.dropout_rate,
            activation=config.model_config.activation
        )
    
    # Create trainer
    trainer = ModelTrainer(config, config.output_dir)
    
    # Train model
    logger.info("Starting model training")
    start_time = time.time()
    
    history = trainer.train(
        model=model,
        train_dataset=data_loaders['train'],
        validation_dataset=data_loaders['val']
    )
    
    training_time = time.time() - start_time
    
    # Evaluate model
    logger.info("Evaluating trained model")
    test_metrics = trainer.evaluate(
        model=model,
        test_dataset=data_loaders['test'],
        use_best_model=True
    )
    
    return {
        'model_type': args.model,
        'training_time': training_time,
        'history': history.history if hasattr(history, 'history') else history,
        'test_metrics': test_metrics,
        'best_model_path': trainer.best_model_path
    }


def run_hf_budget_analysis(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """Run comprehensive HF budget analysis."""
    logger.info("Starting HF budget analysis")
    
    # Create experiment orchestrator
    orchestrator = ExperimentOrchestrator(
        base_config=config,
        output_dir=config.output_dir,
        n_parallel_jobs=args.n_jobs,
        random_seed=args.seed
    )
    
    # Determine models to test
    models_to_test = args.models if args.models else [args.model]
    
    # Create hyperparameter space
    hyperparameter_space = HyperparameterSpace()
    
    # Run analysis
    results_df = orchestrator.run_hf_budget_analysis(
        hf_budgets=args.budgets,
        model_types=models_to_test,
        hyperparameter_space=hyperparameter_space,
        n_hyperparameter_trials=args.n_trials,
        n_random_seeds=args.n_seeds
    )
    
    return {
        'results_dataframe': results_df,
        'output_directory': config.output_dir,
        'models_tested': models_to_test,
        'budgets_tested': args.budgets
    }


def main():
    """Main training function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging()
    
    logger.info("Starting SABR model training")
    
    try:
        # Set up GPU
        setup_gpu(args.gpu)
        
        # Set random seed
        set_random_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
        
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_manager = ConfigManager()
        
        if Path(args.config).exists():
            config = config_manager.load_config(args.config)
        else:
            logger.warning(f"Configuration file {args.config} not found, using default")
            config = config_manager.create_default_config("training")
        
        # Override with command-line arguments
        config = override_config_from_args(config, args)
        
        # Set output directory if not specified
        if args.output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            config.output_dir = f"results/training_{timestamp}"
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Print configuration summary
        print_training_summary(config, args)
        
        # Dry run check
        if args.dry_run:
            print("\nDry run completed. No training performed.")
            return
        
        # Verify data directory exists
        if not Path(args.data_dir).exists():
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
        # Run training
        start_time = time.time()
        
        if args.hf_budget_analysis:
            results = run_hf_budget_analysis(config, args)
            experiment_type = "HF Budget Analysis"
        else:
            results = train_single_model(config, args)
            experiment_type = "Single Model Training"
        
        total_time = time.time() - start_time
        
        # Success message
        print(f"\n" + "="*60)
        print(f"{experiment_type.upper()} COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Output Directory: {config.output_dir}")
        
        if args.hf_budget_analysis:
            print(f"Models Tested: {results['models_tested']}")
            print(f"HF Budgets: {results['budgets_tested']}")
            print(f"Results Shape: {results['results_dataframe'].shape}")
        else:
            print(f"Model Type: {results['model_type']}")
            print(f"Training Time: {results['training_time']:.2f} seconds")
            print(f"Test Metrics: {results['test_metrics']}")
        
        print(f"\nNext Steps:")
        print(f"  1. Evaluate results: python evaluate_model.py --results-dir {config.output_dir}")
        print(f"  2. Generate visualizations: python evaluate_model.py --results-dir {config.output_dir} --visualize")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\nTraining cancelled by user.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nError: {e}")
        print("Check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()