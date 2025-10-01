"""
Main training script for SABR MDA-CNN models.

This script demonstrates how to use the training infrastructure to train
models with various configurations and monitoring capabilities.
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ConfigManager, ExperimentConfig
from utils.logging_utils import setup_logging, ExperimentLogger
from utils.reproducibility import set_random_seeds
from training.trainer import ModelTrainer
from training.training_config import (
    TrainingSetup, OptimizerConfig, LossConfig, 
    SchedulerConfig, TrainingStrategy
)
from models.mda_cnn import MDACNN
from models.baseline_models import create_baseline_model
from preprocessing.data_loader import create_data_loaders

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SABR MDA-CNN model')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='new/configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['mda_cnn', 'baseline_mlp', 'residual_mlp', 'cnn_only'],
        default='mda_cnn',
        help='Type of model to train'
    )
    parser.add_argument(
        '--loss-function',
        type=str,
        choices=['mse', 'mae', 'weighted_mse', 'huber', 'adaptive_weighted_mse'],
        default='mse',
        help='Loss function to use'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device to use (-1 for CPU)'
    )
    
    return parser.parse_args()


def setup_gpu(gpu_id: int):
    """Set up GPU configuration."""
    if gpu_id >= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and gpu_id < len(gpus):
            try:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                # Set visible GPU
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
                logger.info(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
            except RuntimeError as e:
                logger.warning(f"GPU setup failed: {e}")
        else:
            logger.warning(f"GPU {gpu_id} not available, using CPU")
    else:
        # Force CPU usage
        tf.config.experimental.set_visible_devices([], 'GPU')
        logger.info("Using CPU")


def create_model(
    model_type: str,
    config: ExperimentConfig
) -> keras.Model:
    """
    Create model based on type and configuration.
    
    Args:
        model_type: Type of model to create
        config: Experiment configuration
        
    Returns:
        Created model instance
    """
    model_config = config.model_config
    
    if model_type == 'mda_cnn':
        model = MDACNN(
            patch_size=tuple(model_config.patch_size),
            cnn_filters=model_config.cnn_filters,
            cnn_kernel_size=model_config.cnn_kernel_size,
            mlp_hidden_dims=model_config.mlp_hidden_dims,
            fusion_dims=model_config.fusion_dims,
            dropout_rate=model_config.dropout_rate,
            activation=model_config.activation
        )
    elif model_type in ['baseline_mlp', 'residual_mlp', 'cnn_only']:
        model = create_baseline_model(
            model_type=model_type,
            input_dim=8,  # Number of point features
            hidden_dims=model_config.mlp_hidden_dims,
            dropout_rate=model_config.dropout_rate,
            activation=model_config.activation
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type} model")
    return model


def setup_training_components(
    config: ExperimentConfig,
    loss_function: str
) -> Dict[str, Any]:
    """
    Set up training components (optimizer, loss, scheduler).
    
    Args:
        config: Experiment configuration
        loss_function: Loss function name
        
    Returns:
        Dictionary of training components
    """
    training_setup = TrainingSetup()
    
    # Create optimizer configuration
    optimizer_config = OptimizerConfig(
        name="adam",
        learning_rate=config.training_config.learning_rate,
        weight_decay=config.training_config.weight_decay,
        clipnorm=config.training_config.gradient_clip_norm
    )
    
    # Create loss configuration
    loss_config = LossConfig(name=loss_function)
    
    # Create scheduler configuration
    scheduler_config = SchedulerConfig(
        name="reduce_on_plateau",
        patience=config.training_config.lr_scheduler_patience,
        factor=config.training_config.lr_scheduler_factor
    )
    
    # Create components
    optimizer = training_setup.create_optimizer(optimizer_config)
    loss_fn = training_setup.create_loss_function(loss_config)
    scheduler = training_setup.create_lr_scheduler(scheduler_config, optimizer)
    
    return {
        'optimizer': optimizer,
        'loss_function': loss_fn,
        'scheduler': scheduler,
        'loss_name': loss_function
    }


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level, console_output=True)
    
    logger.info("Starting SABR MDA-CNN training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set up GPU
    setup_gpu(args.gpu)
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Set random seeds for reproducibility
        set_random_seeds(config.data_gen_config.random_seed)
        
        # Set up output directory
        output_dir = args.output_dir or config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up experiment logger
        experiment_logger = ExperimentLogger(
            experiment_name=config.name,
            log_dir=output_path / "logs"
        )
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            config=config,
            batch_size=config.training_config.batch_size
        )
        
        # Create model
        logger.info(f"Creating {args.model_type} model...")
        model = create_model(args.model_type, config)
        
        # Set up training components
        training_components = setup_training_components(config, args.loss_function)
        
        # Create trainer
        trainer = ModelTrainer(
            config=config,
            output_dir=output_dir,
            experiment_logger=experiment_logger
        )
        
        # Compile model
        trainer.compile_model(
            model=model,
            loss_name=training_components['loss_name'],
            metrics=['mse', 'mae', 'rmse']
        )
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            model.load_weights(args.resume_from)
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            model=model,
            train_dataset=train_loader,
            validation_dataset=val_loader,
            loss_name=args.loss_function
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(
            model=model,
            test_dataset=test_loader,
            use_best_model=True
        )
        
        # Save final results
        results_file = output_path / "training_results.json"
        import json
        
        results_data = {
            'config': config.__dict__,
            'training_history': {k: [float(v) for v in history.history[k]] 
                               for k in history.history.keys()},
            'test_results': {k: float(v) for k, v in test_results.items()},
            'model_type': args.model_type,
            'loss_function': args.loss_function
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Best model saved to: {trainer.best_model_path}")
        
        # Print final metrics
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        print(f"Model type: {args.model_type}")
        print(f"Loss function: {args.loss_function}")
        print(f"Total epochs: {len(history.history['loss'])}")
        print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
        print(f"Final test loss: {test_results.get('loss', 'N/A')}")
        if 'rmse' in test_results:
            print(f"Final test RMSE: {test_results['rmse']:.6f}")
        print(f"Output directory: {output_path}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()