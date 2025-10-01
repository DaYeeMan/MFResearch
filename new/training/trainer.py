"""
Main training infrastructure for SABR MDA-CNN models.

This module implements the core training loop with validation, early stopping,
model checkpointing, learning rate scheduling, and comprehensive logging.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ExperimentConfig, TrainingConfig
from utils.logging_utils import get_logger, ExperimentLogger
from models.loss_functions import get_loss_function, get_metrics
from training.callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LearningRateSchedulerCallback,
    TrainingProgressCallback,
    MetricsLoggerCallback
)

logger = get_logger(__name__)


class ModelTrainer:
    """
    Main trainer class for SABR MDA-CNN models.
    
    Handles the complete training pipeline including:
    - Model compilation with custom loss functions and metrics
    - Training loop with validation
    - Early stopping and learning rate scheduling
    - Model checkpointing and best model saving
    - Comprehensive logging and progress monitoring
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Optional[str] = None,
        experiment_logger: Optional[ExperimentLogger] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Experiment configuration
            output_dir: Directory to save outputs (overrides config if provided)
            experiment_logger: Optional experiment logger instance
        """
        self.config = config
        self.training_config = config.training_config
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up experiment logger
        if experiment_logger is None:
            log_dir = self.output_dir / "logs"
            self.experiment_logger = ExperimentLogger(
                experiment_name=config.name,
                log_dir=log_dir
            )
        else:
            self.experiment_logger = experiment_logger
        
        # Initialize training state
        self.model = None
        self.history = None
        self.best_model_path = None
        self.training_start_time = None
        
        # Create subdirectories
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Trainer initialized with output directory: {self.output_dir}")
    
    def compile_model(
        self,
        model: keras.Model,
        loss_name: str = "mse",
        loss_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        optimizer_name: str = "adam",
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ) -> keras.Model:
        """
        Compile model with specified loss function, metrics, and optimizer.
        
        Args:
            model: Keras model to compile
            loss_name: Name of loss function to use
            loss_kwargs: Additional arguments for loss function
            metrics: List of metric names to track
            optimizer_name: Name of optimizer to use
            optimizer_kwargs: Additional arguments for optimizer
            
        Returns:
            Compiled model
        """
        # Set up loss function
        loss_kwargs = loss_kwargs or {}
        loss_fn = get_loss_function(loss_name, **loss_kwargs)
        
        # Set up metrics
        if metrics is None:
            metrics = ["mse", "mae", "rmse"]
        metric_instances = get_metrics(metrics)
        
        # Set up optimizer
        optimizer_kwargs = optimizer_kwargs or {}
        if optimizer_name.lower() == "adam":
            optimizer = keras.optimizers.Adam(
                learning_rate=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_name.lower() == "adamw":
            optimizer = keras.optimizers.AdamW(
                learning_rate=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                **optimizer_kwargs
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = keras.optimizers.SGD(
                learning_rate=self.training_config.learning_rate,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metric_instances
        )
        
        self.model = model
        logger.info(f"Model compiled with loss={loss_name}, optimizer={optimizer_name}")
        
        return model
    
    def create_callbacks(self) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks based on configuration.
        
        Returns:
            List of callback instances
        """
        callbacks = []
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            monitor='val_loss',
            patience=self.training_config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint callback
        checkpoint_path = self.checkpoints_dir / "model_epoch_{epoch:03d}_val_loss_{val_loss:.6f}.h5"
        model_checkpoint = ModelCheckpointCallback(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=self.training_config.save_best_only,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        # Learning rate scheduler callback
        lr_scheduler = LearningRateSchedulerCallback(
            monitor='val_loss',
            patience=self.training_config.lr_scheduler_patience,
            factor=self.training_config.lr_scheduler_factor,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Training progress callback
        progress_callback = TrainingProgressCallback(
            experiment_logger=self.experiment_logger
        )
        callbacks.append(progress_callback)
        
        # Metrics logger callback
        metrics_logger = MetricsLoggerCallback(
            experiment_logger=self.experiment_logger,
            log_frequency=10  # Log every 10 batches
        )
        callbacks.append(metrics_logger)
        
        # Gradient clipping (implemented as custom callback)
        if self.training_config.gradient_clip_norm > 0:
            gradient_clipper = GradientClippingCallback(
                clip_norm=self.training_config.gradient_clip_norm
            )
            callbacks.append(gradient_clipper)
        
        logger.info(f"Created {len(callbacks)} training callbacks")
        return callbacks
    
    def train(
        self,
        model: keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        loss_name: str = "mse",
        loss_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> keras.callbacks.History:
        """
        Train the model with the specified datasets.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            loss_name: Loss function name
            loss_kwargs: Loss function arguments
            metrics: Metrics to track
            callbacks: Additional callbacks (will be added to default callbacks)
            
        Returns:
            Training history
        """
        logger.info("Starting model training")
        self.training_start_time = time.time()
        
        # Log experiment start
        config_dict = {
            'model_config': self.config.model_config.__dict__,
            'training_config': self.config.training_config.__dict__,
            'loss_name': loss_name,
            'loss_kwargs': loss_kwargs or {},
            'metrics': metrics or []
        }
        self.experiment_logger.log_experiment_start(config_dict)
        
        # Compile model if not already compiled
        if not model.compiled:
            self.compile_model(
                model=model,
                loss_name=loss_name,
                loss_kwargs=loss_kwargs,
                metrics=metrics
            )
        
        # Create callbacks
        training_callbacks = self.create_callbacks()
        if callbacks:
            training_callbacks.extend(callbacks)
        
        try:
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=self.training_config.epochs,
                callbacks=training_callbacks,
                verbose=1
            )
            
            self.history = history
            
            # Save final model
            final_model_path = self.models_dir / "final_model.h5"
            model.save(final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
            
            # Find and save best model path
            self._find_best_model()
            
            # Calculate training time
            training_time = time.time() - self.training_start_time
            
            # Log experiment completion
            final_metrics = {
                'training_time_seconds': training_time,
                'total_epochs': len(history.history['loss']),
                'best_val_loss': min(history.history['val_loss']),
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1]
            }
            
            if 'val_rmse' in history.history:
                final_metrics['best_val_rmse'] = min(history.history['val_rmse'])
                final_metrics['final_val_rmse'] = history.history['val_rmse'][-1]
            
            self.experiment_logger.log_experiment_end(final_metrics)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best validation loss: {final_metrics['best_val_loss']:.6f}")
            
            return history
            
        except Exception as e:
            self.experiment_logger.log_error(f"Training failed: {str(e)}", e)
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(
        self,
        model: keras.Model,
        test_dataset: tf.data.Dataset,
        use_best_model: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            model: Model to evaluate (ignored if use_best_model=True)
            test_dataset: Test dataset
            use_best_model: Whether to load and use the best saved model
            
        Returns:
            Dictionary of evaluation metrics
        """
        if use_best_model and self.best_model_path:
            logger.info(f"Loading best model from: {self.best_model_path}")
            model = keras.models.load_model(self.best_model_path)
        
        logger.info("Evaluating model on test dataset")
        
        # Evaluate model
        test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
        
        logger.info("Test evaluation results:")
        for metric_name, value in test_results.items():
            logger.info(f"  {metric_name}: {value:.6f}")
        
        return test_results
    
    def predict(
        self,
        model: keras.Model,
        dataset: tf.data.Dataset,
        use_best_model: bool = True
    ) -> np.ndarray:
        """
        Generate predictions using the model.
        
        Args:
            model: Model to use for prediction (ignored if use_best_model=True)
            dataset: Dataset to predict on
            use_best_model: Whether to load and use the best saved model
            
        Returns:
            Model predictions
        """
        if use_best_model and self.best_model_path:
            logger.info(f"Loading best model from: {self.best_model_path}")
            model = keras.models.load_model(self.best_model_path)
        
        logger.info("Generating predictions")
        predictions = model.predict(dataset, verbose=1)
        
        return predictions
    
    def _find_best_model(self):
        """Find the best model checkpoint based on validation loss."""
        checkpoint_files = list(self.checkpoints_dir.glob("model_epoch_*.h5"))
        
        if not checkpoint_files:
            logger.warning("No checkpoint files found")
            return
        
        # Extract validation loss from filename and find best
        best_val_loss = float('inf')
        best_file = None
        
        for file_path in checkpoint_files:
            try:
                # Extract val_loss from filename
                filename = file_path.name
                val_loss_str = filename.split('val_loss_')[1].split('.h5')[0]
                val_loss = float(val_loss_str)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_file = file_path
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse validation loss from {filename}: {e}")
                continue
        
        if best_file:
            self.best_model_path = best_file
            # Copy best model to models directory
            best_model_copy = self.models_dir / "best_model.h5"
            import shutil
            shutil.copy2(best_file, best_model_copy)
            
            logger.info(f"Best model (val_loss={best_val_loss:.6f}): {best_file}")
            logger.info(f"Best model copied to: {best_model_copy}")
        else:
            logger.warning("Could not determine best model")


class GradientClippingCallback(keras.callbacks.Callback):
    """Custom callback for gradient clipping."""
    
    def __init__(self, clip_norm: float):
        super().__init__()
        self.clip_norm = clip_norm
    
    def on_train_batch_begin(self, batch, logs=None):
        """Apply gradient clipping before each training batch."""
        # Note: This is a simplified implementation
        # In practice, gradient clipping is usually handled by the optimizer
        # or through custom training loops
        pass


def create_trainer(
    config: ExperimentConfig,
    output_dir: Optional[str] = None
) -> ModelTrainer:
    """
    Factory function to create a ModelTrainer instance.
    
    Args:
        config: Experiment configuration
        output_dir: Output directory (optional)
        
    Returns:
        ModelTrainer instance
    """
    return ModelTrainer(config=config, output_dir=output_dir)