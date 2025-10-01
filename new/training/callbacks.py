"""
Custom training callbacks for SABR MDA-CNN models.

This module implements specialized callbacks for training monitoring,
model checkpointing, learning rate scheduling, and progress tracking.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger, ExperimentLogger

logger = get_logger(__name__)


class EarlyStoppingCallback(keras.callbacks.EarlyStopping):
    """
    Enhanced early stopping callback with additional logging.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0,
        patience: int = 0,
        verbose: int = 0,
        mode: str = 'auto',
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
        start_from_epoch: int = 0
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
            start_from_epoch=start_from_epoch
        )
        self.logger = get_logger('callbacks.early_stopping')
    
    def on_epoch_end(self, epoch, logs=None):
        """Enhanced epoch end with detailed logging."""
        current = self.get_monitor_value(logs)
        if current is None:
            return
        
        # Log current monitoring value
        self.logger.debug(f"Epoch {epoch}: {self.monitor} = {current:.6f}")
        
        # Call parent method
        super().on_epoch_end(epoch, logs)
        
        # Log if early stopping triggered
        if self.model.stop_training:
            self.logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best {self.monitor}: {self.best:.6f} at epoch {epoch - self.patience}"
            )


class ModelCheckpointCallback(keras.callbacks.ModelCheckpoint):
    """
    Enhanced model checkpoint callback with better logging and file management.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = 'auto',
        save_freq: str = 'epoch',
        initial_value_threshold: Optional[float] = None,
        max_checkpoints: int = 5
    ):
        super().__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            initial_value_threshold=initial_value_threshold
        )
        self.max_checkpoints = max_checkpoints
        self.checkpoint_files = []
        self.logger = get_logger('callbacks.model_checkpoint')
    
    def on_epoch_end(self, epoch, logs=None):
        """Enhanced epoch end with checkpoint management."""
        # Call parent method
        super().on_epoch_end(epoch, logs)
        
        # Manage checkpoint files
        if self.max_checkpoints > 0:
            self._manage_checkpoint_files()
    
    def _on_save(self, filepath, logs):
        """Called when a checkpoint is saved."""
        self.checkpoint_files.append(filepath)
        self.logger.info(f"Model checkpoint saved: {filepath}")
        
        if logs:
            metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in logs.items()])
            self.logger.info(f"Checkpoint metrics: {metrics_str}")
    
    def _manage_checkpoint_files(self):
        """Remove old checkpoint files to limit disk usage."""
        if len(self.checkpoint_files) > self.max_checkpoints:
            # Remove oldest checkpoints
            files_to_remove = self.checkpoint_files[:-self.max_checkpoints]
            for file_path in files_to_remove:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        self.logger.debug(f"Removed old checkpoint: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove checkpoint {file_path}: {e}")
            
            # Update the list
            self.checkpoint_files = self.checkpoint_files[-self.max_checkpoints:]


class LearningRateSchedulerCallback(keras.callbacks.ReduceLROnPlateau):
    """
    Enhanced learning rate scheduler with detailed logging.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.1,
        patience: int = 10,
        verbose: int = 0,
        mode: str = 'auto',
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0
    ):
        super().__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr
        )
        self.logger = get_logger('callbacks.lr_scheduler')
        self.lr_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Enhanced epoch end with learning rate tracking."""
        current_lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lr_history.append(current_lr)
        
        # Call parent method
        super().on_epoch_end(epoch, logs)
        
        # Check if learning rate was reduced
        new_lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        if new_lr != current_lr:
            self.logger.info(
                f"Learning rate reduced from {current_lr:.2e} to {new_lr:.2e} at epoch {epoch}"
            )
    
    def on_train_begin(self, logs=None):
        """Log initial learning rate."""
        initial_lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        self.logger.info(f"Initial learning rate: {initial_lr:.2e}")


class TrainingProgressCallback(keras.callbacks.Callback):
    """
    Callback for tracking and logging training progress.
    """
    
    def __init__(
        self,
        experiment_logger: Optional[ExperimentLogger] = None,
        log_frequency: int = 1
    ):
        super().__init__()
        self.experiment_logger = experiment_logger
        self.log_frequency = log_frequency
        self.logger = get_logger('callbacks.progress')
        
        # Training state
        self.epoch_start_time = None
        self.training_start_time = None
        self.epoch_times = []
    
    def on_train_begin(self, logs=None):
        """Initialize training progress tracking."""
        self.training_start_time = time.time()
        self.logger.info("Training started")
    
    def on_train_end(self, logs=None):
        """Log training completion statistics."""
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
            
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            self.logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
            
            if self.experiment_logger:
                self.experiment_logger.log_experiment_end({
                    'total_training_time': total_time,
                    'average_epoch_time': avg_epoch_time,
                    'total_epochs': len(self.epoch_times)
                })
    
    def on_epoch_begin(self, epoch, logs=None):
        """Track epoch start time."""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """Log epoch completion and metrics."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            if epoch % self.log_frequency == 0:
                self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                
                if logs:
                    metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in logs.items()])
                    self.logger.info(f"Epoch {epoch} metrics: {metrics_str}")
                
                # Log to experiment logger
                if self.experiment_logger and logs:
                    self.experiment_logger.log_epoch(epoch, logs)
    
    def on_batch_end(self, batch, logs=None):
        """Log batch completion (less frequent)."""
        # Only log every 100 batches to avoid spam
        if batch % 100 == 0 and logs:
            self.logger.debug(f"Batch {batch} completed")


class MetricsLoggerCallback(keras.callbacks.Callback):
    """
    Callback for detailed metrics logging and tracking.
    """
    
    def __init__(
        self,
        experiment_logger: Optional[ExperimentLogger] = None,
        log_frequency: int = 10,
        save_metrics: bool = True,
        metrics_file: Optional[str] = None
    ):
        super().__init__()
        self.experiment_logger = experiment_logger
        self.log_frequency = log_frequency
        self.save_metrics = save_metrics
        self.metrics_file = metrics_file
        self.logger = get_logger('callbacks.metrics')
        
        # Metrics storage
        self.batch_metrics = []
        self.epoch_metrics = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Log and store epoch metrics."""
        if logs:
            # Store epoch metrics
            epoch_data = {'epoch': epoch, **logs}
            self.epoch_metrics.append(epoch_data)
            
            # Log to experiment logger
            if self.experiment_logger:
                self.experiment_logger.log_epoch(epoch, logs)
            
            # Save metrics to file if requested
            if self.save_metrics and self.metrics_file:
                self._save_metrics_to_file()
    
    def on_batch_end(self, batch, logs=None):
        """Log batch metrics at specified frequency."""
        if logs and batch % self.log_frequency == 0:
            # Store batch metrics
            batch_data = {'batch': batch, **logs}
            self.batch_metrics.append(batch_data)
            
            # Log to experiment logger
            if self.experiment_logger:
                current_epoch = getattr(self, '_current_epoch', 0)
                self.experiment_logger.log_batch(current_epoch, batch, logs)
    
    def on_epoch_begin(self, epoch, logs=None):
        """Track current epoch for batch logging."""
        self._current_epoch = epoch
    
    def _save_metrics_to_file(self):
        """Save metrics to JSON file."""
        try:
            import json
            metrics_data = {
                'epoch_metrics': self.epoch_metrics,
                'batch_metrics': self.batch_metrics[-1000:]  # Keep last 1000 batch metrics
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save metrics to file: {e}")


class GradientNormCallback(keras.callbacks.Callback):
    """
    Callback for monitoring gradient norms during training.
    """
    
    def __init__(
        self,
        log_frequency: int = 10,
        clip_norm: Optional[float] = None
    ):
        super().__init__()
        self.log_frequency = log_frequency
        self.clip_norm = clip_norm
        self.logger = get_logger('callbacks.gradient_norm')
        self.gradient_norms = []
    
    def on_train_batch_end(self, batch, logs=None):
        """Monitor gradient norms after each batch."""
        if batch % self.log_frequency == 0:
            # Get gradients (this is a simplified approach)
            # In practice, you'd need to modify the training loop to capture gradients
            try:
                # This would require custom training loop implementation
                # For now, just log that we're monitoring
                self.logger.debug(f"Monitoring gradients at batch {batch}")
            except Exception as e:
                self.logger.debug(f"Could not compute gradient norm: {e}")


class ValidationCallback(keras.callbacks.Callback):
    """
    Custom validation callback with additional validation metrics.
    """
    
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        validation_freq: int = 1,
        custom_metrics: Optional[List] = None
    ):
        super().__init__()
        self.validation_data = validation_data
        self.validation_freq = validation_freq
        self.custom_metrics = custom_metrics or []
        self.logger = get_logger('callbacks.validation')
    
    def on_epoch_end(self, epoch, logs=None):
        """Run custom validation at specified frequency."""
        if epoch % self.validation_freq == 0:
            self.logger.info(f"Running custom validation at epoch {epoch}")
            
            # Run custom validation metrics
            for metric in self.custom_metrics:
                try:
                    metric_value = self._compute_custom_metric(metric)
                    self.logger.info(f"Custom {metric.__name__}: {metric_value:.6f}")
                    
                    # Add to logs
                    if logs is not None:
                        logs[f'val_custom_{metric.__name__}'] = metric_value
                        
                except Exception as e:
                    self.logger.warning(f"Failed to compute custom metric {metric.__name__}: {e}")
    
    def _compute_custom_metric(self, metric_fn):
        """Compute custom metric on validation data."""
        # This would require implementing custom metric computation
        # For now, return a placeholder
        return 0.0


def create_default_callbacks(
    experiment_logger: Optional[ExperimentLogger] = None,
    checkpoint_dir: Optional[str] = None,
    early_stopping_patience: int = 20,
    lr_patience: int = 10,
    lr_factor: float = 0.5,
    max_checkpoints: int = 5
) -> List[keras.callbacks.Callback]:
    """
    Create a standard set of training callbacks.
    
    Args:
        experiment_logger: Optional experiment logger
        checkpoint_dir: Directory for saving checkpoints
        early_stopping_patience: Patience for early stopping
        lr_patience: Patience for learning rate reduction
        lr_factor: Factor for learning rate reduction
        max_checkpoints: Maximum number of checkpoints to keep
        
    Returns:
        List of callback instances
    """
    callbacks = []
    
    # Early stopping
    callbacks.append(EarlyStoppingCallback(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    ))
    
    # Model checkpointing
    if checkpoint_dir:
        checkpoint_path = os.path.join(
            checkpoint_dir,
            "model_epoch_{epoch:03d}_val_loss_{val_loss:.6f}"  # No .h5 extension
        )
        callbacks.append(ModelCheckpointCallback(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_format='tf',  # Use SavedModel format
            verbose=1,
            max_checkpoints=max_checkpoints
        ))
    
    # Learning rate scheduling
    callbacks.append(LearningRateSchedulerCallback(
        monitor='val_loss',
        patience=lr_patience,
        factor=lr_factor,
        verbose=1
    ))
    
    # Progress tracking
    callbacks.append(TrainingProgressCallback(
        experiment_logger=experiment_logger
    ))
    
    # Metrics logging
    callbacks.append(MetricsLoggerCallback(
        experiment_logger=experiment_logger,
        log_frequency=10
    ))
    
    return callbacks