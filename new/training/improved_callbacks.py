"""
Improved callbacks with better checkpoint management and format recommendations.

This module provides enhanced callbacks that use optimal file formats for different
types of data and models.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import tensorflow as tf
from tensorflow import keras
import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ImprovedModelCheckpoint(keras.callbacks.Callback):
    """
    Enhanced model checkpoint callback with optimal format selection.
    
    Uses SavedModel format for full model preservation and better compatibility.
    Includes metadata saving and checkpoint management.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = 'auto',
        max_checkpoints: int = 5,
        save_metadata: bool = True
    ):
        super().__init__()
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.max_checkpoints = max_checkpoints
        self.save_metadata = save_metadata
        
        # Initialize monitoring
        if mode == 'auto':
            if 'acc' in monitor or monitor.startswith('fmeasure'):
                self.mode = 'max'
            else:
                self.mode = 'min'
        
        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        else:
            self.monitor_op = np.greater
            self.best = -np.Inf
        
        self.checkpoints = []
        
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint if conditions are met."""
        logs = logs or {}
        
        # Get current metric value
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"Monitor metric '{self.monitor}' not found in logs")
            return
        
        # Check if we should save
        should_save = not self.save_best_only or self.monitor_op(current, self.best)
        
        if should_save:
            if self.save_best_only:
                self.best = current
            
            # Create checkpoint path
            checkpoint_name = f"epoch_{epoch:03d}_val_loss_{current:.6f}"
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            # Save model
            self._save_model(checkpoint_path, epoch, logs)
            
            # Manage checkpoint count
            self._manage_checkpoints()
    
    def _save_model(self, checkpoint_path: Path, epoch: int, logs: Dict[str, Any]):
        """Save model with optimal format."""
        try:
            if self.save_weights_only:
                # Save only weights (smaller files)
                weights_path = checkpoint_path.with_suffix('.weights.h5')
                self.model.save_weights(str(weights_path))
                logger.info(f"Saved model weights: {weights_path}")
            else:
                # Save full model in SavedModel format (recommended)
                self.model.save(str(checkpoint_path), save_format='tf')
                logger.info(f"Saved full model: {checkpoint_path}")
            
            # Save metadata
            if self.save_metadata:
                self._save_metadata(checkpoint_path, epoch, logs)
            
            # Track checkpoint
            self.checkpoints.append({
                'path': checkpoint_path,
                'epoch': epoch,
                'metrics': logs.copy(),
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_metadata(self, checkpoint_path: Path, epoch: int, logs: Dict[str, Any]):
        """Save checkpoint metadata."""
        metadata = {
            'epoch': epoch,
            'metrics': {k: float(v) for k, v in logs.items()},
            'timestamp': time.time(),
            'model_config': self.model.get_config(),
            'optimizer_config': self.model.optimizer.get_config(),
            'best_metric': float(self.best)
        }
        
        metadata_path = checkpoint_path.parent / f"{checkpoint_path.name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _manage_checkpoints(self):
        """Remove old checkpoints to limit disk usage."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by metric value (keep best ones)
            if self.save_best_only:
                # Keep most recent if saving best only
                checkpoints_to_remove = self.checkpoints[:-self.max_checkpoints]
            else:
                # Keep best performing ones
                sorted_checkpoints = sorted(
                    self.checkpoints,
                    key=lambda x: x['metrics'].get(self.monitor, 0),
                    reverse=(self.mode == 'max')
                )
                checkpoints_to_remove = sorted_checkpoints[self.max_checkpoints:]
            
            # Remove old checkpoints
            for checkpoint in checkpoints_to_remove:
                self._remove_checkpoint(checkpoint)
            
            # Update list
            self.checkpoints = [cp for cp in self.checkpoints if cp not in checkpoints_to_remove]
    
    def _remove_checkpoint(self, checkpoint: Dict[str, Any]):
        """Remove a checkpoint and its metadata."""
        try:
            checkpoint_path = checkpoint['path']
            
            # Remove model directory/file
            if checkpoint_path.is_dir():
                import shutil
                shutil.rmtree(checkpoint_path)
            elif checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Remove metadata
            metadata_path = checkpoint_path.parent / f"{checkpoint_path.name}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.debug(f"Removed checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")


class CheckpointManager:
    """
    Utility class for managing model checkpoints with different formats.
    """
    
    @staticmethod
    def save_model_checkpoint(
        model: keras.Model,
        checkpoint_path: str,
        format_type: str = 'savedmodel',
        include_optimizer: bool = True,
        save_metadata: bool = True,
        **kwargs
    ):
        """
        Save model checkpoint with specified format.
        
        Args:
            model: Keras model to save
            checkpoint_path: Path to save checkpoint
            format_type: 'savedmodel', 'h5', or 'weights_only'
            include_optimizer: Whether to include optimizer state
            save_metadata: Whether to save additional metadata
            **kwargs: Additional arguments for specific formats
        """
        checkpoint_path = Path(checkpoint_path)
        
        if format_type == 'savedmodel':
            # Recommended: Full model in SavedModel format
            model.save(
                str(checkpoint_path),
                save_format='tf',
                include_optimizer=include_optimizer,
                **kwargs
            )
            
        elif format_type == 'h5':
            # Legacy: H5 format (weights + architecture)
            checkpoint_path = checkpoint_path.with_suffix('.h5')
            model.save(
                str(checkpoint_path),
                save_format='h5',
                include_optimizer=include_optimizer,
                **kwargs
            )
            
        elif format_type == 'weights_only':
            # Lightweight: Only weights
            checkpoint_path = checkpoint_path.with_suffix('.weights.h5')
            model.save_weights(str(checkpoint_path), **kwargs)
            
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
        
        logger.info(f"Saved model checkpoint ({format_type}): {checkpoint_path}")
        
        # Save metadata if requested
        if save_metadata:
            CheckpointManager._save_checkpoint_metadata(
                checkpoint_path, model, format_type, include_optimizer
            )
    
    @staticmethod
    def _save_checkpoint_metadata(
        checkpoint_path: Path,
        model: keras.Model,
        format_type: str,
        include_optimizer: bool
    ):
        """Save checkpoint metadata."""
        metadata = {
            'format_type': format_type,
            'include_optimizer': include_optimizer,
            'model_summary': {
                'total_params': model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
                'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
            },
            'timestamp': time.time(),
            'tensorflow_version': tf.__version__
        }
        
        metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def create_improved_callbacks(
    checkpoint_dir: str,
    monitor: str = 'val_loss',
    early_stopping_patience: int = 20,
    lr_patience: int = 10,
    lr_factor: float = 0.5,
    max_checkpoints: int = 5,
    checkpoint_format: str = 'savedmodel'
) -> List[keras.callbacks.Callback]:
    """
    Create improved callbacks with optimal checkpoint management.
    
    Args:
        checkpoint_dir: Directory for saving checkpoints
        monitor: Metric to monitor
        early_stopping_patience: Patience for early stopping
        lr_patience: Patience for learning rate reduction
        lr_factor: Factor for learning rate reduction
        max_checkpoints: Maximum number of checkpoints to keep
        checkpoint_format: Format for checkpoints ('savedmodel', 'h5', 'weights_only')
    
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Early stopping
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    ))
    
    # Learning rate reduction
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        patience=lr_patience,
        factor=lr_factor,
        min_lr=1e-7,
        verbose=1
    ))
    
    # Improved model checkpointing
    callbacks.append(ImprovedModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=(checkpoint_format == 'weights_only'),
        max_checkpoints=max_checkpoints
    ))
    
    return callbacks


# Usage recommendations
CHECKPOINT_FORMAT_RECOMMENDATIONS = {
    'development': 'savedmodel',  # Full model for development
    'production': 'savedmodel',   # Full model for production deployment
    'experimentation': 'weights_only',  # Lightweight for many experiments
    'legacy_compatibility': 'h5'  # Only if required for compatibility
}