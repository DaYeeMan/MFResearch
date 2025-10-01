"""
Training configuration and setup utilities for SABR MDA-CNN models.

This module provides utilities for setting up training configurations,
optimizers, learning rate schedules, and training strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import tensorflow as tf
from tensorflow import keras
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger
from models.loss_functions import get_loss_function, get_metrics

logger = get_logger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer settings."""
    name: str = "adam"                    # Optimizer name
    learning_rate: float = 3e-4           # Initial learning rate
    weight_decay: float = 1e-5            # L2 regularization
    beta_1: float = 0.9                   # Adam beta_1 parameter
    beta_2: float = 0.999                 # Adam beta_2 parameter
    epsilon: float = 1e-7                 # Adam epsilon parameter
    clipnorm: Optional[float] = None      # Gradient clipping by norm
    clipvalue: Optional[float] = None     # Gradient clipping by value
    
    def __post_init__(self):
        """Validate optimizer configuration."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")
        if not 0 < self.beta_1 < 1:
            raise ValueError("Beta_1 must be in (0, 1)")
        if not 0 < self.beta_2 < 1:
            raise ValueError("Beta_2 must be in (0, 1)")


@dataclass
class LossConfig:
    """Configuration for loss function settings."""
    name: str = "mse"                     # Loss function name
    kwargs: Dict[str, Any] = field(default_factory=dict)  # Additional loss arguments
    
    # Weighted MSE specific settings
    wing_weight: float = 2.0              # Weight for wing regions
    atm_weight: float = 1.0               # Weight for ATM region
    wing_threshold: float = 0.1           # Threshold for wing definition
    
    # Huber loss specific settings
    huber_delta: float = 1.0              # Huber loss delta parameter
    
    # Quantile loss specific settings
    quantile: float = 0.5                 # Quantile for quantile loss


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduling."""
    name: str = "reduce_on_plateau"       # Scheduler name
    monitor: str = "val_loss"             # Metric to monitor
    patience: int = 10                    # Patience for plateau detection
    factor: float = 0.5                   # Reduction factor
    min_lr: float = 1e-7                  # Minimum learning rate
    cooldown: int = 0                     # Cooldown period
    
    # Step decay specific settings
    step_size: int = 30                   # Step size for step decay
    gamma: float = 0.1                    # Decay factor for step decay
    
    # Cosine annealing specific settings
    t_max: int = 100                      # Maximum number of iterations
    eta_min: float = 0                    # Minimum learning rate for cosine


@dataclass
class TrainingStrategy:
    """Configuration for training strategy and techniques."""
    mixed_precision: bool = False         # Use mixed precision training
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    warmup_epochs: int = 0                # Number of warmup epochs
    warmup_factor: float = 0.1            # Warmup learning rate factor
    
    # Regularization techniques
    dropout_schedule: bool = False        # Use dropout scheduling
    initial_dropout: float = 0.2          # Initial dropout rate
    final_dropout: float = 0.1            # Final dropout rate
    
    # Data augmentation
    noise_augmentation: bool = False      # Add noise to inputs
    noise_std: float = 0.01               # Standard deviation of noise


class TrainingSetup:
    """Utility class for setting up training components."""
    
    def __init__(self):
        self.logger = get_logger('training_setup')
    
    def create_optimizer(
        self,
        config: OptimizerConfig
    ) -> keras.optimizers.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            config: Optimizer configuration
            
        Returns:
            Configured optimizer instance
        """
        optimizer_kwargs = {
            'learning_rate': config.learning_rate,
            'beta_1': config.beta_1,
            'beta_2': config.beta_2,
            'epsilon': config.epsilon
        }
        
        # Add gradient clipping if specified
        if config.clipnorm is not None:
            optimizer_kwargs['clipnorm'] = config.clipnorm
        if config.clipvalue is not None:
            optimizer_kwargs['clipvalue'] = config.clipvalue
        
        if config.name.lower() == "adam":
            optimizer = keras.optimizers.Adam(**optimizer_kwargs)
        elif config.name.lower() == "adamw":
            optimizer_kwargs['weight_decay'] = config.weight_decay
            optimizer = keras.optimizers.AdamW(**optimizer_kwargs)
        elif config.name.lower() == "sgd":
            optimizer = keras.optimizers.SGD(
                learning_rate=config.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif config.name.lower() == "rmsprop":
            optimizer = keras.optimizers.RMSprop(
                learning_rate=config.learning_rate,
                rho=0.9,
                epsilon=config.epsilon
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.name}")
        
        self.logger.info(f"Created {config.name} optimizer with lr={config.learning_rate}")
        return optimizer
    
    def create_loss_function(
        self,
        config: LossConfig
    ) -> keras.losses.Loss:
        """
        Create loss function based on configuration.
        
        Args:
            config: Loss configuration
            
        Returns:
            Configured loss function instance
        """
        loss_kwargs = config.kwargs.copy()
        
        # Add specific parameters based on loss type
        if config.name == "weighted_mse":
            from models.loss_functions import create_wing_weight_function
            weight_fn = create_wing_weight_function(
                atm_weight=config.atm_weight,
                wing_weight=config.wing_weight,
                wing_threshold=config.wing_threshold
            )
            loss_kwargs['weight_fn'] = weight_fn
        elif config.name == "huber":
            loss_kwargs['delta'] = config.huber_delta
        elif config.name == "quantile":
            loss_kwargs['quantile'] = config.quantile
        
        loss_fn = get_loss_function(config.name, **loss_kwargs)
        self.logger.info(f"Created {config.name} loss function")
        return loss_fn
    
    def create_lr_scheduler(
        self,
        config: SchedulerConfig,
        optimizer: keras.optimizers.Optimizer
    ) -> Optional[keras.callbacks.Callback]:
        """
        Create learning rate scheduler based on configuration.
        
        Args:
            config: Scheduler configuration
            optimizer: Optimizer instance
            
        Returns:
            Learning rate scheduler callback or None
        """
        if config.name == "reduce_on_plateau":
            scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=config.monitor,
                factor=config.factor,
                patience=config.patience,
                min_lr=config.min_lr,
                cooldown=config.cooldown,
                verbose=1
            )
        elif config.name == "step_decay":
            def step_decay(epoch):
                return config.factor ** (epoch // config.step_size)
            
            scheduler = keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
        elif config.name == "exponential_decay":
            scheduler = keras.callbacks.LearningRateScheduler(
                lambda epoch: config.factor ** epoch,
                verbose=1
            )
        elif config.name == "cosine_annealing":
            def cosine_annealing(epoch):
                import math
                return config.eta_min + (1 - config.eta_min) * \
                       (1 + math.cos(math.pi * epoch / config.t_max)) / 2
            
            scheduler = keras.callbacks.LearningRateScheduler(cosine_annealing, verbose=1)
        elif config.name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {config.name}")
        
        self.logger.info(f"Created {config.name} learning rate scheduler")
        return scheduler
    
    def setup_mixed_precision(self, enable: bool = True):
        """
        Set up mixed precision training.
        
        Args:
            enable: Whether to enable mixed precision
        """
        if enable:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            self.logger.info("Mixed precision training enabled")
        else:
            policy = keras.mixed_precision.Policy('float32')
            keras.mixed_precision.set_global_policy(policy)
            self.logger.info("Mixed precision training disabled")
    
    def create_warmup_scheduler(
        self,
        warmup_epochs: int,
        base_lr: float,
        warmup_factor: float = 0.1
    ) -> keras.callbacks.Callback:
        """
        Create warmup learning rate scheduler.
        
        Args:
            warmup_epochs: Number of warmup epochs
            base_lr: Base learning rate after warmup
            warmup_factor: Factor for initial learning rate
            
        Returns:
            Warmup scheduler callback
        """
        def warmup_schedule(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return warmup_factor + (1.0 - warmup_factor) * epoch / warmup_epochs
            else:
                return 1.0
        
        scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch: base_lr * warmup_schedule(epoch),
            verbose=1
        )
        
        self.logger.info(f"Created warmup scheduler for {warmup_epochs} epochs")
        return scheduler
    
    def create_metrics_list(
        self,
        metric_names: List[str]
    ) -> List[keras.metrics.Metric]:
        """
        Create list of metrics based on names.
        
        Args:
            metric_names: List of metric names
            
        Returns:
            List of metric instances
        """
        metrics = get_metrics(metric_names)
        self.logger.info(f"Created metrics: {metric_names}")
        return metrics


class GradientAccumulationOptimizer:
    """
    Wrapper for gradient accumulation training.
    
    This class implements gradient accumulation to simulate larger batch sizes
    when memory is limited.
    """
    
    def __init__(
        self,
        optimizer: keras.optimizers.Optimizer,
        accumulation_steps: int = 1
    ):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.accumulated_gradients = []
        self.step_count = 0
        
    def accumulate_gradients(self, gradients):
        """Accumulate gradients over multiple steps."""
        if not self.accumulated_gradients:
            # Initialize accumulated gradients
            self.accumulated_gradients = [tf.zeros_like(g) for g in gradients]
        
        # Add current gradients
        for i, grad in enumerate(gradients):
            self.accumulated_gradients[i] = self.accumulated_gradients[i] + grad
        
        self.step_count += 1
        
        # Apply accumulated gradients if we've reached the accumulation steps
        if self.step_count >= self.accumulation_steps:
            # Average the accumulated gradients
            averaged_gradients = [
                acc_grad / self.accumulation_steps 
                for acc_grad in self.accumulated_gradients
            ]
            
            # Apply gradients
            return averaged_gradients, True
        
        return None, False
    
    def reset_accumulation(self):
        """Reset gradient accumulation state."""
        self.accumulated_gradients = []
        self.step_count = 0


def create_training_setup() -> TrainingSetup:
    """
    Factory function to create TrainingSetup instance.
    
    Returns:
        TrainingSetup instance
    """
    return TrainingSetup()


def get_default_training_configs() -> Dict[str, Any]:
    """
    Get default training configurations for different scenarios.
    
    Returns:
        Dictionary of default configurations
    """
    return {
        'optimizer': OptimizerConfig(),
        'loss': LossConfig(),
        'scheduler': SchedulerConfig(),
        'strategy': TrainingStrategy()
    }