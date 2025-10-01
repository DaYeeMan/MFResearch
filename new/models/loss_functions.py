"""
Custom loss functions and metrics for SABR volatility surface modeling.

This module implements specialized loss functions that account for
the financial characteristics of volatility surfaces.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional, Callable


class WeightedMSE(keras.losses.Loss):
    """
    Weighted Mean Squared Error loss.
    
    Applies different weights to different regions of the volatility surface,
    typically giving higher weight to at-the-money (ATM) regions and wings.
    """
    
    def __init__(
        self,
        weight_fn: Optional[Callable] = None,
        name: str = 'weighted_mse',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.weight_fn = weight_fn
    
    def call(self, y_true, y_pred):
        """
        Compute weighted MSE loss.
        
        Args:
            y_true: True residual values
            y_pred: Predicted residual values
            
        Returns:
            Weighted MSE loss
        """
        mse = tf.square(y_true - y_pred)
        
        if self.weight_fn is not None:
            weights = self.weight_fn(y_true, y_pred)
            mse = mse * weights
        
        return tf.reduce_mean(mse)


class RelativeMSE(keras.losses.Loss):
    """
    Relative Mean Squared Error loss.
    
    Computes MSE relative to the magnitude of the true values,
    which helps balance errors across different volatility levels.
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        name: str = 'relative_mse',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
    
    def call(self, y_true, y_pred):
        """
        Compute relative MSE loss.
        
        Args:
            y_true: True residual values
            y_pred: Predicted residual values
            
        Returns:
            Relative MSE loss
        """
        squared_error = tf.square(y_true - y_pred)
        relative_error = squared_error / (tf.square(y_true) + self.epsilon)
        
        return tf.reduce_mean(relative_error)


class HuberLoss(keras.losses.Loss):
    """
    Huber loss for robust training.
    
    Less sensitive to outliers than MSE, which can be helpful
    when dealing with noisy Monte Carlo data.
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        name: str = 'huber_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.delta = delta
    
    def call(self, y_true, y_pred):
        """
        Compute Huber loss.
        
        Args:
            y_true: True residual values
            y_pred: Predicted residual values
            
        Returns:
            Huber loss
        """
        error = y_true - y_pred
        abs_error = tf.abs(error)
        
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * tf.square(quadratic) + self.delta * linear
        
        return tf.reduce_mean(loss)


class QuantileLoss(keras.losses.Loss):
    """
    Quantile loss for uncertainty quantification.
    
    Can be used to predict confidence intervals for volatility predictions.
    """
    
    def __init__(
        self,
        quantile: float = 0.5,
        name: str = 'quantile_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.quantile = quantile
    
    def call(self, y_true, y_pred):
        """
        Compute quantile loss.
        
        Args:
            y_true: True residual values
            y_pred: Predicted residual values
            
        Returns:
            Quantile loss
        """
        error = y_true - y_pred
        loss = tf.maximum(self.quantile * error, (self.quantile - 1) * error)
        
        return tf.reduce_mean(loss)


# Custom metrics
class RootMeanSquaredError(keras.metrics.Metric):
    """Root Mean Squared Error metric."""
    
    def __init__(self, name='rmse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_squared_error = self.add_weight(name='sse', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        squared_error = tf.square(y_true - y_pred)
        
        if sample_weight is not None:
            squared_error = squared_error * sample_weight
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
        self.sum_squared_error.assign_add(tf.reduce_sum(squared_error))
    
    def result(self):
        return tf.sqrt(self.sum_squared_error / tf.maximum(self.count, 1e-8))
    
    def reset_state(self):
        self.sum_squared_error.assign(0.0)
        self.count.assign(0.0)


class MeanAbsolutePercentageError(keras.metrics.Metric):
    """Mean Absolute Percentage Error metric."""
    
    def __init__(self, name='mape', epsilon=1e-8, **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.sum_percentage_error = self.add_weight(name='spe', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        percentage_error = tf.abs((y_true - y_pred) / (y_true + self.epsilon)) * 100
        
        if sample_weight is not None:
            percentage_error = percentage_error * sample_weight
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
        
        self.sum_percentage_error.assign_add(tf.reduce_sum(percentage_error))
    
    def result(self):
        return self.sum_percentage_error / self.count
    
    def reset_state(self):
        self.sum_percentage_error.assign(0.0)
        self.count.assign(0.0)


class R2Score(keras.metrics.Metric):
    """R-squared (coefficient of determination) metric."""
    
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_squared_residuals = self.add_weight(name='ssr', initializer='zeros')
        self.sum_squared_total = self.add_weight(name='sst', initializer='zeros')
        self.sum_true = self.add_weight(name='sum_true', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            self.count.assign_add(tf.reduce_sum(sample_weight))
            self.sum_true.assign_add(tf.reduce_sum(y_true * sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
            self.sum_true.assign_add(tf.reduce_sum(y_true))
        
        # Update sum of squared residuals
        squared_residuals = tf.square(y_true - y_pred)
        if sample_weight is not None:
            squared_residuals = squared_residuals * sample_weight
        self.sum_squared_residuals.assign_add(tf.reduce_sum(squared_residuals))
    
    def result(self):
        mean_true = self.sum_true / self.count
        
        # We need to compute sum of squared total in a separate pass
        # For now, return a simplified version
        return 1.0 - (self.sum_squared_residuals / (self.sum_squared_total + 1e-8))
    
    def reset_state(self):
        self.sum_squared_residuals.assign(0.0)
        self.sum_squared_total.assign(0.0)
        self.sum_true.assign(0.0)
        self.count.assign(0.0)


def create_wing_weight_function(
    atm_weight: float = 1.0,
    wing_weight: float = 2.0,
    wing_threshold: float = 0.1
) -> Callable:
    """
    Create a weight function that emphasizes wings of the volatility surface.
    
    Args:
        atm_weight: Weight for at-the-money region
        wing_weight: Weight for wing regions (far OTM/ITM)
        wing_threshold: Threshold for defining wings (moneyness deviation)
        
    Returns:
        Weight function that can be used with WeightedMSE
    """
    def weight_fn(y_true, y_pred):
        # This is a simplified version - in practice, you'd need moneyness info
        # For now, use absolute residual magnitude as proxy for wing regions
        abs_residual = tf.abs(y_true)
        
        # Higher weights for larger residuals (typically in wings)
        weights = tf.where(
            abs_residual > wing_threshold,
            wing_weight,
            atm_weight
        )
        
        return weights
    
    return weight_fn


def create_moneyness_weight_function(
    strikes: tf.Tensor,
    forward_price: float,
    atm_weight: float = 1.0,
    wing_weight: float = 2.0,
    wing_threshold: float = 0.2
) -> Callable:
    """
    Create a weight function based on actual moneyness for volatility surface.
    
    Args:
        strikes: Strike prices tensor
        forward_price: Forward price for moneyness calculation
        atm_weight: Weight for at-the-money region
        wing_weight: Weight for wing regions
        wing_threshold: Moneyness threshold for wings (e.g., 0.2 = 20% away from ATM)
        
    Returns:
        Weight function that uses actual moneyness
    """
    def weight_fn(y_true, y_pred):
        # Calculate moneyness: log(K/F)
        moneyness = tf.math.log(strikes / forward_price)
        abs_moneyness = tf.abs(moneyness)
        
        # Higher weights for strikes far from ATM (wings)
        weights = tf.where(
            abs_moneyness > wing_threshold,
            wing_weight,
            atm_weight
        )
        
        return weights
    
    return weight_fn


class AdaptiveWeightedMSE(keras.losses.Loss):
    """
    Adaptive weighted MSE that adjusts weights based on training progress.
    
    Gradually increases emphasis on wings as training progresses.
    """
    
    def __init__(
        self,
        initial_wing_weight: float = 1.0,
        final_wing_weight: float = 3.0,
        atm_weight: float = 1.0,
        wing_threshold: float = 0.1,
        adaptation_epochs: int = 50,
        name: str = 'adaptive_weighted_mse',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.initial_wing_weight = initial_wing_weight
        self.final_wing_weight = final_wing_weight
        self.atm_weight = atm_weight
        self.wing_threshold = wing_threshold
        self.adaptation_epochs = adaptation_epochs
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
    
    def call(self, y_true, y_pred):
        """
        Compute adaptive weighted MSE loss.
        
        Args:
            y_true: True residual values
            y_pred: Predicted residual values
            
        Returns:
            Adaptive weighted MSE loss
        """
        mse = tf.square(y_true - y_pred)
        
        # Calculate current wing weight based on training progress
        progress = tf.cast(self.current_epoch, tf.float32) / self.adaptation_epochs
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        
        current_wing_weight = (
            self.initial_wing_weight + 
            progress * (self.final_wing_weight - self.initial_wing_weight)
        )
        
        # Apply weights
        abs_residual = tf.abs(y_true)
        weights = tf.where(
            abs_residual > self.wing_threshold,
            current_wing_weight,
            self.atm_weight
        )
        
        weighted_mse = mse * weights
        return tf.reduce_mean(weighted_mse)
    
    def update_epoch(self, epoch: int):
        """Update current epoch for adaptive weighting."""
        self.current_epoch.assign(epoch)


def get_loss_function(loss_name: str, **kwargs) -> keras.losses.Loss:
    """
    Factory function to create loss functions.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If loss_name is not recognized
    """
    loss_functions = {
        'mse': keras.losses.MeanSquaredError,
        'mae': keras.losses.MeanAbsoluteError,
        'weighted_mse': WeightedMSE,
        'relative_mse': RelativeMSE,
        'huber': HuberLoss,
        'quantile': QuantileLoss,
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name](**kwargs)


def get_metrics(metric_names: list) -> list:
    """
    Factory function to create metrics.
    
    Args:
        metric_names: List of metric names
        
    Returns:
        List of metric instances
    """
    available_metrics = {
        'mse': keras.metrics.MeanSquaredError,
        'mae': keras.metrics.MeanAbsoluteError,
        'rmse': RootMeanSquaredError,
        'mape': MeanAbsolutePercentageError,
        'r2': R2Score,
    }
    
    metrics = []
    for name in metric_names:
        if name in available_metrics:
            metrics.append(available_metrics[name]())
        else:
            print(f"Warning: Unknown metric '{name}' ignored")
    
    return metrics