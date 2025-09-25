"""
Baseline models for comparison with MDA-CNN.

This module implements various baseline architectures:
- Direct MLP: point features → volatility
- Residual MLP: point features → residual (no patches)
- CNN-only: patches → residual (no point features)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class DirectMLP(keras.Model):
    """
    Direct MLP baseline that predicts absolute volatility from point features.
    
    This model directly maps point features (SABR params, strike, maturity)
    to absolute volatility values without using residual learning.
    """
    
    def __init__(
        self,
        n_features: int = 7,  # SABR params + strike + maturity (no Hagan vol)
        hidden_dims: Tuple[int, ...] = (128, 128, 64),
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        name: str = 'direct_mlp',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        self.layers_list = []
        
        for i, units in enumerate(hidden_dims):
            self.layers_list.append(
                layers.Dense(
                    units=units,
                    activation=activation,
                    name=f'dense_{i+1}'
                )
            )
            
            # Add dropout except for the last hidden layer
            if i < len(hidden_dims) - 1:
                self.layers_list.append(
                    layers.Dropout(dropout_rate, name=f'dropout_{i+1}')
                )
        
        # Output layer with ReLU activation (volatility must be positive)
        self.output_layer = layers.Dense(
            units=1,
            activation='relu',
            name='volatility_output'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: Point features of shape (batch_size, n_features)
            training: Whether in training mode
            
        Returns:
            Absolute volatility predictions of shape (batch_size, 1)
        """
        x = inputs
        
        for layer in self.layers_list:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        return self.output_layer(x)


class ResidualMLP(keras.Model):
    """
    Residual MLP baseline that predicts residuals from point features only.
    
    This model uses residual learning like MDA-CNN but without surface patches,
    only point features including the Hagan volatility.
    """
    
    def __init__(
        self,
        n_features: int = 8,  # SABR params + strike + maturity + Hagan vol
        hidden_dims: Tuple[int, ...] = (128, 128, 64),
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        name: str = 'residual_mlp',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        self.layers_list = []
        
        for i, units in enumerate(hidden_dims):
            self.layers_list.append(
                layers.Dense(
                    units=units,
                    activation=activation,
                    name=f'dense_{i+1}'
                )
            )
            
            # Add dropout except for the last hidden layer
            if i < len(hidden_dims) - 1:
                self.layers_list.append(
                    layers.Dropout(dropout_rate, name=f'dropout_{i+1}')
                )
        
        # Output layer with linear activation for residual prediction
        self.output_layer = layers.Dense(
            units=1,
            activation='linear',
            name='residual_output'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: Point features of shape (batch_size, n_features)
            training: Whether in training mode
            
        Returns:
            Residual predictions of shape (batch_size, 1)
        """
        x = inputs
        
        for layer in self.layers_list:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        return self.output_layer(x)


class CNNOnly(keras.Model):
    """
    CNN-only baseline that predicts residuals from surface patches only.
    
    This model uses only the CNN branch without point features,
    for ablation studies to understand the contribution of each branch.
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (9, 9),
        filters: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        name: str = 'cnn_only',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.patch_size = patch_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Build CNN layers
        self.conv_layers = []
        for i, f in enumerate(filters):
            self.conv_layers.append(
                layers.Conv2D(
                    filters=f,
                    kernel_size=kernel_size,
                    activation=activation,
                    padding='same',
                    name=f'conv_{i+1}'
                )
            )
        
        # Global average pooling
        self.gap = layers.GlobalAveragePooling2D(name='gap')
        
        # Dense layers
        self.dense1 = layers.Dense(128, activation=activation, name='dense1')
        self.dropout = layers.Dropout(dropout_rate, name='dropout')
        self.dense2 = layers.Dense(64, activation=activation, name='dense2')
        
        # Output layer with linear activation for residual prediction
        self.output_layer = layers.Dense(1, activation='linear', name='residual_output')
    
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: Surface patches of shape (batch_size, height, width, 1)
            training: Whether in training mode
            
        Returns:
            Residual predictions of shape (batch_size, 1)
        """
        x = inputs
        
        # CNN processing
        for layer in self.conv_layers:
            x = layer(x)
        
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        return self.output_layer(x)


def create_baseline_model(
    model_type: str,
    **kwargs
) -> keras.Model:
    """
    Factory function to create baseline models.
    
    Args:
        model_type: Type of baseline model ('direct_mlp', 'residual_mlp', 'cnn_only')
        **kwargs: Additional arguments for model constructor
        
    Returns:
        Compiled baseline model
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == 'direct_mlp':
        return DirectMLP(**kwargs)
    elif model_type == 'residual_mlp':
        return ResidualMLP(**kwargs)
    elif model_type == 'cnn_only':
        return CNNOnly(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: 'direct_mlp', 'residual_mlp', 'cnn_only'")


class EnsembleModel(keras.Model):
    """
    Ensemble model that combines predictions from multiple models.
    
    This can be used to combine MDA-CNN with baseline models
    for potentially improved performance.
    """
    
    def __init__(
        self,
        models: list,
        weights: Optional[list] = None,
        name: str = 'ensemble',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.models = models
        self.n_models = len(models)
        
        if weights is None:
            self.model_weights = [1.0 / self.n_models] * self.n_models
        else:
            if len(weights) != self.n_models:
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            total_weight = sum(weights)
            self.model_weights = [w / total_weight for w in weights]
    
    def call(self, inputs, training=None):
        """
        Forward pass through ensemble.
        
        Args:
            inputs: Model inputs (format depends on constituent models)
            training: Whether in training mode
            
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(inputs, training=training)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = tf.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.model_weights):
            weighted_pred += weight * pred
        
        return weighted_pred