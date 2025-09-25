"""
Model utilities and common components.

This module provides utility functions and common components
used across different model architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict, Any, Optional


def create_conv_block(
    filters: int,
    kernel_size: int = 3,
    activation: str = 'relu',
    padding: str = 'same',
    use_batch_norm: bool = False,
    dropout_rate: float = 0.0,
    name_prefix: str = 'conv_block'
) -> list:
    """
    Create a convolutional block with optional batch normalization and dropout.
    
    Args:
        filters: Number of filters
        kernel_size: Convolution kernel size
        activation: Activation function
        padding: Padding type
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate (0 means no dropout)
        name_prefix: Prefix for layer names
        
    Returns:
        List of layers forming the conv block
    """
    block_layers = []
    
    # Convolution layer
    block_layers.append(
        layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation=None if use_batch_norm else activation,
            padding=padding,
            name=f'{name_prefix}_conv'
        )
    )
    
    # Batch normalization
    if use_batch_norm:
        block_layers.append(
            layers.BatchNormalization(name=f'{name_prefix}_bn')
        )
        block_layers.append(
            layers.Activation(activation, name=f'{name_prefix}_act')
        )
    
    # Dropout
    if dropout_rate > 0:
        block_layers.append(
            layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout')
        )
    
    return block_layers


def create_dense_block(
    units: int,
    activation: str = 'relu',
    use_batch_norm: bool = False,
    dropout_rate: float = 0.0,
    name_prefix: str = 'dense_block'
) -> list:
    """
    Create a dense block with optional batch normalization and dropout.
    
    Args:
        units: Number of units
        activation: Activation function
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate (0 means no dropout)
        name_prefix: Prefix for layer names
        
    Returns:
        List of layers forming the dense block
    """
    block_layers = []
    
    # Dense layer
    block_layers.append(
        layers.Dense(
            units=units,
            activation=None if use_batch_norm else activation,
            name=f'{name_prefix}_dense'
        )
    )
    
    # Batch normalization
    if use_batch_norm:
        block_layers.append(
            layers.BatchNormalization(name=f'{name_prefix}_bn')
        )
        block_layers.append(
            layers.Activation(activation, name=f'{name_prefix}_act')
        )
    
    # Dropout
    if dropout_rate > 0:
        block_layers.append(
            layers.Dropout(dropout_rate, name=f'{name_prefix}_dropout')
        )
    
    return block_layers


class ResidualBlock(layers.Layer):
    """
    Residual block for CNN architectures.
    
    Implements skip connections to help with gradient flow
    in deeper networks.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        name: str = 'residual_block',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        # First conv block
        self.conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=None,
            name=f'{name}_conv1'
        )
        
        if use_batch_norm:
            self.bn1 = layers.BatchNormalization(name=f'{name}_bn1')
        
        self.act1 = layers.Activation(activation, name=f'{name}_act1')
        
        # Second conv block
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=None,
            name=f'{name}_conv2'
        )
        
        if use_batch_norm:
            self.bn2 = layers.BatchNormalization(name=f'{name}_bn2')
        
        # Skip connection projection if needed
        self.skip_conv = None
        
        # Final activation
        self.final_act = layers.Activation(activation, name=f'{name}_final_act')
    
    def build(self, input_shape):
        # Create skip connection projection if input channels != output channels
        if input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                padding='same',
                activation=None,
                name=f'{self.name}_skip_conv'
            )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        
        # Skip connection
        skip = inputs
        if self.skip_conv is not None:
            skip = self.skip_conv(inputs)
        
        # Add skip connection and apply final activation
        x = x + skip
        x = self.final_act(x)
        
        return x


class AttentionLayer(layers.Layer):
    """
    Simple attention mechanism for feature fusion.
    
    Computes attention weights to focus on important features
    when combining CNN and MLP representations.
    """
    
    def __init__(
        self,
        attention_dim: int = 64,
        name: str = 'attention',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention_dense = layers.Dense(
            attention_dim,
            activation='tanh',
            name=f'{name}_dense'
        )
        
        self.attention_weights = layers.Dense(
            1,
            activation='sigmoid',
            name=f'{name}_weights'
        )
    
    def call(self, inputs, training=None):
        """
        Apply attention mechanism.
        
        Args:
            inputs: List of feature tensors to attend over
            training: Whether in training mode
            
        Returns:
            Attended feature representation
        """
        # Concatenate all input features
        concat_features = tf.concat(inputs, axis=-1)
        
        # Compute attention weights
        attention_scores = self.attention_dense(concat_features)
        attention_weights = self.attention_weights(attention_scores)
        
        # Apply attention weights
        attended_features = attention_weights * concat_features
        
        return attended_features


def get_model_summary(model: keras.Model, input_shapes: Dict[str, Tuple]) -> str:
    """
    Get a formatted model summary.
    
    Args:
        model: Keras model
        input_shapes: Dictionary mapping input names to shapes
        
    Returns:
        Formatted model summary string
    """
    # Build model with dummy inputs
    dummy_inputs = {}
    for name, shape in input_shapes.items():
        dummy_inputs[name] = tf.zeros((1,) + shape)
    
    # Forward pass to build the model
    _ = model(dummy_inputs)
    
    # Get summary
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    
    return '\n'.join(summary_lines)


def count_parameters(model: keras.Model) -> Dict[str, int]:
    """
    Count trainable and non-trainable parameters in a model.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    non_trainable_params = sum([tf.size(var).numpy() for var in model.non_trainable_variables])
    
    return {
        'trainable': trainable_params,
        'non_trainable': non_trainable_params,
        'total': trainable_params + non_trainable_params
    }


def create_model_checkpoint_callback(
    filepath: str,
    monitor: str = 'val_loss',
    save_best_only: bool = True,
    save_weights_only: bool = False,
    mode: str = 'min',
    verbose: int = 1
) -> keras.callbacks.ModelCheckpoint:
    """
    Create a model checkpoint callback.
    
    Args:
        filepath: Path to save the model
        monitor: Metric to monitor
        save_best_only: Whether to save only the best model
        save_weights_only: Whether to save only weights
        mode: 'min' or 'max' for the monitored metric
        verbose: Verbosity level
        
    Returns:
        ModelCheckpoint callback
    """
    return keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        mode=mode,
        verbose=verbose
    )


def create_early_stopping_callback(
    monitor: str = 'val_loss',
    patience: int = 10,
    restore_best_weights: bool = True,
    mode: str = 'min',
    verbose: int = 1
) -> keras.callbacks.EarlyStopping:
    """
    Create an early stopping callback.
    
    Args:
        monitor: Metric to monitor
        patience: Number of epochs with no improvement to wait
        restore_best_weights: Whether to restore best weights
        mode: 'min' or 'max' for the monitored metric
        verbose: Verbosity level
        
    Returns:
        EarlyStopping callback
    """
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights,
        mode=mode,
        verbose=verbose
    )


def create_reduce_lr_callback(
    monitor: str = 'val_loss',
    factor: float = 0.5,
    patience: int = 5,
    min_lr: float = 1e-7,
    mode: str = 'min',
    verbose: int = 1
) -> keras.callbacks.ReduceLROnPlateau:
    """
    Create a learning rate reduction callback.
    
    Args:
        monitor: Metric to monitor
        factor: Factor by which to reduce learning rate
        patience: Number of epochs with no improvement to wait
        min_lr: Minimum learning rate
        mode: 'min' or 'max' for the monitored metric
        verbose: Verbosity level
        
    Returns:
        ReduceLROnPlateau callback
    """
    return keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        mode=mode,
        verbose=verbose
    )