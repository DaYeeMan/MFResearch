"""
MDA-CNN (Multi-fidelity Data Aggregation CNN) model architecture.

This module implements the main MDA-CNN model that combines:
- CNN branch for processing low-fidelity surface patches
- MLP branch for processing point features
- Fusion layer for combining representations
- Residual prediction head
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class MDACNN(keras.Model):
    """
    Multi-fidelity Data Aggregation CNN for SABR volatility surface prediction.
    
    The model predicts residuals D(ξ) = σ_MC(ξ) - σ_Hagan(ξ) by combining:
    - CNN processing of local LF surface patches
    - MLP processing of point features (SABR params, strike, maturity)
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (9, 9),
        n_point_features: int = 8,
        cnn_filters: Tuple[int, ...] = (32, 64, 128),
        cnn_kernel_size: int = 3,
        mlp_hidden_dims: Tuple[int, ...] = (64, 64),
        fusion_hidden_dims: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.2,
        activation: str = 'relu',
        name: str = 'mda_cnn',
        **kwargs
    ):
        """
        Initialize MDA-CNN model.
        
        Args:
            patch_size: Size of input surface patches (height, width)
            n_point_features: Number of point features (SABR params + strike + maturity + hagan_vol)
            cnn_filters: Number of filters for each CNN layer
            cnn_kernel_size: Kernel size for CNN layers
            mlp_hidden_dims: Hidden layer dimensions for MLP branch
            fusion_hidden_dims: Hidden layer dimensions for fusion head
            dropout_rate: Dropout rate for regularization
            activation: Activation function to use
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.patch_size = patch_size
        self.n_point_features = n_point_features
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.mlp_hidden_dims = mlp_hidden_dims
        self.fusion_hidden_dims = fusion_hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Build CNN branch for surface patches
        self._build_cnn_branch()
        
        # Build MLP branch for point features
        self._build_mlp_branch()
        
        # Build fusion head
        self._build_fusion_head()
    
    def _build_cnn_branch(self):
        """Build CNN branch for processing surface patches."""
        self.cnn_layers = []
        
        # Convolutional layers
        for i, filters in enumerate(self.cnn_filters):
            conv_layer = layers.Conv2D(
                filters=filters,
                kernel_size=self.cnn_kernel_size,
                activation=self.activation,
                padding='same',
                name=f'cnn_conv_{i+1}'
            )
            self.cnn_layers.append(conv_layer)
        
        # Global average pooling to reduce spatial dimensions
        self.cnn_gap = layers.GlobalAveragePooling2D(name='cnn_gap')
        
        # Dense layer for CNN feature extraction
        self.cnn_dense = layers.Dense(
            units=128,
            activation=self.activation,
            name='cnn_dense'
        )
    
    def _build_mlp_branch(self):
        """Build MLP branch for processing point features."""
        self.mlp_layers = []
        
        for i, units in enumerate(self.mlp_hidden_dims):
            dense_layer = layers.Dense(
                units=units,
                activation=self.activation,
                name=f'mlp_dense_{i+1}'
            )
            self.mlp_layers.append(dense_layer)
    
    def _build_fusion_head(self):
        """Build fusion head for combining CNN and MLP representations."""
        self.fusion_layers = []
        
        # Concatenation layer
        self.concat_layer = layers.Concatenate(name='fusion_concat')
        
        # Fusion dense layers
        for i, units in enumerate(self.fusion_hidden_dims):
            dense_layer = layers.Dense(
                units=units,
                activation=self.activation,
                name=f'fusion_dense_{i+1}'
            )
            self.fusion_layers.append(dense_layer)
            
            # Add dropout after each fusion layer except the last
            if i < len(self.fusion_hidden_dims) - 1:
                dropout_layer = layers.Dropout(
                    rate=self.dropout_rate,
                    name=f'fusion_dropout_{i+1}'
                )
                self.fusion_layers.append(dropout_layer)
        
        # Final prediction head - linear activation for residual prediction
        self.prediction_head = layers.Dense(
            units=1,
            activation='linear',
            name='residual_prediction'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass through the model.
        
        Args:
            inputs: Dictionary with keys:
                - 'patches': Surface patches of shape (batch_size, height, width, 1)
                - 'features': Point features of shape (batch_size, n_point_features)
            training: Whether in training mode
            
        Returns:
            Residual predictions of shape (batch_size, 1)
        """
        patches = inputs['patches']
        features = inputs['features']
        
        # CNN branch processing
        cnn_out = patches
        for layer in self.cnn_layers:
            cnn_out = layer(cnn_out)
        
        cnn_out = self.cnn_gap(cnn_out)
        cnn_features = self.cnn_dense(cnn_out)
        
        # MLP branch processing
        mlp_out = features
        for layer in self.mlp_layers:
            mlp_out = layer(mlp_out)
        
        # Fusion of CNN and MLP features
        fused_features = self.concat_layer([cnn_features, mlp_out])
        
        # Process through fusion head
        fusion_out = fused_features
        for layer in self.fusion_layers:
            if isinstance(layer, layers.Dropout):
                fusion_out = layer(fusion_out, training=training)
            else:
                fusion_out = layer(fusion_out)
        
        # Final residual prediction
        residual_pred = self.prediction_head(fusion_out)
        
        return residual_pred
    
    def get_config(self):
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'n_point_features': self.n_point_features,
            'cnn_filters': self.cnn_filters,
            'cnn_kernel_size': self.cnn_kernel_size,
            'mlp_hidden_dims': self.mlp_hidden_dims,
            'fusion_hidden_dims': self.fusion_hidden_dims,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
        })
        return config


def create_mda_cnn_model(
    patch_size: Tuple[int, int] = (9, 9),
    n_point_features: int = 8,
    **kwargs
) -> MDACNN:
    """
    Factory function to create and compile MDA-CNN model.
    
    Args:
        patch_size: Size of input surface patches
        n_point_features: Number of point features
        **kwargs: Additional arguments for MDACNN constructor
        
    Returns:
        Compiled MDACNN model
    """
    model = MDACNN(
        patch_size=patch_size,
        n_point_features=n_point_features,
        **kwargs
    )
    
    return model


class CNNBranch(keras.Model):
    """Standalone CNN branch for ablation studies."""
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (9, 9),
        filters: Tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        activation: str = 'relu',
        name: str = 'cnn_branch',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.patch_size = patch_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        
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
        
        self.gap = layers.GlobalAveragePooling2D(name='gap')
        self.dense = layers.Dense(128, activation=activation, name='dense')
        self.output_layer = layers.Dense(1, activation='linear', name='output')
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        x = self.gap(x)
        x = self.dense(x)
        return self.output_layer(x)


class MLPBranch(keras.Model):
    """Standalone MLP branch for ablation studies."""
    
    def __init__(
        self,
        n_features: int = 8,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = 'relu',
        dropout_rate: float = 0.2,
        name: str = 'mlp_branch',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Build MLP layers
        self.dense_layers = []
        for i, units in enumerate(hidden_dims):
            self.dense_layers.append(
                layers.Dense(
                    units=units,
                    activation=activation,
                    name=f'dense_{i+1}'
                )
            )
            if i < len(hidden_dims) - 1:
                self.dense_layers.append(
                    layers.Dropout(dropout_rate, name=f'dropout_{i+1}')
                )
        
        self.output_layer = layers.Dense(1, activation='linear', name='output')
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.dense_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return self.output_layer(x)