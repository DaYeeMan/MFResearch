"""
Models package for SABR volatility surface modeling.

This package contains:
- MDA-CNN: Main multi-fidelity model architecture
- Baseline models: Direct MLP, Residual MLP, CNN-only
- Model utilities: Common components and helper functions
- Loss functions: Custom losses and metrics for volatility modeling
"""

from .mda_cnn import MDACNN, create_mda_cnn_model, CNNBranch, MLPBranch
from .baseline_models import (
    DirectMLP, 
    ResidualMLP, 
    CNNOnly, 
    create_baseline_model,
    EnsembleModel
)
from .model_utils import (
    create_conv_block,
    create_dense_block,
    ResidualBlock,
    AttentionLayer,
    get_model_summary,
    count_parameters,
    create_model_checkpoint_callback,
    create_early_stopping_callback,
    create_reduce_lr_callback
)
from .loss_functions import (
    WeightedMSE,
    RelativeMSE,
    HuberLoss,
    QuantileLoss,
    RootMeanSquaredError,
    MeanAbsolutePercentageError,
    R2Score,
    create_wing_weight_function,
    get_loss_function,
    get_metrics
)

__all__ = [
    # Main models
    'MDACNN',
    'create_mda_cnn_model',
    'CNNBranch',
    'MLPBranch',
    
    # Baseline models
    'DirectMLP',
    'ResidualMLP', 
    'CNNOnly',
    'create_baseline_model',
    'EnsembleModel',
    
    # Model utilities
    'create_conv_block',
    'create_dense_block',
    'ResidualBlock',
    'AttentionLayer',
    'get_model_summary',
    'count_parameters',
    'create_model_checkpoint_callback',
    'create_early_stopping_callback',
    'create_reduce_lr_callback',
    
    # Loss functions and metrics
    'WeightedMSE',
    'RelativeMSE',
    'HuberLoss',
    'QuantileLoss',
    'RootMeanSquaredError',
    'MeanAbsolutePercentageError',
    'R2Score',
    'create_wing_weight_function',
    'get_loss_function',
    'get_metrics',
]