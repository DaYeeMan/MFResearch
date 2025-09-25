"""
Configuration management system for SABR MDA-CNN experiments.
Handles loading, validation, and management of experiment configurations.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SABRParams:
    """SABR model parameters with validation."""
    F0: float = 1.0          # Forward price
    alpha: float = 0.2       # Initial volatility
    beta: float = 0.5        # Elasticity parameter [0,1]
    nu: float = 0.3          # Vol-of-vol parameter
    rho: float = -0.3        # Correlation parameter [-1,1]
    
    def __post_init__(self):
        """Validate SABR parameters."""
        if not 0 <= self.beta <= 1:
            raise ValueError(f"Beta must be in [0,1], got {self.beta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"Rho must be in [-1,1], got {self.rho}")
        if self.alpha <= 0:
            raise ValueError(f"Alpha must be positive, got {self.alpha}")
        if self.nu < 0:
            raise ValueError(f"Nu must be non-negative, got {self.nu}")
        if self.F0 <= 0:
            raise ValueError(f"F0 must be positive, got {self.F0}")


@dataclass
class GridConfig:
    """Grid configuration for volatility surface discretization."""
    strike_range: list = field(default_factory=lambda: [0.5, 2.0])      # Strike range as multiple of F0
    maturity_range: list = field(default_factory=lambda: [0.1, 2.0])    # Maturity range in years
    n_strikes: int = 21                   # Number of strike points
    n_maturities: int = 11                # Number of maturity points
    
    def __post_init__(self):
        """Validate grid configuration."""
        if self.strike_range[0] >= self.strike_range[1]:
            raise ValueError("Strike range must be increasing")
        if self.maturity_range[0] >= self.maturity_range[1]:
            raise ValueError("Maturity range must be increasing")
        if self.n_strikes < 3:
            raise ValueError("Need at least 3 strike points")
        if self.n_maturities < 3:
            raise ValueError("Need at least 3 maturity points")


@dataclass
class DataGenConfig:
    """Configuration for data generation."""
    n_parameter_sets: int = 1000          # Number of SABR parameter combinations
    mc_paths: int = 100000                # Monte Carlo simulation paths
    sampling_strategy: str = "uniform"    # Parameter sampling strategy
    hf_budget: int = 200                  # Number of HF points per surface
    validation_split: float = 0.15        # Validation data fraction
    test_split: float = 0.15              # Test data fraction
    random_seed: int = 42                 # Random seed for reproducibility
    
    def __post_init__(self):
        """Validate data generation configuration."""
        if not 0 < self.validation_split < 1:
            raise ValueError("Validation split must be in (0,1)")
        if not 0 < self.test_split < 1:
            raise ValueError("Test split must be in (0,1)")
        if self.validation_split + self.test_split >= 1:
            raise ValueError("Validation + test splits must be < 1")
        if self.sampling_strategy not in ["uniform", "latin_hypercube", "adaptive"]:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")


@dataclass
class ModelConfig:
    """Configuration for MDA-CNN model architecture."""
    patch_size: list = field(default_factory=lambda: [9, 9])            # CNN input patch size
    cnn_filters: list = field(default_factory=lambda: [32, 64, 128])  # CNN filter sizes
    cnn_kernel_size: int = 3               # CNN kernel size
    mlp_hidden_dims: list = field(default_factory=lambda: [64, 64])   # MLP hidden dimensions
    fusion_dims: list = field(default_factory=lambda: [128, 64])      # Fusion layer dimensions
    dropout_rate: float = 0.2              # Dropout rate
    activation: str = "relu"               # Activation function
    
    def __post_init__(self):
        """Validate model configuration."""
        if len(self.patch_size) != 2:
            raise ValueError("Patch size must be 2D")
        if any(f <= 0 for f in self.cnn_filters):
            raise ValueError("CNN filters must be positive")
        if any(d <= 0 for d in self.mlp_hidden_dims):
            raise ValueError("MLP dimensions must be positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("Dropout rate must be in [0,1)")


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 64                  # Training batch size
    epochs: int = 200                     # Maximum training epochs
    learning_rate: float = 3e-4           # Initial learning rate
    weight_decay: float = 1e-5            # L2 regularization
    early_stopping_patience: int = 20     # Early stopping patience
    lr_scheduler_patience: int = 10       # Learning rate scheduler patience
    lr_scheduler_factor: float = 0.5      # Learning rate reduction factor
    gradient_clip_norm: float = 1.0       # Gradient clipping norm
    save_best_only: bool = True           # Save only best model
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be non-negative")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "sabr_mda_cnn_experiment"  # Experiment name
    description: str = ""                  # Experiment description
    sabr_params: SABRParams = field(default_factory=SABRParams)
    grid_config: GridConfig = field(default_factory=GridConfig)
    data_gen_config: DataGenConfig = field(default_factory=DataGenConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "new/results"        # Output directory
    
    def __post_init__(self):
        """Validate experiment configuration."""
        if not self.name:
            raise ValueError("Experiment name cannot be empty")


class ConfigManager:
    """Manages loading, saving, and validation of experiment configurations."""
    
    def __init__(self, config_dir: str = "new/configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            return self._dict_to_config(config_dict)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def save_config(self, config: ExperimentConfig, config_path: Union[str, Path]):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict(config)
        
        try:
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
                
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise
    
    def create_default_config(self, name: str = "default") -> ExperimentConfig:
        """Create a default experiment configuration."""
        return ExperimentConfig(name=name)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        # Extract nested configurations
        sabr_params = SABRParams(**config_dict.get('sabr_params', {}))
        grid_config = GridConfig(**config_dict.get('grid_config', {}))
        data_gen_config = DataGenConfig(**config_dict.get('data_gen_config', {}))
        model_config = ModelConfig(**config_dict.get('model_config', {}))
        training_config = TrainingConfig(**config_dict.get('training_config', {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['sabr_params', 'grid_config', 'data_gen_config', 
                                  'model_config', 'training_config']}
        
        return ExperimentConfig(
            sabr_params=sabr_params,
            grid_config=grid_config,
            data_gen_config=data_gen_config,
            model_config=model_config,
            training_config=training_config,
            **main_config
        )
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        return asdict(config)
    
    def validate_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration."""
        try:
            # Validation is handled in __post_init__ methods
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False