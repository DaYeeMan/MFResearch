# SABR Volatility Surface MDA-CNN Project

This project implements a Multi-fidelity Data Aggregation CNN (MDA-CNN) for SABR volatility surface modeling, combining high-fidelity Monte Carlo simulations with low-fidelity Hagan analytical approximations.

## Project Structure

```
new/
├── configs/                    # Configuration files
│   ├── default_config.yaml    # Default experiment configuration
│   └── test_config.yaml       # Small-scale test configuration
├── data/                       # Data storage
│   ├── raw/                   # Raw generated data
│   │   ├── hf_surfaces/       # High-fidelity MC surfaces
│   │   ├── lf_surfaces/       # Low-fidelity Hagan surfaces
│   │   └── parameters/        # SABR parameter sets
│   ├── processed/             # Preprocessed training data
│   └── splits/                # Train/validation/test splits
├── data_generation/           # Data generation modules
├── models/                    # Model architectures
├── preprocessing/             # Data preprocessing
├── training/                  # Training infrastructure
├── evaluation/                # Evaluation metrics and analysis
├── visualization/             # Plotting and visualization
├── utils/                     # Utility functions
│   ├── config.py             # Configuration management
│   ├── logging_utils.py      # Logging utilities
│   ├── reproducibility.py   # Random seed management
│   └── common.py             # Common utilities
├── results/                   # Experiment results
└── test_setup.py             # Setup validation script
```

## Quick Start

### 1. Verify Setup

Run the setup test to ensure everything is working:

```bash
python new/test_setup.py
```

### 2. Configuration

The project uses YAML configuration files. See `new/configs/default_config.yaml` for a complete example:

```yaml
name: "sabr_mda_cnn_experiment"
description: "SABR volatility surface modeling with MDA-CNN"

# SABR model parameters
sabr_params:
  F0: 1.0          # Forward price
  alpha: 0.2       # Initial volatility
  beta: 0.5        # Elasticity parameter [0,1]
  nu: 0.3          # Vol-of-vol parameter
  rho: -0.3        # Correlation parameter [-1,1]

# Grid configuration
grid_config:
  strike_range: [0.5, 2.0]      # Strike range as multiple of F0
  maturity_range: [0.1, 2.0]    # Maturity range in years
  n_strikes: 21                 # Number of strike points
  n_maturities: 11              # Number of maturity points

# Data generation
data_gen_config:
  n_parameter_sets: 1000        # Number of SABR parameter combinations
  mc_paths: 100000              # Monte Carlo simulation paths
  hf_budget: 200                # Number of HF points per surface
  random_seed: 42               # Random seed for reproducibility

# Model architecture
model_config:
  patch_size: [9, 9]            # CNN input patch size
  cnn_filters: [32, 64, 128]    # CNN filter sizes
  mlp_hidden_dims: [64, 64]     # MLP hidden dimensions
  dropout_rate: 0.2             # Dropout rate

# Training
training_config:
  batch_size: 64                # Training batch size
  epochs: 200                   # Maximum training epochs
  learning_rate: 0.0003         # Initial learning rate
  early_stopping_patience: 20   # Early stopping patience
```

### 3. Basic Usage

```python
from new.utils.common import setup_experiment

# Set up experiment with configuration
config, exp_logger = setup_experiment(
    "new/configs/default_config.yaml",
    experiment_name="my_experiment"
)

# Your experiment code here...
exp_logger.log_experiment_end({"final_loss": 0.1})
```

## Key Features

### Configuration Management
- YAML-based configuration with validation
- Hierarchical configuration structure
- Easy parameter sweeps and experiments

### Logging and Reproducibility
- Structured logging with JSON format
- Experiment tracking and metrics logging
- Deterministic random seed management
- GPU determinism support

### Modular Architecture
- Clean separation of concerns
- Extensible model architectures
- Pluggable data generation strategies
- Comprehensive evaluation metrics

## Dependencies

### Required
- numpy
- pandas
- matplotlib
- scipy
- pyyaml

### Optional
- tensorflow (for MDA-CNN models)
- torch (alternative ML framework)
- seaborn (enhanced plotting)
- psutil (system monitoring)

## Development Workflow

1. **Data Generation**: Implement SABR Monte Carlo and Hagan surface generators
2. **Preprocessing**: Create patch extraction and feature engineering
3. **Model Training**: Implement MDA-CNN architecture and training loop
4. **Evaluation**: Add comprehensive metrics and analysis
5. **Visualization**: Create plotting tools for results analysis

## Configuration Examples

### Small Test Configuration
Use `new/configs/test_config.yaml` for quick testing:
- 50 parameter sets
- 10,000 MC paths
- 20 HF points per surface
- Smaller model architecture

### Production Configuration
Use `new/configs/default_config.yaml` for full experiments:
- 1,000 parameter sets
- 100,000 MC paths
- 200 HF points per surface
- Full model architecture

## Logging

The project provides comprehensive logging:

```python
from new.utils.logging_utils import setup_logging, ExperimentLogger

# Set up logging
logger = setup_logging(
    log_level="INFO",
    log_dir="new/results/logs",
    experiment_name="my_experiment"
)

# Experiment-specific logging
exp_logger = ExperimentLogger("my_experiment", "new/results/logs")
exp_logger.log_epoch(1, {"loss": 0.5, "accuracy": 0.8})
```

## Reproducibility

Ensure reproducible experiments:

```python
from new.utils.reproducibility import set_random_seed, ReproducibleContext

# Set global seed
set_random_seed(42, deterministic_gpu=True)

# Use context for specific operations
with ReproducibleContext(seed=42):
    # Your reproducible code here
    pass
```

## Next Steps

1. Implement data generation components (Task 2)
2. Create SABR Monte Carlo simulation engine (Task 3)
3. Implement Hagan analytical surface generator (Task 4)
4. Build MDA-CNN model architecture (Task 8)
5. Create training and evaluation pipelines

## Testing

Run the setup test to verify your environment:

```bash
python new/test_setup.py
```

This will test:
- Configuration system
- Logging utilities
- Reproducibility features
- Complete experiment setup

## Support

For issues or questions, refer to the project documentation or check the implementation tasks in the spec files.