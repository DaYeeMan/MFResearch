# Training Infrastructure for SABR MDA-CNN

This module provides a comprehensive training infrastructure for SABR volatility surface modeling with MDA-CNN architecture.

## Components

### 1. ModelTrainer (`trainer.py`)
Main training orchestrator that handles:
- Model compilation with custom loss functions and metrics
- Training loop with validation and early stopping
- Model checkpointing and best model saving
- Learning rate scheduling and gradient clipping
- Comprehensive logging and progress monitoring

### 2. Custom Callbacks (`callbacks.py`)
Enhanced training callbacks including:
- **EarlyStoppingCallback**: Enhanced early stopping with detailed logging
- **ModelCheckpointCallback**: Model checkpointing with file management
- **LearningRateSchedulerCallback**: Learning rate scheduling with tracking
- **TrainingProgressCallback**: Progress tracking and timing
- **MetricsLoggerCallback**: Detailed metrics logging

### 3. Training Configuration (`training_config.py`)
Configuration management for:
- **OptimizerConfig**: Optimizer settings and parameters
- **LossConfig**: Loss function configuration
- **SchedulerConfig**: Learning rate scheduler settings
- **TrainingStrategy**: Advanced training techniques

### 4. Custom Loss Functions (`../models/loss_functions.py`)
Specialized loss functions for volatility surface modeling:
- **WeightedMSE**: Weighted MSE for emphasizing wings
- **RelativeMSE**: Relative error-based loss
- **HuberLoss**: Robust loss for noisy data
- **AdaptiveWeightedMSE**: Adaptive weighting during training

## Key Features

### ✅ Implemented Features

1. **Main Training Loop**
   - Validation and early stopping
   - Model checkpointing with best model saving
   - Learning rate scheduling
   - Gradient clipping
   - Progress monitoring and logging

2. **Custom Loss Functions**
   - MSE (Mean Squared Error)
   - Weighted MSE for wings emphasis
   - Relative MSE
   - Huber loss for robustness
   - Adaptive weighted MSE

3. **Training Callbacks**
   - Early stopping with patience
   - Model checkpointing with file management
   - Learning rate reduction on plateau
   - Training progress tracking
   - Comprehensive metrics logging

4. **Configuration Management**
   - Optimizer configuration (Adam, AdamW, SGD, RMSprop)
   - Loss function configuration
   - Learning rate scheduler configuration
   - Training strategy configuration

5. **Logging and Monitoring**
   - Experiment logging with structured data
   - Training progress tracking
   - Metrics logging at batch and epoch level
   - Model checkpoint management

## Usage Examples

### Basic Training
```python
from training import ModelTrainer
from utils.config import ExperimentConfig

# Create configuration
config = ExperimentConfig(name="my_experiment")

# Create trainer
trainer = ModelTrainer(config=config)

# Train model
history = trainer.train(
    model=model,
    train_dataset=train_ds,
    validation_dataset=val_ds,
    loss_name="mse",
    metrics=['mae', 'rmse']
)
```

### Training with Weighted Loss
```python
from models.loss_functions import create_wing_weight_function

# Create weight function for wings
weight_fn = create_wing_weight_function(
    atm_weight=1.0,
    wing_weight=2.0,
    wing_threshold=0.1
)

# Train with weighted MSE
history = trainer.train(
    model=model,
    train_dataset=train_ds,
    validation_dataset=val_ds,
    loss_name="weighted_mse",
    loss_kwargs={'weight_fn': weight_fn}
)
```

### Custom Training Configuration
```python
from training.training_config import TrainingSetup, OptimizerConfig, LossConfig

setup = TrainingSetup()

# Create custom optimizer
optimizer_config = OptimizerConfig(
    name="adamw",
    learning_rate=1e-3,
    weight_decay=1e-5,
    clipnorm=1.0
)
optimizer = setup.create_optimizer(optimizer_config)

# Create custom loss
loss_config = LossConfig(name="huber", huber_delta=1.0)
loss_fn = setup.create_loss_function(loss_config)
```

## File Structure

```
training/
├── __init__.py              # Module exports
├── trainer.py               # Main ModelTrainer class
├── callbacks.py             # Custom training callbacks
├── training_config.py       # Training configuration utilities
├── train_model.py          # Main training script
├── example_training.py     # Usage examples
├── test_training_infrastructure.py  # Test suite
└── README.md               # This documentation
```

## Testing

Run the test suite to verify functionality:
```bash
python training/test_training_infrastructure.py
```

Run examples to see the infrastructure in action:
```bash
python training/example_training.py
```

## Configuration Files

The training infrastructure integrates with the project's configuration system:
- `configs/default_config.yaml` - Default training parameters
- `configs/test_config.yaml` - Test configuration

## Output Structure

Training outputs are organized as:
```
results/
├── experiment_name/
│   ├── checkpoints/         # Model checkpoints
│   ├── models/             # Final and best models
│   ├── logs/               # Training logs
│   └── training_results.json  # Final results
```

## Requirements Satisfied

This implementation satisfies the following task requirements:

✅ **Main training loop with validation and early stopping**
- Implemented in `ModelTrainer.train()` method
- Early stopping with configurable patience
- Validation monitoring and best model restoration

✅ **Custom loss functions (MSE, weighted MSE for wings)**
- MSE and weighted MSE implemented
- Wing weighting for volatility surface characteristics
- Additional loss functions (Huber, Relative MSE, Adaptive Weighted MSE)

✅ **Model checkpointing and best model saving**
- Automatic checkpointing during training
- Best model identification and saving
- Checkpoint file management with size limits

✅ **Learning rate scheduling and gradient clipping**
- Reduce on plateau scheduler implemented
- Gradient clipping support in optimizer configuration
- Multiple scheduler types available

✅ **Training progress monitoring and logging**
- Comprehensive experiment logging
- Progress tracking with timing information
- Metrics logging at batch and epoch levels
- Structured logging with JSON format support

## Notes

- The infrastructure is designed to work with the existing SABR MDA-CNN project structure
- All components are thoroughly tested and documented
- The system supports both simple and advanced training scenarios
- Custom loss functions are properly integrated with the training pipeline
- Logging provides detailed information for experiment tracking and debugging