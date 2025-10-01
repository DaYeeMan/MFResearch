"""
Example script demonstrating the training infrastructure usage.

This script shows how to use the training components with synthetic data
for testing and demonstration purposes.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ExperimentConfig, TrainingConfig, ModelConfig
from utils.logging_utils import setup_logging, ExperimentLogger
from training.trainer import ModelTrainer
from training.callbacks import create_default_callbacks
from models.loss_functions import WeightedMSE, create_wing_weight_function


def create_synthetic_data(
    n_samples: int = 1000,
    patch_size: tuple = (9, 9),
    n_features: int = 8
) -> tuple:
    """
    Create synthetic data for testing the training infrastructure.
    
    Args:
        n_samples: Number of samples to generate
        patch_size: Size of surface patches
        n_features: Number of point features
        
    Returns:
        Tuple of (patches, features, targets)
    """
    # Generate random surface patches (simulating LF surface patches)
    patches = np.random.normal(0.2, 0.05, (n_samples, *patch_size, 1))
    
    # Generate random point features (SABR params, strike, maturity, etc.)
    features = np.random.normal(0, 1, (n_samples, n_features))
    
    # Generate synthetic targets (residuals between HF and LF)
    # Add some correlation with features to make it learnable
    targets = (
        0.1 * features[:, 0] +  # Correlation with first feature
        0.05 * features[:, 1] + # Correlation with second feature
        np.random.normal(0, 0.02, n_samples)  # Noise
    ).reshape(-1, 1)
    
    return patches, features, targets


def create_synthetic_datasets(
    train_size: int = 800,
    val_size: int = 100,
    test_size: int = 100,
    batch_size: int = 32
) -> tuple:
    """
    Create synthetic TensorFlow datasets for training.
    
    Args:
        train_size: Size of training set
        val_size: Size of validation set
        test_size: Size of test set
        batch_size: Batch size for datasets
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create synthetic data
    train_patches, train_features, train_targets = create_synthetic_data(train_size)
    val_patches, val_features, val_targets = create_synthetic_data(val_size)
    test_patches, test_features, test_targets = create_synthetic_data(test_size)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_patches, train_features), train_targets)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((val_patches, val_features), val_targets)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(
        ((test_patches, test_features), test_targets)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset


def create_simple_model(
    patch_size: tuple = (9, 9),
    n_features: int = 8
) -> keras.Model:
    """
    Create a simple MDA-CNN-like model for testing.
    
    Args:
        patch_size: Size of input patches
        n_features: Number of point features
        
    Returns:
        Keras model
    """
    # Patch input (CNN branch)
    patch_input = keras.layers.Input(shape=(*patch_size, 1), name='patch_input')
    x = keras.layers.Conv2D(16, 3, activation='relu', padding='same')(patch_input)
    x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    cnn_features = keras.layers.Dense(32, activation='relu')(x)
    
    # Point features input (MLP branch)
    point_input = keras.layers.Input(shape=(n_features,), name='point_input')
    y = keras.layers.Dense(32, activation='relu')(point_input)
    y = keras.layers.Dense(32, activation='relu')(y)
    mlp_features = keras.layers.Dense(32, activation='relu')(y)
    
    # Fusion
    combined = keras.layers.concatenate([cnn_features, mlp_features])
    z = keras.layers.Dense(64, activation='relu')(combined)
    z = keras.layers.Dropout(0.2)(z)
    z = keras.layers.Dense(32, activation='relu')(z)
    output = keras.layers.Dense(1, activation='linear')(z)
    
    model = keras.Model(inputs=[patch_input, point_input], outputs=output)
    return model


def example_basic_training():
    """Example of basic training with MSE loss."""
    print("="*60)
    print("EXAMPLE 1: Basic Training with MSE Loss")
    print("="*60)
    
    # Set up logging
    setup_logging(log_level="INFO", console_output=True)
    
    # Create configuration
    config = ExperimentConfig(
        name="example_basic_training",
        training_config=TrainingConfig(
            batch_size=32,
            epochs=10,
            learning_rate=1e-3,
            early_stopping_patience=5
        )
    )
    
    # Create synthetic datasets
    train_ds, val_ds, test_ds = create_synthetic_datasets(batch_size=32)
    
    # Create model
    model = create_simple_model()
    
    # Create trainer
    trainer = ModelTrainer(config=config, output_dir="new/results/example_basic")
    
    # Train model
    history = trainer.train(
        model=model,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        loss_name="mse",
        metrics=['mse', 'mae']
    )
    
    # Evaluate
    test_results = trainer.evaluate(model=model, test_dataset=test_ds)
    
    print(f"Training completed! Final test loss: {test_results['loss']:.6f}")
    return history, test_results


def example_weighted_loss_training():
    """Example of training with weighted MSE loss."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Training with Weighted MSE Loss")
    print("="*60)
    
    # Create configuration
    config = ExperimentConfig(
        name="example_weighted_training",
        training_config=TrainingConfig(
            batch_size=32,
            epochs=10,
            learning_rate=1e-3,
            early_stopping_patience=5
        )
    )
    
    # Create synthetic datasets
    train_ds, val_ds, test_ds = create_synthetic_datasets(batch_size=32)
    
    # Create model
    model = create_simple_model()
    
    # Create trainer
    trainer = ModelTrainer(config=config, output_dir="new/results/example_weighted")
    
    # Train with weighted MSE
    weight_fn = create_wing_weight_function(
        atm_weight=1.0,
        wing_weight=2.0,
        wing_threshold=0.05
    )
    
    history = trainer.train(
        model=model,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        loss_name="weighted_mse",
        loss_kwargs={'weight_fn': weight_fn},
        metrics=['mse', 'mae']
    )
    
    # Evaluate
    test_results = trainer.evaluate(model=model, test_dataset=test_ds)
    
    print(f"Weighted training completed! Final test loss: {test_results['loss']:.6f}")
    return history, test_results


def example_custom_callbacks():
    """Example of training with custom callbacks."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Training with Custom Callbacks")
    print("="*60)
    
    # Create configuration
    config = ExperimentConfig(
        name="example_custom_callbacks",
        training_config=TrainingConfig(
            batch_size=32,
            epochs=15,
            learning_rate=1e-3,
            early_stopping_patience=8,
            lr_scheduler_patience=5,
            lr_scheduler_factor=0.5
        )
    )
    
    # Create synthetic datasets
    train_ds, val_ds, test_ds = create_synthetic_datasets(batch_size=32)
    
    # Create model
    model = create_simple_model()
    
    # Create trainer with experiment logger
    experiment_logger = ExperimentLogger(
        experiment_name=config.name,
        log_dir="new/results/example_callbacks/logs"
    )
    
    trainer = ModelTrainer(
        config=config,
        output_dir="new/results/example_callbacks",
        experiment_logger=experiment_logger
    )
    
    # Create additional custom callbacks
    custom_callbacks = [
        keras.callbacks.CSVLogger("new/results/example_callbacks/training_log.csv"),
        keras.callbacks.TensorBoard(log_dir="new/results/example_callbacks/tensorboard")
    ]
    
    # Train model
    history = trainer.train(
        model=model,
        train_dataset=train_ds,
        validation_dataset=val_ds,
        loss_name="mse",
        metrics=['mse', 'mae', 'rmse'],
        callbacks=custom_callbacks
    )
    
    # Evaluate
    test_results = trainer.evaluate(model=model, test_dataset=test_ds)
    
    print(f"Custom callbacks training completed! Final test loss: {test_results['loss']:.6f}")
    return history, test_results


def main():
    """Run all training examples."""
    print("SABR MDA-CNN Training Infrastructure Examples")
    print("=" * 60)
    
    try:
        # Example 1: Basic training
        history1, results1 = example_basic_training()
        
        # Example 2: Weighted loss training
        history2, results2 = example_weighted_loss_training()
        
        # Example 3: Custom callbacks
        history3, results3 = example_custom_callbacks()
        
        # Summary
        print("\n" + "="*60)
        print("EXAMPLES SUMMARY")
        print("="*60)
        print(f"Basic MSE training - Final test loss: {results1['loss']:.6f}")
        print(f"Weighted MSE training - Final test loss: {results2['loss']:.6f}")
        print(f"Custom callbacks training - Final test loss: {results3['loss']:.6f}")
        print("\nAll examples completed successfully!")
        print("Check the 'new/results/' directory for outputs.")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()