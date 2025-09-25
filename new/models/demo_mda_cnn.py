"""
Demonstration script for MDA-CNN model architecture.

This script shows how to create, compile, and use the MDA-CNN model
for SABR volatility surface residual prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from mda_cnn import MDACNN, create_mda_cnn_model
from baseline_models import create_baseline_model
from loss_functions import get_loss_function, get_metrics


def create_synthetic_data(n_samples=1000, patch_size=(9, 9), n_features=8):
    """Create synthetic data for demonstration."""
    print(f"Creating synthetic data: {n_samples} samples")
    
    # Create random surface patches (representing LF Hagan surfaces)
    patches = np.random.randn(n_samples, *patch_size, 1).astype(np.float32)
    
    # Create point features (SABR params + strike + maturity + Hagan vol)
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create synthetic residuals with some pattern
    # Residuals depend on both patch characteristics and point features
    patch_mean = np.mean(patches.reshape(n_samples, -1), axis=1, keepdims=True)
    feature_sum = np.sum(features, axis=1, keepdims=True)
    
    residuals = (
        0.1 * patch_mean +
        0.05 * feature_sum +
        0.02 * np.random.randn(n_samples, 1)
    ).astype(np.float32)
    
    return patches, features, residuals


def demonstrate_mda_cnn():
    """Demonstrate MDA-CNN model functionality."""
    print("=" * 60)
    print("MDA-CNN Model Demonstration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Create synthetic data
    patches, features, residuals = create_synthetic_data(n_samples=1000)
    
    print(f"Data shapes:")
    print(f"  Patches: {patches.shape}")
    print(f"  Features: {features.shape}")
    print(f"  Residuals: {residuals.shape}")
    
    # Split data
    split_idx = int(0.8 * len(patches))
    
    train_data = {
        'patches': patches[:split_idx],
        'features': features[:split_idx]
    }
    train_targets = residuals[:split_idx]
    
    val_data = {
        'patches': patches[split_idx:],
        'features': features[split_idx:]
    }
    val_targets = residuals[split_idx:]
    
    print(f"\nTrain samples: {len(train_targets)}")
    print(f"Validation samples: {len(val_targets)}")
    
    # Create MDA-CNN model
    print("\n" + "-" * 40)
    print("Creating MDA-CNN Model")
    print("-" * 40)
    
    model = create_mda_cnn_model(
        patch_size=(9, 9),
        n_point_features=8,
        cnn_filters=(16, 32, 64),  # Smaller for demo
        mlp_hidden_dims=(32, 32),
        fusion_hidden_dims=(64, 32),
        dropout_rate=0.2
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Build model by running a forward pass
    _ = model(train_data)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Train model
    print("\n" + "-" * 40)
    print("Training MDA-CNN Model")
    print("-" * 40)
    
    history = model.fit(
        train_data, train_targets,
        validation_data=(val_data, val_targets),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "-" * 40)
    print("Evaluating MDA-CNN Model")
    print("-" * 40)
    
    train_loss, train_mae = model.evaluate(train_data, train_targets, verbose=0)
    val_loss, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    
    print(f"Training Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
    print(f"Validation Loss: {val_loss:.6f}, MAE: {val_mae:.6f}")
    
    # Make predictions
    predictions = model.predict(val_data, verbose=0)
    
    print(f"\nPrediction statistics:")
    print(f"  Mean prediction: {np.mean(predictions):.6f}")
    print(f"  Std prediction: {np.std(predictions):.6f}")
    print(f"  Mean target: {np.mean(val_targets):.6f}")
    print(f"  Std target: {np.std(val_targets):.6f}")
    
    return model, history


def demonstrate_baseline_comparison():
    """Demonstrate baseline model comparison."""
    print("\n" + "=" * 60)
    print("Baseline Model Comparison")
    print("=" * 60)
    
    # Create smaller dataset for quick comparison
    patches, features, residuals = create_synthetic_data(n_samples=500)
    
    # Split data
    split_idx = int(0.8 * len(patches))
    
    train_patches = patches[:split_idx]
    train_features = features[:split_idx]
    train_targets = residuals[:split_idx]
    
    val_patches = patches[split_idx:]
    val_features = features[split_idx:]
    val_targets = residuals[split_idx:]
    
    # Create baseline models
    models = {
        'ResidualMLP': create_baseline_model(
            'residual_mlp',
            n_features=8,
            hidden_dims=(64, 32)
        ),
        'CNNOnly': create_baseline_model(
            'cnn_only',
            patch_size=(9, 9),
            filters=(16, 32)
        ),
        'DirectMLP': create_baseline_model(
            'direct_mlp',
            n_features=7,  # No Hagan vol
            hidden_dims=(64, 32)
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'-' * 20} {name} {'-' * 20}")
        
        # Compile model
        if name == 'DirectMLP':
            # Direct MLP predicts absolute volatility (positive)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            # Use absolute targets for direct MLP
            train_y = np.abs(train_targets) + 0.2  # Add offset to ensure positive
            val_y = np.abs(val_targets) + 0.2
        else:
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            train_y = train_targets
            val_y = val_targets
        
        # Prepare data based on model type
        if name == 'ResidualMLP':
            train_x = train_features
            val_x = val_features
        elif name == 'CNNOnly':
            train_x = train_patches
            val_x = val_patches
        else:  # DirectMLP
            train_x = train_features[:, :7]  # Exclude Hagan vol
            val_x = val_features[:, :7]
        
        # Train model
        history = model.fit(
            train_x, train_y,
            validation_data=(val_x, val_y),
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        val_loss, val_mae = model.evaluate(val_x, val_y, verbose=0)
        results[name] = {'loss': val_loss, 'mae': val_mae}
        
        print(f"Validation Loss: {val_loss:.6f}, MAE: {val_mae:.6f}")
    
    # Print comparison
    print(f"\n{'-' * 40}")
    print("Model Comparison Summary")
    print(f"{'-' * 40}")
    for name, metrics in results.items():
        print(f"{name:12}: Loss={metrics['loss']:.6f}, MAE={metrics['mae']:.6f}")


def demonstrate_custom_losses():
    """Demonstrate custom loss functions."""
    print("\n" + "=" * 60)
    print("Custom Loss Functions Demonstration")
    print("=" * 60)
    
    # Create small dataset
    patches, features, residuals = create_synthetic_data(n_samples=200)
    
    # Create MDA-CNN model
    model = create_mda_cnn_model(
        patch_size=(9, 9),
        n_point_features=8,
        cnn_filters=(8, 16),  # Very small for demo
        mlp_hidden_dims=(16, 16),
        fusion_hidden_dims=(32, 16)
    )
    
    # Test different loss functions
    loss_functions = [
        ('MSE', 'mse'),
        ('Huber', get_loss_function('huber', delta=1.0)),
        ('RelativeMSE', get_loss_function('relative_mse', epsilon=1e-6))
    ]
    
    for loss_name, loss_fn in loss_functions:
        print(f"\n{'-' * 20} {loss_name} Loss {'-' * 20}")
        
        # Create fresh model (reset weights)
        model = create_mda_cnn_model(
            patch_size=(9, 9),
            n_point_features=8,
            cnn_filters=(8, 16),
            mlp_hidden_dims=(16, 16),
            fusion_hidden_dims=(32, 16)
        )
        
        # Compile with custom loss
        model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=get_metrics(['mae', 'rmse'])
        )
        
        # Train briefly
        data = {'patches': patches, 'features': features}
        history = model.fit(
            data, residuals,
            epochs=3,
            batch_size=32,
            verbose=0
        )
        
        final_loss = history.history['loss'][-1]
        
        # Get MAE metric (name might vary)
        mae_key = None
        for key in history.history.keys():
            if 'mae' in key.lower():
                mae_key = key
                break
        
        if mae_key:
            final_mae = history.history[mae_key][-1]
            print(f"Final Loss: {final_loss:.6f}, MAE: {final_mae:.6f}")
        else:
            print(f"Final Loss: {final_loss:.6f}")


if __name__ == '__main__':
    # Run demonstrations
    try:
        # Main MDA-CNN demonstration
        model, history = demonstrate_mda_cnn()
        
        # Baseline comparison
        demonstrate_baseline_comparison()
        
        # Custom loss functions
        demonstrate_custom_losses()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()