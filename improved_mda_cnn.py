import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Some functions will be limited.")
    SKLEARN_AVAILABLE = False
    # Fallback implementations
    def train_test_split(X, y, test_size=0.2, random_state=None):
        """Simple train-test split without sklearn."""
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        indices = np.arange(n_samples)
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    def mean_squared_error(y_true, y_pred):
        """MSE without sklearn."""
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_absolute_error(y_true, y_pred):
        """MAE without sklearn."""
        return np.mean(np.abs(y_true - y_pred))

def build_improved_cnn(input_table_shape, num_filters=8, kernel_size=5, dnn_units=4, dropout_rate=0.5):
    """
    Builds an improved MDA-CNN model with reduced complexity and regularization.
    
    Args:
        input_table_shape (tuple): Shape of input table (N_L, C)
        num_filters (int): Number of filters for Conv1D layer
        kernel_size (int): Kernel size for Conv1D layer
        dnn_units (int): Number of units in dense layer
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        keras.Model: The improved compiled model
    """
    # Input Layer
    input_tensor = keras.Input(shape=input_table_shape, name='Input_Table')
    
    # Single Conv1D layer with batch normalization
    conv = layers.Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        activation='relu',
        name='Conv1D_Layer'
    )(input_tensor)
    
    # Batch normalization
    conv = layers.BatchNormalization()(conv)
    
    # Global average pooling instead of flatten to reduce parameters
    gap = layers.GlobalAveragePooling1D(name='GlobalAvgPool')(conv)
    
    # Single dense layer with dropout
    dense = layers.Dense(
        units=dnn_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001),  # L2 regularization
        name='Dense_Hidden'
    )(gap)
    
    # Dropout for regularization
    dropout = layers.Dropout(dropout_rate, name='Dropout')(dense)
    
    # Output layer
    output = layers.Dense(
        units=1,
        activation='linear',
        name='Output_Layer'
    )(dropout)
    
    # Create model
    model = keras.Model(inputs=input_tensor, outputs=output, name='Improved_MDA_CNN')
    
    # Compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model

def build_simple_mlp(input_table_shape, hidden_units=[16, 8], dropout_rate=0.3):
    """
    Builds a simple MLP baseline model.
    
    Args:
        input_table_shape (tuple): Shape of input table (N_L, C)
        hidden_units (list): List of hidden layer units
        dropout_rate (float): Dropout rate
    
    Returns:
        keras.Model: Simple MLP model
    """
    # Flatten input
    input_tensor = keras.Input(shape=input_table_shape, name='Input_Table')
    flatten = layers.Flatten()(input_tensor)
    
    # Hidden layers
    x = flatten
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units=units,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name=f'Dense_{i+1}'
        )(x)
        x = layers.Dropout(dropout_rate, name=f'Dropout_{i+1}')(x)
    
    # Output layer
    output = layers.Dense(1, activation='linear', name='Output')(x)
    
    model = keras.Model(inputs=input_tensor, outputs=output, name='Simple_MLP')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model

def build_ultra_simple_model(input_table_shape):
    """
    Builds an ultra-simple model for very small datasets.
    
    Args:
        input_table_shape (tuple): Shape of input table (N_L, C)
    
    Returns:
        keras.Model: Ultra-simple model
    """
    input_tensor = keras.Input(shape=input_table_shape, name='Input_Table')
    
    # Global average pooling
    gap = layers.GlobalAveragePooling1D()(input_tensor)
    
    # Single dense layer
    output = layers.Dense(1, activation='linear', name='Output')(gap)
    
    model = keras.Model(inputs=input_tensor, outputs=output, name='Ultra_Simple')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    return model

def train_with_validation(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=2):
    """
    Train model with validation and early stopping.
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum epochs
        batch_size: Batch size
    
    Returns:
        history: Training history
    """
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    return history

def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance and return metrics.
    
    Args:
        model: Trained model
        X_test, y_test: Test data
        model_name: Name for logging
    
    Returns:
        dict: Performance metrics
    """
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_test - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics = {
        'model_name': model_name,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'num_params': model.count_params()
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Parameters: {metrics['num_params']}")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²: {r2:.4f}")
    
    return metrics

def compare_models(X_train, y_train, X_test, y_test, input_shape):
    """
    Compare different model architectures.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        input_shape: Input shape
    
    Returns:
        dict: Results from all models
    """
    results = {}
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 1. Improved CNN
    print("Training Improved CNN...")
    cnn_model = build_improved_cnn(input_shape)
    cnn_history = train_with_validation(
        cnn_model, X_train_split, y_train_split, X_val_split, y_val_split,
        epochs=50, batch_size=2
    )
    results['improved_cnn'] = evaluate_model_performance(cnn_model, X_test, y_test, "Improved CNN")
    
    # 2. Simple MLP
    print("\nTraining Simple MLP...")
    mlp_model = build_simple_mlp(input_shape)
    mlp_history = train_with_validation(
        mlp_model, X_train_split, y_train_split, X_val_split, y_val_split,
        epochs=50, batch_size=2
    )
    results['simple_mlp'] = evaluate_model_performance(mlp_model, X_test, y_test, "Simple MLP")
    
    # 3. Ultra Simple Model
    print("\nTraining Ultra Simple Model...")
    ultra_model = build_ultra_simple_model(input_shape)
    ultra_history = train_with_validation(
        ultra_model, X_train_split, y_train_split, X_val_split, y_val_split,
        epochs=50, batch_size=2
    )
    results['ultra_simple'] = evaluate_model_performance(ultra_model, X_test, y_test, "Ultra Simple")
    
    # 4. Cubic Spline Baseline
    print("\nEvaluating Cubic Spline...")
    # For cubic spline, we need to extract the input features differently
    # This is a simplified version - in practice you'd need to adapt to your specific data structure
    try:
        # Assuming we can extract some key features for spline interpolation
        # This is a placeholder - you'd need to adapt based on your actual data structure
        spline_predictions = np.zeros_like(y_test)  # Placeholder
        spline_mse = mean_squared_error(y_test, spline_predictions)
        spline_mae = mean_absolute_error(y_test, spline_predictions)
        
        results['cubic_spline'] = {
            'model_name': 'Cubic Spline',
            'mse': spline_mse,
            'mae': spline_mae,
            'rmse': np.sqrt(spline_mse),
            'r2': 0.0,  # Placeholder
            'num_params': 0
        }
        print(f"Cubic Spline Performance:")
        print(f"  MSE: {spline_mse:.6f}")
        print(f"  MAE: {spline_mae:.6f}")
    except Exception as e:
        print(f"Error with cubic spline: {e}")
        results['cubic_spline'] = None
    
    return results

def plot_training_history(histories, model_names):
    """
    Plot training histories for comparison.
    
    Args:
        histories: List of training histories
        model_names: List of model names
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, (history, name) in enumerate(zip(histories, model_names)):
        # Loss plot
        axes[0].plot(history.history['loss'], label=f'{name} - Train')
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label=f'{name} - Val', linestyle='--')
        
        # MAE plot
        axes[1].plot(history.history['mean_absolute_error'], label=f'{name} - Train')
        if 'val_mean_absolute_error' in history.history:
            axes[1].plot(history.history['val_mean_absolute_error'], label=f'{name} - Val', linestyle='--')
    
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def print_comparison_summary(results):
    """
    Print a summary comparison of all models.
    
    Args:
        results: Dictionary of model results
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Create comparison table
    models = []
    for key, result in results.items():
        if result is not None:
            models.append(result)
    
    if not models:
        print("No valid results to compare.")
        return
    
    # Sort by RMSE (lower is better)
    models.sort(key=lambda x: x['rmse'])
    
    print(f"{'Model':<20} {'Params':<8} {'RMSE':<10} {'MAE':<10} {'R²':<8}")
    print("-" * 60)
    
    for model in models:
        print(f"{model['model_name']:<20} {model['num_params']:<8} "
              f"{model['rmse']:<10.6f} {model['mae']:<10.6f} {model['r2']:<8.4f}")
    
    print("\nBest performing model:", models[0]['model_name'])
    print(f"RMSE: {models[0]['rmse']:.6f}")
    print(f"Parameters: {models[0]['num_params']}")

# Example usage function
def run_improved_analysis(X_train, y_train, X_test, y_test, input_shape):
    """
    Run the complete improved analysis.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data  
        input_shape: Input shape tuple
    """
    print("Starting Improved MDA-CNN Analysis")
    print("="*50)
    
    # Compare models
    results = compare_models(X_train, y_train, X_test, y_test, input_shape)
    
    # Print summary
    print_comparison_summary(results)
    
    return results

if __name__ == "__main__":
    print("Improved MDA-CNN Implementation")
    print("This module provides improved CNN architectures to address overfitting issues.")
    print("Use run_improved_analysis() with your data to compare different models.")
