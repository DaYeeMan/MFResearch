"""
Unit tests for custom loss functions and metrics.

Tests the custom loss functions and metrics:
- WeightedMSE
- RelativeMSE  
- HuberLoss
- QuantileLoss
- Custom metrics (RMSE, MAPE, R2Score)
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

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


class TestWeightedMSE:
    """Test cases for WeightedMSE loss function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.y_true = tf.constant([[1.0], [2.0], [3.0], [4.0]], dtype=tf.float32)
        self.y_pred = tf.constant([[1.1], [1.9], [3.2], [3.8]], dtype=tf.float32)
    
    def test_initialization_no_weights(self):
        """Test WeightedMSE initialization without weight function."""
        loss_fn = WeightedMSE()
        
        loss = loss_fn(self.y_true, self.y_pred)
        
        # Should be equivalent to regular MSE
        mse = keras.losses.MeanSquaredError()
        expected_loss = mse(self.y_true, self.y_pred)
        
        np.testing.assert_allclose(loss.numpy(), expected_loss.numpy())
    
    def test_initialization_with_weights(self):
        """Test WeightedMSE with custom weight function."""
        def weight_fn(y_true, y_pred):
            # Higher weight for larger true values
            return y_true / tf.reduce_max(y_true)
        
        loss_fn = WeightedMSE(weight_fn=weight_fn)
        
        loss = loss_fn(self.y_true, self.y_pred)
        
        # Should be different from regular MSE
        mse = keras.losses.MeanSquaredError()
        regular_loss = mse(self.y_true, self.y_pred)
        
        assert not np.allclose(loss.numpy(), regular_loss.numpy())
        assert tf.math.is_finite(loss)


class TestRelativeMSE:
    """Test cases for RelativeMSE loss function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.y_true = tf.constant([[1.0], [2.0], [0.1], [10.0]], dtype=tf.float32)
        self.y_pred = tf.constant([[1.1], [1.8], [0.11], [9.5]], dtype=tf.float32)
    
    def test_initialization(self):
        """Test RelativeMSE initialization."""
        loss_fn = RelativeMSE(epsilon=1e-6)
        
        loss = loss_fn(self.y_true, self.y_pred)
        
        assert tf.math.is_finite(loss)
        assert loss >= 0
    
    def test_relative_scaling(self):
        """Test that RelativeMSE scales errors relative to true values."""
        loss_fn = RelativeMSE()
        
        # Small true values should have larger relative errors
        y_true_small = tf.constant([[0.1], [0.2]], dtype=tf.float32)
        y_pred_small = tf.constant([[0.11], [0.21]], dtype=tf.float32)
        
        # Large true values should have smaller relative errors
        y_true_large = tf.constant([[10.0], [20.0]], dtype=tf.float32)
        y_pred_large = tf.constant([[10.1], [20.1]], dtype=tf.float32)
        
        loss_small = loss_fn(y_true_small, y_pred_small)
        loss_large = loss_fn(y_true_large, y_pred_large)
        
        # Relative error should be larger for small values
        assert loss_small > loss_large
    
    def test_epsilon_handling(self):
        """Test epsilon handling for zero true values."""
        loss_fn = RelativeMSE(epsilon=1e-8)
        
        y_true_zero = tf.constant([[0.0], [0.0]], dtype=tf.float32)
        y_pred_zero = tf.constant([[0.1], [0.2]], dtype=tf.float32)
        
        loss = loss_fn(y_true_zero, y_pred_zero)
        
        assert tf.math.is_finite(loss)
        assert loss >= 0


class TestHuberLoss:
    """Test cases for HuberLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.y_true = tf.constant([[1.0], [2.0], [3.0], [4.0]], dtype=tf.float32)
        self.y_pred = tf.constant([[1.1], [1.5], [5.0], [3.8]], dtype=tf.float32)  # One outlier
    
    def test_initialization(self):
        """Test HuberLoss initialization."""
        loss_fn = HuberLoss(delta=1.0)
        
        loss = loss_fn(self.y_true, self.y_pred)
        
        assert tf.math.is_finite(loss)
        assert loss >= 0
    
    def test_delta_parameter(self):
        """Test different delta values."""
        loss_fn_small = HuberLoss(delta=0.5)
        loss_fn_large = HuberLoss(delta=2.0)
        
        loss_small = loss_fn_small(self.y_true, self.y_pred)
        loss_large = loss_fn_large(self.y_true, self.y_pred)
        
        # Both should be finite
        assert tf.math.is_finite(loss_small)
        assert tf.math.is_finite(loss_large)
    
    def test_outlier_robustness(self):
        """Test that Huber loss is more robust to outliers than MSE."""
        # Create data with outlier
        y_true = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)
        y_pred_normal = tf.constant([[1.1], [2.1], [3.1]], dtype=tf.float32)
        y_pred_outlier = tf.constant([[1.1], [2.1], [10.0]], dtype=tf.float32)  # Outlier
        
        huber_fn = HuberLoss(delta=1.0)
        mse_fn = keras.losses.MeanSquaredError()
        
        # Normal case
        huber_normal = huber_fn(y_true, y_pred_normal)
        mse_normal = mse_fn(y_true, y_pred_normal)
        
        # Outlier case
        huber_outlier = huber_fn(y_true, y_pred_outlier)
        mse_outlier = mse_fn(y_true, y_pred_outlier)
        
        # Huber should be less affected by outlier than MSE
        huber_ratio = huber_outlier / huber_normal
        mse_ratio = mse_outlier / mse_normal
        
        assert huber_ratio < mse_ratio


class TestQuantileLoss:
    """Test cases for QuantileLoss."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.y_true = tf.constant([[1.0], [2.0], [3.0], [4.0]], dtype=tf.float32)
        self.y_pred = tf.constant([[0.8], [2.2], [2.8], [4.2]], dtype=tf.float32)
    
    def test_median_quantile(self):
        """Test quantile loss for median (0.5 quantile)."""
        loss_fn = QuantileLoss(quantile=0.5)
        
        loss = loss_fn(self.y_true, self.y_pred)
        
        assert tf.math.is_finite(loss)
        assert loss >= 0
    
    def test_different_quantiles(self):
        """Test different quantile values."""
        loss_fn_25 = QuantileLoss(quantile=0.25)
        loss_fn_75 = QuantileLoss(quantile=0.75)
        
        loss_25 = loss_fn_25(self.y_true, self.y_pred)
        loss_75 = loss_fn_75(self.y_true, self.y_pred)
        
        assert tf.math.is_finite(loss_25)
        assert tf.math.is_finite(loss_75)
        assert loss_25 >= 0
        assert loss_75 >= 0
    
    def test_asymmetric_penalty(self):
        """Test asymmetric penalty for over/under-prediction."""
        loss_fn_low = QuantileLoss(quantile=0.1)  # Penalizes over-prediction more
        loss_fn_high = QuantileLoss(quantile=0.9)  # Penalizes under-prediction more
        
        # Over-prediction case
        y_true_over = tf.constant([[1.0]], dtype=tf.float32)
        y_pred_over = tf.constant([[1.5]], dtype=tf.float32)
        
        # Under-prediction case  
        y_true_under = tf.constant([[1.0]], dtype=tf.float32)
        y_pred_under = tf.constant([[0.5]], dtype=tf.float32)
        
        loss_low_over = loss_fn_low(y_true_over, y_pred_over)
        loss_low_under = loss_fn_low(y_true_under, y_pred_under)
        
        loss_high_over = loss_fn_high(y_true_over, y_pred_over)
        loss_high_under = loss_fn_high(y_true_under, y_pred_under)
        
        # Low quantile should penalize over-prediction more
        assert loss_low_over > loss_low_under
        
        # High quantile should penalize under-prediction more
        assert loss_high_under > loss_high_over


class TestRootMeanSquaredError:
    """Test cases for custom RMSE metric."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metric = RootMeanSquaredError()
        self.y_true = tf.constant([[1.0], [2.0], [3.0], [4.0]], dtype=tf.float32)
        self.y_pred = tf.constant([[1.1], [1.9], [3.2], [3.8]], dtype=tf.float32)
    
    def test_metric_computation(self):
        """Test RMSE metric computation."""
        self.metric.update_state(self.y_true, self.y_pred)
        result = self.metric.result()
        
        # Compute expected RMSE
        mse = tf.reduce_mean(tf.square(self.y_true - self.y_pred))
        expected_rmse = tf.sqrt(mse)
        
        np.testing.assert_allclose(result.numpy(), expected_rmse.numpy())
    
    def test_metric_reset(self):
        """Test metric state reset."""
        self.metric.update_state(self.y_true, self.y_pred)
        result1 = self.metric.result()
        
        self.metric.reset_state()
        result2 = self.metric.result()
        
        # After reset, result should be close to 0 (due to epsilon handling)
        assert result2.numpy() < 1e-6
        assert result1.numpy() > 0.0


class TestMeanAbsolutePercentageError:
    """Test cases for custom MAPE metric."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metric = MeanAbsolutePercentageError()
        self.y_true = tf.constant([[1.0], [2.0], [4.0], [5.0]], dtype=tf.float32)
        self.y_pred = tf.constant([[1.1], [1.8], [4.2], [4.5]], dtype=tf.float32)
    
    def test_metric_computation(self):
        """Test MAPE metric computation."""
        self.metric.update_state(self.y_true, self.y_pred)
        result = self.metric.result()
        
        # Should be a percentage
        assert result >= 0
        assert result <= 100  # For reasonable predictions
        assert tf.math.is_finite(result)
    
    def test_epsilon_handling(self):
        """Test epsilon handling for small true values."""
        metric = MeanAbsolutePercentageError(epsilon=1e-6)
        
        y_true_small = tf.constant([[1e-8], [2e-8]], dtype=tf.float32)
        y_pred_small = tf.constant([[1.1e-8], [1.8e-8]], dtype=tf.float32)
        
        metric.update_state(y_true_small, y_pred_small)
        result = metric.result()
        
        assert tf.math.is_finite(result)


class TestWingWeightFunction:
    """Test cases for wing weight function."""
    
    def test_weight_function_creation(self):
        """Test creating wing weight function."""
        weight_fn = create_wing_weight_function(
            atm_weight=1.0,
            wing_weight=2.0,
            wing_threshold=0.1
        )
        
        # Test with sample data
        y_true = tf.constant([[0.05], [0.15], [0.02], [0.25]], dtype=tf.float32)
        y_pred = tf.constant([[0.06], [0.14], [0.03], [0.24]], dtype=tf.float32)
        
        weights = weight_fn(y_true, y_pred)
        
        assert weights.shape == y_true.shape
        assert tf.reduce_all(weights > 0)


class TestLossFactoryFunctions:
    """Test cases for loss factory functions."""
    
    def test_get_loss_function_mse(self):
        """Test getting MSE loss function."""
        loss_fn = get_loss_function('mse')
        
        assert isinstance(loss_fn, keras.losses.MeanSquaredError)
    
    def test_get_loss_function_custom(self):
        """Test getting custom loss functions."""
        loss_fn = get_loss_function('weighted_mse')
        assert isinstance(loss_fn, WeightedMSE)
        
        loss_fn = get_loss_function('relative_mse', epsilon=1e-6)
        assert isinstance(loss_fn, RelativeMSE)
        
        loss_fn = get_loss_function('huber', delta=1.5)
        assert isinstance(loss_fn, HuberLoss)
        
        loss_fn = get_loss_function('quantile', quantile=0.25)
        assert isinstance(loss_fn, QuantileLoss)
    
    def test_get_loss_function_unknown(self):
        """Test error for unknown loss function."""
        with pytest.raises(ValueError):
            get_loss_function('unknown_loss')
    
    def test_get_metrics(self):
        """Test getting metrics list."""
        metrics = get_metrics(['mse', 'mae', 'rmse'])
        
        assert len(metrics) == 3
        assert isinstance(metrics[0], keras.metrics.MeanSquaredError)
        assert isinstance(metrics[1], keras.metrics.MeanAbsoluteError)
        assert isinstance(metrics[2], RootMeanSquaredError)
    
    def test_get_metrics_unknown(self):
        """Test handling unknown metrics."""
        # Should ignore unknown metrics and print warning
        metrics = get_metrics(['mse', 'unknown_metric', 'mae'])
        
        assert len(metrics) == 2  # Only known metrics


class TestLossIntegration:
    """Integration tests for loss functions with models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.n_features = 4
        
        # Create simple model
        self.model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(self.n_features,)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])
        
        # Create sample data
        np.random.seed(42)
        self.x_data = np.random.randn(self.batch_size, self.n_features).astype(np.float32)
        self.y_data = np.random.randn(self.batch_size, 1).astype(np.float32)
    
    def test_model_training_with_custom_losses(self):
        """Test training model with custom loss functions."""
        custom_losses = [
            ('weighted_mse', WeightedMSE()),
            ('relative_mse', RelativeMSE()),
            ('huber', HuberLoss(delta=1.0)),
            ('quantile', QuantileLoss(quantile=0.5))
        ]
        
        for loss_name, loss_fn in custom_losses:
            # Create fresh model
            model = keras.models.clone_model(self.model)
            model.build(input_shape=(None, self.n_features))
            
            # Compile with custom loss
            model.compile(
                optimizer='adam',
                loss=loss_fn,
                metrics=['mae']
            )
            
            # Train for a few epochs
            history = model.fit(
                self.x_data, self.y_data,
                epochs=3,
                batch_size=4,
                verbose=0
            )
            
            # Check training progressed
            assert len(history.history['loss']) == 3
            assert all(np.isfinite(loss) for loss in history.history['loss'])
            
            # Test prediction
            predictions = model.predict(self.x_data, verbose=0)
            assert predictions.shape == (self.batch_size, 1)
            assert np.all(np.isfinite(predictions))


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])