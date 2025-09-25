"""
Unit tests for baseline model architectures.

Tests the baseline models:
- DirectMLP: point features → absolute volatility
- ResidualMLP: point features → residual (no patches)
- CNNOnly: patches → residual (no point features)
- EnsembleModel: combination of multiple models
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .baseline_models import (
    DirectMLP, 
    ResidualMLP, 
    CNNOnly, 
    create_baseline_model,
    EnsembleModel
)


class TestDirectMLP:
    """Test cases for DirectMLP baseline model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 7  # SABR params + strike + maturity (no Hagan vol)
        self.batch_size = 4
        
        self.sample_features = np.random.randn(
            self.batch_size, self.n_features
        ).astype(np.float32)
        
        # Positive targets for volatility
        self.sample_targets = np.abs(np.random.randn(
            self.batch_size, 1
        )).astype(np.float32)
    
    def test_initialization(self):
        """Test DirectMLP initialization."""
        model = DirectMLP(
            n_features=self.n_features,
            hidden_dims=(64, 32),
            dropout_rate=0.1
        )
        
        assert model.n_features == self.n_features
        assert model.hidden_dims == (64, 32)
        assert model.dropout_rate == 0.1
        assert model.activation == 'relu'
    
    def test_forward_pass(self):
        """Test forward pass through DirectMLP."""
        model = DirectMLP(
            n_features=self.n_features,
            hidden_dims=(32, 16)
        )
        
        outputs = model(self.sample_features, training=False)
        
        # Check output shape
        assert outputs.shape == (self.batch_size, 1)
        
        # Check output is finite and non-negative (ReLU activation)
        assert tf.reduce_all(tf.math.is_finite(outputs))
        assert tf.reduce_all(outputs >= 0)
    
    def test_training(self):
        """Test DirectMLP training."""
        model = DirectMLP(
            n_features=self.n_features,
            hidden_dims=(32, 16)
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train for a few epochs
        history = model.fit(
            self.sample_features, self.sample_targets,
            epochs=3,
            batch_size=2,
            verbose=0
        )
        
        assert 'loss' in history.history
        assert len(history.history['loss']) == 3


class TestResidualMLP:
    """Test cases for ResidualMLP baseline model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 8  # SABR params + strike + maturity + Hagan vol
        self.batch_size = 4
        
        self.sample_features = np.random.randn(
            self.batch_size, self.n_features
        ).astype(np.float32)
        
        # Residual targets can be positive or negative
        self.sample_targets = np.random.randn(
            self.batch_size, 1
        ).astype(np.float32)
    
    def test_initialization(self):
        """Test ResidualMLP initialization."""
        model = ResidualMLP(
            n_features=self.n_features,
            hidden_dims=(64, 32),
            dropout_rate=0.2
        )
        
        assert model.n_features == self.n_features
        assert model.hidden_dims == (64, 32)
        assert model.dropout_rate == 0.2
    
    def test_forward_pass(self):
        """Test forward pass through ResidualMLP."""
        model = ResidualMLP(
            n_features=self.n_features,
            hidden_dims=(32, 16)
        )
        
        outputs = model(self.sample_features, training=False)
        
        # Check output shape
        assert outputs.shape == (self.batch_size, 1)
        
        # Check output is finite (can be positive or negative)
        assert tf.reduce_all(tf.math.is_finite(outputs))
    
    def test_dropout_behavior(self):
        """Test dropout behavior in training vs inference."""
        model = ResidualMLP(
            n_features=self.n_features,
            hidden_dims=(32, 16),
            dropout_rate=0.5  # High dropout for testing
        )
        
        # Training mode should have variability due to dropout
        outputs1 = model(self.sample_features, training=True)
        outputs2 = model(self.sample_features, training=True)
        
        # Should be different due to dropout
        assert not tf.reduce_all(tf.equal(outputs1, outputs2))
        
        # Inference mode should be consistent
        outputs3 = model(self.sample_features, training=False)
        outputs4 = model(self.sample_features, training=False)
        
        assert tf.reduce_all(tf.equal(outputs3, outputs4))


class TestCNNOnly:
    """Test cases for CNNOnly baseline model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.patch_size = (9, 9)
        self.batch_size = 4
        
        self.sample_patches = np.random.randn(
            self.batch_size, *self.patch_size, 1
        ).astype(np.float32)
        
        self.sample_targets = np.random.randn(
            self.batch_size, 1
        ).astype(np.float32)
    
    def test_initialization(self):
        """Test CNNOnly initialization."""
        model = CNNOnly(
            patch_size=self.patch_size,
            filters=(16, 32),
            kernel_size=3,
            dropout_rate=0.1
        )
        
        assert model.patch_size == self.patch_size
        assert model.filters == (16, 32)
        assert model.kernel_size == 3
        assert model.dropout_rate == 0.1
    
    def test_forward_pass(self):
        """Test forward pass through CNNOnly."""
        model = CNNOnly(
            patch_size=self.patch_size,
            filters=(16, 32)
        )
        
        outputs = model(self.sample_patches, training=False)
        
        # Check output shape
        assert outputs.shape == (self.batch_size, 1)
        
        # Check output is finite
        assert tf.reduce_all(tf.math.is_finite(outputs))
    
    def test_training(self):
        """Test CNNOnly training."""
        model = CNNOnly(
            patch_size=self.patch_size,
            filters=(8, 16)  # Small for fast testing
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train for a few epochs
        history = model.fit(
            self.sample_patches, self.sample_targets,
            epochs=3,
            batch_size=2,
            verbose=0
        )
        
        assert 'loss' in history.history
        assert len(history.history['loss']) == 3


class TestEnsembleModel:
    """Test cases for EnsembleModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 8
        self.batch_size = 4
        
        self.sample_features = np.random.randn(
            self.batch_size, self.n_features
        ).astype(np.float32)
        
        # Create simple models for ensemble
        self.model1 = ResidualMLP(
            n_features=self.n_features,
            hidden_dims=(16, 8)
        )
        
        self.model2 = ResidualMLP(
            n_features=self.n_features,
            hidden_dims=(8, 4)
        )
    
    def test_initialization_equal_weights(self):
        """Test ensemble initialization with equal weights."""
        ensemble = EnsembleModel(
            models=[self.model1, self.model2]
        )
        
        assert len(ensemble.models) == 2
        assert ensemble.n_models == 2
        assert ensemble.model_weights == [0.5, 0.5]
    
    def test_initialization_custom_weights(self):
        """Test ensemble initialization with custom weights."""
        custom_weights = [0.7, 0.3]
        ensemble = EnsembleModel(
            models=[self.model1, self.model2],
            weights=custom_weights
        )
        
        assert ensemble.model_weights == custom_weights
    
    def test_initialization_weight_normalization(self):
        """Test that weights are normalized."""
        unnormalized_weights = [2.0, 1.0]
        ensemble = EnsembleModel(
            models=[self.model1, self.model2],
            weights=unnormalized_weights
        )
        
        # Should be normalized to [2/3, 1/3]
        expected_weights = [2.0/3.0, 1.0/3.0]
        np.testing.assert_allclose(ensemble.model_weights, expected_weights)
    
    def test_initialization_weight_mismatch(self):
        """Test error when number of weights doesn't match number of models."""
        with pytest.raises(ValueError):
            EnsembleModel(
                models=[self.model1, self.model2],
                weights=[0.5, 0.3, 0.2]  # 3 weights for 2 models
            )
    
    def test_forward_pass(self):
        """Test forward pass through ensemble."""
        ensemble = EnsembleModel(
            models=[self.model1, self.model2],
            weights=[0.6, 0.4]
        )
        
        outputs = ensemble(self.sample_features, training=False)
        
        # Check output shape
        assert outputs.shape == (self.batch_size, 1)
        
        # Check output is finite
        assert tf.reduce_all(tf.math.is_finite(outputs))
        
        # Verify it's actually a weighted combination
        pred1 = self.model1(self.sample_features, training=False)
        pred2 = self.model2(self.sample_features, training=False)
        expected = 0.6 * pred1 + 0.4 * pred2
        
        np.testing.assert_allclose(outputs.numpy(), expected.numpy(), rtol=1e-6)


class TestModelFactory:
    """Test cases for baseline model factory function."""
    
    def test_create_direct_mlp(self):
        """Test creating DirectMLP via factory."""
        model = create_baseline_model(
            'direct_mlp',
            n_features=7,
            hidden_dims=(32, 16)
        )
        
        assert isinstance(model, DirectMLP)
        assert model.n_features == 7
        assert model.hidden_dims == (32, 16)
    
    def test_create_residual_mlp(self):
        """Test creating ResidualMLP via factory."""
        model = create_baseline_model(
            'residual_mlp',
            n_features=8,
            hidden_dims=(64, 32)
        )
        
        assert isinstance(model, ResidualMLP)
        assert model.n_features == 8
        assert model.hidden_dims == (64, 32)
    
    def test_create_cnn_only(self):
        """Test creating CNNOnly via factory."""
        model = create_baseline_model(
            'cnn_only',
            patch_size=(7, 7),
            filters=(16, 32)
        )
        
        assert isinstance(model, CNNOnly)
        assert model.patch_size == (7, 7)
        assert model.filters == (16, 32)
    
    def test_create_unknown_model(self):
        """Test error for unknown model type."""
        with pytest.raises(ValueError):
            create_baseline_model('unknown_model')


class TestBaselineIntegration:
    """Integration tests for baseline models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 8
        self.patch_size = (9, 9)
        self.batch_size = 8
        
        # Create synthetic data
        np.random.seed(42)
        self.features = np.random.randn(
            self.batch_size, self.n_features
        ).astype(np.float32)
        
        self.patches = np.random.randn(
            self.batch_size, *self.patch_size, 1
        ).astype(np.float32)
        
        # Create targets with some pattern
        self.residual_targets = (
            0.1 * np.sum(self.features, axis=1, keepdims=True) +
            0.05 * np.random.randn(self.batch_size, 1)
        ).astype(np.float32)
        
        self.volatility_targets = np.abs(self.residual_targets) + 0.2
    
    def test_all_models_training(self):
        """Test that all baseline models can train successfully."""
        models = {
            'direct_mlp': DirectMLP(n_features=7, hidden_dims=(16, 8)),
            'residual_mlp': ResidualMLP(n_features=8, hidden_dims=(16, 8)),
            'cnn_only': CNNOnly(patch_size=self.patch_size, filters=(8, 16))
        }
        
        for name, model in models.items():
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            if name == 'direct_mlp':
                x_data = self.features[:, :7]  # Exclude Hagan vol
                y_data = self.volatility_targets
            elif name == 'residual_mlp':
                x_data = self.features
                y_data = self.residual_targets
            else:  # cnn_only
                x_data = self.patches
                y_data = self.residual_targets
            
            # Train for a few epochs
            history = model.fit(
                x_data, y_data,
                epochs=3,
                batch_size=4,
                verbose=0
            )
            
            # Check training progressed
            assert len(history.history['loss']) == 3
            
            # Check predictions work
            predictions = model.predict(x_data, verbose=0)
            assert predictions.shape == (self.batch_size, 1)
            assert tf.reduce_all(tf.math.is_finite(predictions))
    
    def test_model_comparison(self):
        """Test comparing different baseline models."""
        # Create models
        mlp_model = ResidualMLP(n_features=8, hidden_dims=(32, 16))
        cnn_model = CNNOnly(patch_size=self.patch_size, filters=(16, 32))
        
        # Compile models
        for model in [mlp_model, cnn_model]:
            model.compile(optimizer='adam', loss='mse')
        
        # Train models
        mlp_history = mlp_model.fit(
            self.features, self.residual_targets,
            epochs=5, batch_size=4, verbose=0
        )
        
        cnn_history = cnn_model.fit(
            self.patches, self.residual_targets,
            epochs=5, batch_size=4, verbose=0
        )
        
        # Get final losses
        mlp_loss = mlp_history.history['loss'][-1]
        cnn_loss = cnn_history.history['loss'][-1]
        
        # Both should have finite losses
        assert np.isfinite(mlp_loss)
        assert np.isfinite(cnn_loss)
        
        # Test predictions
        mlp_pred = mlp_model.predict(self.features, verbose=0)
        cnn_pred = cnn_model.predict(self.patches, verbose=0)
        
        assert mlp_pred.shape == (self.batch_size, 1)
        assert cnn_pred.shape == (self.batch_size, 1)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])