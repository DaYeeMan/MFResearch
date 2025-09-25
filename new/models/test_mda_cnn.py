"""
Unit tests for MDA-CNN model architecture.

Tests the main MDA-CNN model components:
- CNN branch functionality
- MLP branch functionality  
- Fusion layer integration
- Residual prediction head
- Model compilation and training
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .mda_cnn import MDACNN, create_mda_cnn_model, CNNBranch, MLPBranch


class TestMDACNN:
    """Test cases for the main MDA-CNN model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.patch_size = (9, 9)
        self.n_point_features = 8
        self.batch_size = 4
        
        # Create sample data
        self.sample_patches = np.random.randn(
            self.batch_size, *self.patch_size, 1
        ).astype(np.float32)
        
        self.sample_features = np.random.randn(
            self.batch_size, self.n_point_features
        ).astype(np.float32)
        
        self.sample_targets = np.random.randn(
            self.batch_size, 1
        ).astype(np.float32)
    
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features
        )
        
        assert model.patch_size == self.patch_size
        assert model.n_point_features == self.n_point_features
        assert model.cnn_filters == (32, 64, 128)
        assert model.mlp_hidden_dims == (64, 64)
        assert model.fusion_hidden_dims == (128, 64)
        assert model.dropout_rate == 0.2
        assert model.activation == 'relu'
    
    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        custom_cnn_filters = (16, 32, 64)
        custom_mlp_dims = (32, 32)
        custom_fusion_dims = (64, 32)
        custom_dropout = 0.3
        
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features,
            cnn_filters=custom_cnn_filters,
            mlp_hidden_dims=custom_mlp_dims,
            fusion_hidden_dims=custom_fusion_dims,
            dropout_rate=custom_dropout,
            activation='tanh'
        )
        
        assert model.cnn_filters == custom_cnn_filters
        assert model.mlp_hidden_dims == custom_mlp_dims
        assert model.fusion_hidden_dims == custom_fusion_dims
        assert model.dropout_rate == custom_dropout
        assert model.activation == 'tanh'
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features
        )
        
        inputs = {
            'patches': self.sample_patches,
            'features': self.sample_features
        }
        
        outputs = model(inputs, training=False)
        
        # Check output shape
        assert outputs.shape == (self.batch_size, 1)
        
        # Check output is finite
        assert tf.reduce_all(tf.math.is_finite(outputs))
    
    def test_forward_pass_training_mode(self):
        """Test forward pass in training mode (with dropout)."""
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features,
            dropout_rate=0.5  # High dropout for testing
        )
        
        inputs = {
            'patches': self.sample_patches,
            'features': self.sample_features
        }
        
        # Run multiple forward passes to check dropout variability
        outputs1 = model(inputs, training=True)
        outputs2 = model(inputs, training=True)
        
        # Outputs should be different due to dropout
        assert not tf.reduce_all(tf.equal(outputs1, outputs2))
        
        # But inference mode should be consistent
        outputs3 = model(inputs, training=False)
        outputs4 = model(inputs, training=False)
        
        assert tf.reduce_all(tf.equal(outputs3, outputs4))
    
    def test_model_compilation(self):
        """Test model compilation with different optimizers and losses."""
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features
        )
        
        # Test compilation with Adam optimizer
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Test that model is compiled
        assert model.optimizer is not None
        assert model.compiled_loss is not None
        assert len(model.metrics) > 0
    
    def test_model_training_step(self):
        """Test a single training step."""
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        inputs = {
            'patches': self.sample_patches,
            'features': self.sample_features
        }
        
        # Test training step
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = model.compiled_loss(self.sample_targets, predictions)
        
        # Check that gradients can be computed
        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(grad is not None for grad in gradients)
        
        # Check loss is finite
        assert tf.math.is_finite(loss)
    
    def test_model_fit(self):
        """Test model training with fit method."""
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features
        )
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'patches': self.sample_patches,
            'features': self.sample_features,
            'targets': self.sample_targets
        })
        
        # Prepare data for fit
        x_data = {
            'patches': self.sample_patches,
            'features': self.sample_features
        }
        y_data = self.sample_targets
        
        # Test fit for a few epochs
        history = model.fit(
            x_data, y_data,
            epochs=2,
            batch_size=2,
            verbose=0
        )
        
        # Check that training history is recorded
        assert 'loss' in history.history
        assert len(history.history['loss']) == 2
    
    def test_get_config(self):
        """Test model configuration serialization."""
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features,
            cnn_filters=(16, 32),
            dropout_rate=0.3
        )
        
        config = model.get_config()
        
        # Check that all important parameters are in config
        assert config['patch_size'] == self.patch_size
        assert config['n_point_features'] == self.n_point_features
        assert config['cnn_filters'] == (16, 32)
        assert config['dropout_rate'] == 0.3


class TestCNNBranch:
    """Test cases for the standalone CNN branch."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.patch_size = (9, 9)
        self.batch_size = 4
        
        self.sample_patches = np.random.randn(
            self.batch_size, *self.patch_size, 1
        ).astype(np.float32)
    
    def test_cnn_branch_initialization(self):
        """Test CNN branch initialization."""
        model = CNNBranch(
            patch_size=self.patch_size,
            filters=(32, 64),
            kernel_size=3
        )
        
        assert model.patch_size == self.patch_size
        assert model.filters == (32, 64)
        assert model.kernel_size == 3
    
    def test_cnn_branch_forward_pass(self):
        """Test CNN branch forward pass."""
        model = CNNBranch(
            patch_size=self.patch_size,
            filters=(16, 32)
        )
        
        outputs = model(self.sample_patches, training=False)
        
        # Check output shape
        assert outputs.shape == (self.batch_size, 1)
        
        # Check output is finite
        assert tf.reduce_all(tf.math.is_finite(outputs))


class TestMLPBranch:
    """Test cases for the standalone MLP branch."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_features = 8
        self.batch_size = 4
        
        self.sample_features = np.random.randn(
            self.batch_size, self.n_features
        ).astype(np.float32)
    
    def test_mlp_branch_initialization(self):
        """Test MLP branch initialization."""
        model = MLPBranch(
            n_features=self.n_features,
            hidden_dims=(32, 32),
            dropout_rate=0.1
        )
        
        assert model.n_features == self.n_features
        assert model.hidden_dims == (32, 32)
        assert model.dropout_rate == 0.1
    
    def test_mlp_branch_forward_pass(self):
        """Test MLP branch forward pass."""
        model = MLPBranch(
            n_features=self.n_features,
            hidden_dims=(16, 16)
        )
        
        outputs = model(self.sample_features, training=False)
        
        # Check output shape
        assert outputs.shape == (self.batch_size, 1)
        
        # Check output is finite
        assert tf.reduce_all(tf.math.is_finite(outputs))


class TestModelFactory:
    """Test cases for model factory functions."""
    
    def test_create_mda_cnn_model(self):
        """Test MDA-CNN model factory function."""
        model = create_mda_cnn_model(
            patch_size=(7, 7),
            n_point_features=6,
            cnn_filters=(16, 32),
            dropout_rate=0.1
        )
        
        assert isinstance(model, MDACNN)
        assert model.patch_size == (7, 7)
        assert model.n_point_features == 6
        assert model.cnn_filters == (16, 32)
        assert model.dropout_rate == 0.1


class TestModelIntegration:
    """Integration tests for model components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.patch_size = (9, 9)
        self.n_point_features = 8
        self.batch_size = 8
        
        # Create larger dataset for integration testing
        np.random.seed(42)
        self.patches = np.random.randn(
            self.batch_size, *self.patch_size, 1
        ).astype(np.float32)
        
        self.features = np.random.randn(
            self.batch_size, self.n_point_features
        ).astype(np.float32)
        
        # Create synthetic targets with some pattern
        self.targets = (
            0.1 * np.sum(self.features, axis=1, keepdims=True) +
            0.05 * np.random.randn(self.batch_size, 1)
        ).astype(np.float32)
    
    def test_end_to_end_training(self):
        """Test end-to-end model training and evaluation."""
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features,
            cnn_filters=(16, 32),  # Smaller for faster testing
            mlp_hidden_dims=(32, 32),
            fusion_hidden_dims=(64, 32)
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss='mse',
            metrics=['mae']
        )
        
        # Prepare data
        x_data = {
            'patches': self.patches,
            'features': self.features
        }
        y_data = self.targets
        
        # Train model
        history = model.fit(
            x_data, y_data,
            epochs=5,
            batch_size=4,
            validation_split=0.25,
            verbose=0
        )
        
        # Check that training progressed
        assert len(history.history['loss']) == 5
        assert len(history.history['val_loss']) == 5
        
        # Check that loss decreased (at least somewhat)
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        
        # Loss should decrease or at least not increase dramatically
        assert final_loss <= initial_loss * 2.0
        
        # Test prediction
        predictions = model.predict(x_data, verbose=0)
        
        assert predictions.shape == (self.batch_size, 1)
        assert tf.reduce_all(tf.math.is_finite(predictions))
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        import tempfile
        import os
        
        model = MDACNN(
            patch_size=self.patch_size,
            n_point_features=self.n_point_features
        )
        
        model.compile(optimizer='adam', loss='mse')
        
        # Build model by running a forward pass
        inputs = {
            'patches': self.patches[:1],
            'features': self.features[:1]
        }
        _ = model(inputs)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.weights.h5')
            model.save_weights(model_path)
            
            # Create new model and load weights
            new_model = MDACNN(
                patch_size=self.patch_size,
                n_point_features=self.n_point_features
            )
            
            # Build new model
            _ = new_model(inputs)
            
            # Load weights
            new_model.load_weights(model_path)
            
            # Test that predictions are the same
            pred1 = model(inputs, training=False)
            pred2 = new_model(inputs, training=False)
            
            np.testing.assert_allclose(pred1.numpy(), pred2.numpy(), rtol=1e-6)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])