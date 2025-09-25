"""
Tests for data normalization and scaling utilities.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from .normalization import (
    NormalizationStats, PatchNormalizer, FeatureScaler, create_data_splits
)


class TestPatchNormalizer:
    """Test patch normalization functionality."""
    
    def create_test_patches(self, n_samples: int = 100, height: int = 9, width: int = 9) -> np.ndarray:
        """Create test patches with known statistics."""
        # Create patches with different characteristics
        patches = np.zeros((n_samples, height, width))
        
        for i in range(n_samples):
            # Create patches with varying means and scales
            base_value = i * 0.1
            noise_scale = 0.5 + i * 0.01
            
            patch = np.random.normal(base_value, noise_scale, (height, width))
            patches[i] = patch
        
        return patches.astype(np.float32)
    
    def test_local_normalization(self):
        """Test local patch normalization."""
        patches = self.create_test_patches(50, 9, 9)
        
        normalizer = PatchNormalizer(method='local', robust=False)
        normalized_patches = normalizer.transform(patches)
        
        # Check that each patch is individually normalized
        for i in range(len(patches)):
            patch = normalized_patches[i]
            finite_mask = np.isfinite(patch)
            
            if np.sum(finite_mask) > 1:
                patch_mean = np.mean(patch[finite_mask])
                patch_std = np.std(patch[finite_mask])
                
                # Should be approximately normalized
                assert abs(patch_mean) < 1e-6
                assert abs(patch_std - 1.0) < 1e-6
    
    def test_global_normalization(self):
        """Test global patch normalization."""
        patches = self.create_test_patches(50, 9, 9)
        
        normalizer = PatchNormalizer(method='global', robust=False)
        normalizer.fit(patches)
        normalized_patches = normalizer.transform(patches)
        
        # Check global statistics
        finite_mask = np.isfinite(normalized_patches)
        finite_values = normalized_patches[finite_mask]
        
        global_mean = np.mean(finite_values)
        global_std = np.std(finite_values)
        
        # Should be approximately normalized globally
        assert abs(global_mean) < 0.1
        assert abs(global_std - 1.0) < 0.1
    
    def test_standardize_normalization(self):
        """Test per-pixel standardization."""
        patches = self.create_test_patches(50, 5, 5)  # Smaller for easier testing
        
        normalizer = PatchNormalizer(method='standardize', robust=False)
        normalizer.fit(patches)
        normalized_patches = normalizer.transform(patches)
        
        # Check per-pixel normalization
        for i in range(5):
            for j in range(5):
                pixel_values = normalized_patches[:, i, j]
                finite_mask = np.isfinite(pixel_values)
                
                if np.sum(finite_mask) > 1:
                    pixel_mean = np.mean(pixel_values[finite_mask])
                    pixel_std = np.std(pixel_values[finite_mask])
                    
                    # Should be normalized per pixel
                    assert abs(pixel_mean) < 0.1
                    assert abs(pixel_std - 1.0) < 0.1
    
    def test_robust_normalization(self):
        """Test robust normalization using median and IQR."""
        # Create patches with outliers
        patches = np.random.normal(0, 1, (50, 9, 9))
        
        # Add outliers
        patches[0, 0, 0] = 100  # Large outlier
        patches[1, 1, 1] = -100  # Large negative outlier
        
        # Test robust vs non-robust
        normalizer_robust = PatchNormalizer(method='global', robust=True)
        normalizer_standard = PatchNormalizer(method='global', robust=False)
        
        normalizer_robust.fit(patches)
        normalizer_standard.fit(patches)
        
        normalized_robust = normalizer_robust.transform(patches)
        normalized_standard = normalizer_standard.transform(patches)
        
        # Robust normalization should be less affected by outliers
        robust_std = np.std(normalized_robust[np.isfinite(normalized_robust)])
        standard_std = np.std(normalized_standard[np.isfinite(normalized_standard)])
        
        # This is a heuristic test - robust should have more reasonable scale
        assert robust_std < standard_std * 2
    
    def test_outlier_clipping(self):
        """Test outlier clipping functionality."""
        patches = np.random.normal(0, 1, (50, 9, 9))
        
        # Add extreme outliers
        patches[0, 0, 0] = 50
        patches[1, 1, 1] = -50
        
        normalizer = PatchNormalizer(
            method='local', 
            clip_outliers=True, 
            outlier_percentiles=(5, 95)
        )
        
        normalized_patches = normalizer.transform(patches)
        
        # Check that extreme values are clipped
        finite_values = normalized_patches[np.isfinite(normalized_patches)]
        assert np.max(finite_values) < 10  # Should be clipped
        assert np.min(finite_values) > -10  # Should be clipped
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        patches = np.random.normal(0, 1, (20, 9, 9))
        
        # Add NaN values
        patches[0, 0, 0] = np.nan
        patches[1, :, :] = np.nan  # Entire patch
        patches[2, 4, 4] = np.inf
        
        normalizer = PatchNormalizer(method='local')
        normalized_patches = normalizer.transform(patches)
        
        # NaN values should be replaced with 0
        assert np.sum(np.isnan(normalized_patches)) == 0
        assert np.sum(np.isinf(normalized_patches)) == 0
        
        # Patches with all NaN should be all zeros
        assert np.all(normalized_patches[1] == 0)
    
    def test_save_load(self):
        """Test saving and loading normalizer state."""
        patches = self.create_test_patches(30, 9, 9)
        
        normalizer = PatchNormalizer(method='global', robust=True)
        normalizer.fit(patches)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "normalizer.pkl"
            
            # Save normalizer
            normalizer.save(filepath)
            
            # Create new normalizer and load
            new_normalizer = PatchNormalizer(method='local')  # Different initial config
            new_normalizer.load(filepath)
            
            # Check that configuration was loaded
            assert new_normalizer.method == 'global'
            assert new_normalizer.robust == True
            assert new_normalizer.is_fitted == True
            
            # Check that normalization produces same results
            normalized1 = normalizer.transform(patches[:5])
            normalized2 = new_normalizer.transform(patches[:5])
            
            np.testing.assert_array_almost_equal(normalized1, normalized2)
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        patches = self.create_test_patches(20, 9, 9)
        
        normalizer = PatchNormalizer(method='global', robust=False)
        normalizer.fit(patches)
        
        normalized_patches = normalizer.transform(patches)
        reconstructed_patches = normalizer.inverse_transform(normalized_patches)
        
        # Should approximately reconstruct original patches
        np.testing.assert_array_almost_equal(patches, reconstructed_patches, decimal=5)


class TestFeatureScaler:
    """Test feature scaling functionality."""
    
    def create_test_features(self, n_samples: int = 100, n_features: int = 8) -> np.ndarray:
        """Create test features with different scales."""
        features = np.zeros((n_samples, n_features))
        
        # Create features with different scales and distributions
        features[:, 0] = np.random.uniform(0, 1, n_samples)  # [0, 1]
        features[:, 1] = np.random.uniform(50, 150, n_samples)  # [50, 150]
        features[:, 2] = np.random.normal(0, 1, n_samples)  # Normal(0, 1)
        features[:, 3] = np.random.normal(100, 20, n_samples)  # Normal(100, 20)
        features[:, 4] = np.random.exponential(2, n_samples)  # Exponential
        features[:, 5] = np.random.uniform(-1, 1, n_samples)  # [-1, 1]
        features[:, 6] = np.random.lognormal(0, 1, n_samples)  # Log-normal
        features[:, 7] = np.random.uniform(0.1, 0.5, n_samples)  # [0.1, 0.5]
        
        return features.astype(np.float32)
    
    def test_standard_scaling(self):
        """Test standard scaling (z-score normalization)."""
        features = self.create_test_features(200, 8)
        
        scaler = FeatureScaler(method='standard', robust=False)
        scaler.fit(features)
        scaled_features = scaler.transform(features)
        
        # Check that each feature is standardized
        for i in range(features.shape[1]):
            feature_values = scaled_features[:, i]
            finite_mask = np.isfinite(feature_values)
            
            if np.sum(finite_mask) > 1:
                feature_mean = np.mean(feature_values[finite_mask])
                feature_std = np.std(feature_values[finite_mask])
                
                # Should be approximately standardized
                assert abs(feature_mean) < 0.1
                assert abs(feature_std - 1.0) < 0.1
    
    def test_minmax_scaling(self):
        """Test min-max scaling."""
        features = self.create_test_features(200, 8)
        
        scaler = FeatureScaler(method='minmax', feature_range=(0, 1))
        scaler.fit(features)
        scaled_features = scaler.transform(features)
        
        # Check that features are in [0, 1] range
        assert np.all(scaled_features >= 0)
        assert np.all(scaled_features <= 1)
        
        # Check that min and max are approximately achieved
        for i in range(features.shape[1]):
            feature_values = scaled_features[:, i]
            finite_mask = np.isfinite(feature_values)
            
            if np.sum(finite_mask) > 10:
                feature_min = np.min(feature_values[finite_mask])
                feature_max = np.max(feature_values[finite_mask])
                
                assert feature_min < 0.1  # Close to 0
                assert feature_max > 0.9  # Close to 1
    
    def test_robust_scaling(self):
        """Test robust scaling using median and IQR."""
        features = self.create_test_features(200, 8)
        
        # Add outliers
        features[0, :] = 1000  # Large outliers
        features[1, :] = -1000  # Large negative outliers
        
        scaler_robust = FeatureScaler(method='robust')
        scaler_standard = FeatureScaler(method='standard')
        
        scaler_robust.fit(features)
        scaler_standard.fit(features)
        
        scaled_robust = scaler_robust.transform(features)
        scaled_standard = scaler_standard.transform(features)
        
        # Robust scaling should be less affected by outliers
        for i in range(features.shape[1]):
            robust_std = np.std(scaled_robust[:, i])
            standard_std = np.std(scaled_standard[:, i])
            
            # Robust should have more reasonable scale
            assert robust_std < standard_std * 2
    
    def test_quantile_scaling(self):
        """Test quantile-based scaling."""
        features = self.create_test_features(200, 8)
        
        scaler = FeatureScaler(method='quantile')
        scaler.fit(features)
        scaled_features = scaler.transform(features)
        
        # Most values should be in [0, 1] range (between 5th and 95th percentiles)
        for i in range(features.shape[1]):
            feature_values = scaled_features[:, i]
            finite_mask = np.isfinite(feature_values)
            
            if np.sum(finite_mask) > 10:
                # Most values should be in reasonable range
                in_range_fraction = np.mean((feature_values >= 0) & (feature_values <= 1))
                assert in_range_fraction > 0.8  # At least 80% in range
    
    def test_outlier_clipping(self):
        """Test outlier clipping in standard scaling."""
        features = self.create_test_features(100, 4)
        
        # Add extreme outliers
        features[0, :] = 100
        features[1, :] = -100
        
        scaler = FeatureScaler(
            method='standard', 
            clip_outliers=True, 
            outlier_std_threshold=2.0
        )
        scaler.fit(features)
        scaled_features = scaler.transform(features)
        
        # Check that extreme values are clipped
        assert np.all(scaled_features >= -2.0)
        assert np.all(scaled_features <= 2.0)
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        features = self.create_test_features(50, 4)
        
        # Add NaN values
        features[0, 0] = np.nan
        features[1, :] = np.nan
        features[2, 2] = np.inf
        
        scaler = FeatureScaler(method='standard')
        scaler.fit(features)
        scaled_features = scaler.transform(features)
        
        # NaN values should be replaced with 0
        assert np.sum(np.isnan(scaled_features)) == 0
        assert np.sum(np.isinf(scaled_features)) == 0
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        features = self.create_test_features(100, 6)
        
        for method in ['standard', 'minmax', 'robust', 'quantile']:
            scaler = FeatureScaler(method=method)
            scaler.fit(features)
            
            scaled_features = scaler.transform(features)
            reconstructed_features = scaler.inverse_transform(scaled_features)
            
            # Should approximately reconstruct original features
            np.testing.assert_array_almost_equal(
                features, reconstructed_features, decimal=4
            )
    
    def test_save_load(self):
        """Test saving and loading scaler state."""
        features = self.create_test_features(100, 6)
        
        scaler = FeatureScaler(method='standard', robust=True)
        scaler.fit(features)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "scaler.pkl"
            
            # Save scaler
            scaler.save(filepath)
            
            # Create new scaler and load
            new_scaler = FeatureScaler(method='minmax')  # Different initial config
            new_scaler.load(filepath)
            
            # Check that configuration was loaded
            assert new_scaler.method == 'standard'
            assert new_scaler.robust == True
            assert new_scaler.is_fitted == True
            
            # Check that scaling produces same results
            scaled1 = scaler.transform(features[:10])
            scaled2 = new_scaler.transform(features[:10])
            
            np.testing.assert_array_almost_equal(scaled1, scaled2)
    
    def test_different_feature_ranges(self):
        """Test min-max scaling with different target ranges."""
        features = self.create_test_features(100, 4)
        
        # Test different target ranges
        ranges = [(0, 1), (-1, 1), (0, 10), (-5, 5)]
        
        for feature_range in ranges:
            scaler = FeatureScaler(method='minmax', feature_range=feature_range)
            scaler.fit(features)
            scaled_features = scaler.transform(features)
            
            min_val, max_val = feature_range
            
            # Check that features are in target range
            assert np.all(scaled_features >= min_val - 1e-6)
            assert np.all(scaled_features <= max_val + 1e-6)


class TestDataSplits:
    """Test data splitting functionality."""
    
    def test_basic_splits(self):
        """Test basic train/val/test splits."""
        n_samples = 1000
        
        train_indices, val_indices, test_indices = create_data_splits(
            n_samples, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15,
            random_seed=42
        )
        
        # Check sizes
        assert len(train_indices) == 700
        assert len(val_indices) == 150
        assert len(test_indices) == 150
        
        # Check no overlap
        all_indices = set(train_indices) | set(val_indices) | set(test_indices)
        assert len(all_indices) == n_samples
        
        # Check all indices are valid
        assert min(all_indices) == 0
        assert max(all_indices) == n_samples - 1
    
    def test_reproducible_splits(self):
        """Test that splits are reproducible with same seed."""
        n_samples = 500
        
        # Create splits with same seed
        train1, val1, test1 = create_data_splits(n_samples, random_seed=42)
        train2, val2, test2 = create_data_splits(n_samples, random_seed=42)
        
        # Should be identical
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2
        
        # Create splits with different seed
        train3, val3, test3 = create_data_splits(n_samples, random_seed=123)
        
        # Should be different
        assert train1 != train3
    
    def test_different_ratios(self):
        """Test splits with different ratios."""
        n_samples = 1000
        
        # Test different ratio combinations
        ratio_sets = [
            (0.8, 0.1, 0.1),
            (0.6, 0.2, 0.2),
            (0.5, 0.25, 0.25),
            (0.9, 0.05, 0.05)
        ]
        
        for train_ratio, val_ratio, test_ratio in ratio_sets:
            train_indices, val_indices, test_indices = create_data_splits(
                n_samples, train_ratio, val_ratio, test_ratio
            )
            
            # Check approximate sizes (allowing for rounding)
            expected_train = int(n_samples * train_ratio)
            expected_val = int(n_samples * val_ratio)
            expected_test = int(n_samples * test_ratio)
            
            assert abs(len(train_indices) - expected_train) <= 1
            assert abs(len(val_indices) - expected_val) <= 1
            assert abs(len(test_indices) - expected_test) <= 1
            
            # Check total
            total = len(train_indices) + len(val_indices) + len(test_indices)
            assert total == n_samples
    
    def test_stratified_splits(self):
        """Test stratified splits."""
        n_samples = 1000
        
        # Create stratification labels (3 classes)
        labels = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
        
        train_indices, val_indices, test_indices = create_data_splits(
            n_samples,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_by=labels
        )
        
        # Check that class proportions are approximately preserved
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]
        test_labels = labels[test_indices]
        
        for class_label in [0, 1, 2]:
            original_prop = np.mean(labels == class_label)
            train_prop = np.mean(train_labels == class_label)
            val_prop = np.mean(val_labels == class_label)
            test_prop = np.mean(test_labels == class_label)
            
            # Proportions should be similar (allowing some variation)
            assert abs(train_prop - original_prop) < 0.1
            assert abs(val_prop - original_prop) < 0.15  # More tolerance for smaller sets
            assert abs(test_prop - original_prop) < 0.15
    
    def test_edge_cases(self):
        """Test edge cases for data splits."""
        # Very small dataset
        train_indices, val_indices, test_indices = create_data_splits(
            10, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        assert len(train_indices) + len(val_indices) + len(test_indices) == 10
        
        # Single sample
        train_indices, val_indices, test_indices = create_data_splits(
            1, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        
        assert len(train_indices) == 1
        assert len(val_indices) == 0
        assert len(test_indices) == 0
    
    def test_invalid_ratios(self):
        """Test error handling for invalid ratios."""
        with pytest.raises(ValueError):
            create_data_splits(100, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)  # Sum > 1
        
        with pytest.raises(ValueError):
            create_data_splits(100, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)  # Sum < 1


if __name__ == "__main__":
    pytest.main([__file__])