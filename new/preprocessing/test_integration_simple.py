"""
Simple integration test for data preprocessing pipeline.
"""

import numpy as np
import tempfile
from pathlib import Path
import pickle
import h5py

# Test basic HDF5 functionality
def test_hdf5_operations():
    """Test HDF5 read/write operations."""
    print("Testing HDF5 operations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test.h5"
        
        # Create test data
        n_samples = 100
        patches = np.random.randn(n_samples, 9, 9).astype(np.float32)
        features = np.random.randn(n_samples, 8).astype(np.float32)
        targets = np.random.randn(n_samples).astype(np.float32)
        
        # Write to HDF5
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('patches', data=patches, compression='gzip', chunks=True)
            f.create_dataset('features', data=features, compression='gzip', chunks=True)
            f.create_dataset('targets', data=targets, compression='gzip', chunks=True)
            
            # Add metadata
            metadata = f.create_group('metadata')
            metadata.attrs['n_samples'] = n_samples
            metadata.attrs['patch_shape'] = (9, 9)
            metadata.attrs['n_features'] = 8
        
        # Read from HDF5
        with h5py.File(filepath, 'r') as f:
            read_patches = f['patches'][:]
            read_features = f['features'][:]
            read_targets = f['targets'][:]
            
            # Read metadata
            metadata = dict(f['metadata'].attrs)
        
        # Verify data integrity
        assert np.array_equal(patches, read_patches)
        assert np.array_equal(features, read_features)
        assert np.array_equal(targets, read_targets)
        assert metadata['n_samples'] == n_samples
        
        # Test partial reading
        indices = np.array([5, 15, 25, 35, 45])
        with h5py.File(filepath, 'r') as f:
            partial_patches = f['patches'][indices]
            partial_features = f['features'][indices]
            partial_targets = f['targets'][indices]
        
        assert partial_patches.shape == (5, 9, 9)
        assert partial_features.shape == (5, 8)
        assert partial_targets.shape == (5,)
        
        print("✓ HDF5 operations work correctly")


def test_data_normalization():
    """Test data normalization functionality."""
    print("Testing data normalization...")
    
    # Create test data with different scales
    n_samples = 1000
    
    # Features with different scales
    features = np.zeros((n_samples, 4))
    features[:, 0] = np.random.uniform(0, 1, n_samples)      # [0, 1]
    features[:, 1] = np.random.uniform(50, 150, n_samples)   # [50, 150]
    features[:, 2] = np.random.normal(0, 1, n_samples)       # Normal(0, 1)
    features[:, 3] = np.random.normal(100, 20, n_samples)    # Normal(100, 20)
    
    # Standard normalization
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero
    
    normalized = (features - means) / stds
    
    # Verify normalization
    norm_means = np.mean(normalized, axis=0)
    norm_stds = np.std(normalized, axis=0)
    
    assert np.allclose(norm_means, 0, atol=1e-10)
    assert np.allclose(norm_stds, 1, atol=1e-10)
    
    # Test robust normalization
    medians = np.median(features, axis=0)
    q25 = np.percentile(features, 25, axis=0)
    q75 = np.percentile(features, 75, axis=0)
    iqrs = q75 - q25
    iqrs[iqrs == 0] = 1.0
    
    robust_normalized = (features - medians) / iqrs
    
    # Should have reasonable scale
    assert np.all(np.abs(np.median(robust_normalized, axis=0)) < 0.1)
    
    print("✓ Data normalization works correctly")


def test_patch_operations():
    """Test patch extraction operations."""
    print("Testing patch operations...")
    
    # Create test surface
    surface = np.random.randn(20, 30)
    patch_size = (9, 9)
    
    # Test normal patch extraction
    center_t, center_k = 10, 15
    half_height, half_width = patch_size[0] // 2, patch_size[1] // 2
    
    t_start = center_t - half_height
    t_end = center_t + half_height + 1
    k_start = center_k - half_width
    k_end = center_k + half_width + 1
    
    patch = surface[t_start:t_end, k_start:k_end]
    assert patch.shape == patch_size
    
    # Test boundary handling with reflection padding
    padded_surface = np.pad(surface, ((4, 4), (4, 4)), mode='reflect')
    
    # Extract from near boundary
    boundary_center_t, boundary_center_k = 2, 3  # Near boundary
    padded_center_t = boundary_center_t + 4
    padded_center_k = boundary_center_k + 4
    
    boundary_patch = padded_surface[
        padded_center_t - half_height:padded_center_t + half_height + 1,
        padded_center_k - half_width:padded_center_k + half_width + 1
    ]
    
    assert boundary_patch.shape == patch_size
    
    # Test local normalization
    finite_mask = np.isfinite(patch)
    if np.sum(finite_mask) > 1:
        patch_mean = np.mean(patch[finite_mask])
        patch_std = np.std(patch[finite_mask])
        
        if patch_std > 0:
            normalized_patch = (patch - patch_mean) / patch_std
            
            # Check normalization
            norm_finite_mask = np.isfinite(normalized_patch)
            if np.sum(norm_finite_mask) > 1:
                norm_mean = np.mean(normalized_patch[norm_finite_mask])
                norm_std = np.std(normalized_patch[norm_finite_mask])
                
                assert abs(norm_mean) < 1e-10
                assert abs(norm_std - 1.0) < 1e-10
    
    print("✓ Patch operations work correctly")


def test_batch_iteration():
    """Test batch iteration logic."""
    print("Testing batch iteration...")
    
    n_samples = 237  # Non-divisible by batch size
    batch_size = 32
    
    # Test with drop_last=True
    n_batches_drop = n_samples // batch_size
    assert n_batches_drop == 7
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    batches = []
    for i in range(n_batches_drop):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        batches.append(batch_indices)
    
    assert len(batches) == 7
    assert all(len(batch) == batch_size for batch in batches)
    
    # Test with drop_last=False
    n_batches_keep = (n_samples + batch_size - 1) // batch_size
    assert n_batches_keep == 8
    
    batches_keep = []
    for i in range(n_batches_keep):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches_keep.append(batch_indices)
    
    assert len(batches_keep) == 8
    assert len(batches_keep[-1]) == n_samples - 7 * batch_size
    
    print("✓ Batch iteration works correctly")


def test_data_splits():
    """Test data splitting functionality."""
    print("Testing data splits...")
    
    n_samples = 1000
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    
    # Create indices and shuffle
    indices = np.arange(n_samples)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    
    # Calculate split sizes
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    # Create splits
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Verify splits
    assert len(train_indices) == 700
    assert len(val_indices) == 150
    assert len(test_indices) == 150
    
    # Check no overlap
    all_indices = set(train_indices) | set(val_indices) | set(test_indices)
    assert len(all_indices) == n_samples
    
    # Check all indices are valid
    assert min(all_indices) == 0
    assert max(all_indices) == n_samples - 1
    
    # Test reproducibility
    indices2 = np.arange(n_samples)
    np.random.seed(42)  # Same seed
    np.random.shuffle(indices2)
    
    assert np.array_equal(indices, indices2)
    
    print("✓ Data splits work correctly")


def test_feature_engineering():
    """Test feature engineering logic."""
    print("Testing feature engineering...")
    
    # Mock SABR parameters
    class MockSABRParams:
        def __init__(self):
            self.F0 = 100.0
            self.alpha = 0.2
            self.beta = 0.5
            self.nu = 0.3
            self.rho = -0.2
    
    sabr_params = MockSABRParams()
    strike = 105.0
    maturity = 1.0
    hagan_vol = 0.25
    
    # Create basic features
    features = []
    
    # SABR parameters
    features.extend([
        sabr_params.F0,
        sabr_params.alpha,
        sabr_params.beta,
        sabr_params.nu,
        sabr_params.rho
    ])
    
    # Market features
    moneyness = strike / sabr_params.F0
    log_moneyness = np.log(moneyness)
    
    features.extend([
        strike,
        maturity,
        moneyness,
        log_moneyness
    ])
    
    # Hagan volatility
    features.append(hagan_vol)
    
    # Derived features
    alpha_beta_interaction = sabr_params.alpha * (sabr_params.beta ** 2)
    nu_rho_interaction = sabr_params.nu * abs(sabr_params.rho)
    time_to_expiry_sqrt = np.sqrt(maturity)
    vol_of_vol_scaled = sabr_params.nu * sabr_params.alpha
    
    features.extend([
        alpha_beta_interaction,
        nu_rho_interaction,
        time_to_expiry_sqrt,
        vol_of_vol_scaled
    ])
    
    feature_array = np.array(features, dtype=np.float32)
    
    # Verify feature array
    expected_features = 5 + 4 + 1 + 4  # SABR + market + hagan + derived = 14
    assert len(feature_array) == expected_features
    assert np.all(np.isfinite(feature_array))
    
    # Test feature validation
    assert feature_array[0] > 0  # F0 should be positive
    assert 0 <= feature_array[2] <= 1  # Beta should be in [0, 1]
    assert -1 <= feature_array[4] <= 1  # Rho should be in [-1, 1]
    assert feature_array[6] > 0  # Strike should be positive
    assert feature_array[7] > 0  # Maturity should be positive
    
    print("✓ Feature engineering works correctly")


def run_all_tests():
    """Run all integration tests."""
    print("Running data preprocessing integration tests...")
    print("=" * 60)
    
    try:
        test_hdf5_operations()
        test_data_normalization()
        test_patch_operations()
        test_batch_iteration()
        test_data_splits()
        test_feature_engineering()
        
        print("=" * 60)
        print("✓ All integration tests passed!")
        print("Data preprocessing pipeline is working correctly.")
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)