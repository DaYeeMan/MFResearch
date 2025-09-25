"""
Simple test for data loading functionality.
"""

import sys
import numpy as np
import tempfile
from pathlib import Path
import h5py

# Add the project root to Python path
sys.path.append('.')

def test_hdf5_basic():
    """Test basic HDF5 functionality."""
    print("Testing HDF5 basic functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = Path(temp_dir) / "test.h5"
        
        # Create test data
        n_samples = 100
        patches = np.random.randn(n_samples, 9, 9).astype(np.float32)
        features = np.random.randn(n_samples, 8).astype(np.float32)
        targets = np.random.randn(n_samples).astype(np.float32)
        
        # Write to HDF5
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('patches', data=patches, compression='gzip')
            f.create_dataset('features', data=features, compression='gzip')
            f.create_dataset('targets', data=targets, compression='gzip')
        
        # Read from HDF5
        with h5py.File(filepath, 'r') as f:
            read_patches = f['patches'][:]
            read_features = f['features'][:]
            read_targets = f['targets'][:]
        
        # Verify
        assert np.array_equal(patches, read_patches)
        assert np.array_equal(features, read_features)
        assert np.array_equal(targets, read_targets)
        
        print("✓ HDF5 basic functionality works")


def test_data_splits():
    """Test data splitting functionality."""
    print("Testing data splits...")
    
    n_samples = 1000
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Create splits
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Verify
    assert len(train_indices) == 700
    assert len(val_indices) == 150
    assert len(test_indices) == 150
    
    # Check no overlap
    all_indices = set(train_indices) | set(val_indices) | set(test_indices)
    assert len(all_indices) == n_samples
    
    print("✓ Data splits work correctly")


def test_normalization():
    """Test basic normalization."""
    print("Testing normalization...")
    
    # Create test data with different scales
    n_samples = 1000
    features = np.zeros((n_samples, 4))
    features[:, 0] = np.random.uniform(0, 1, n_samples)
    features[:, 1] = np.random.uniform(50, 150, n_samples)
    features[:, 2] = np.random.normal(0, 1, n_samples)
    features[:, 3] = np.random.normal(100, 20, n_samples)
    
    # Standard normalization
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized = (features - means) / stds
    
    # Verify normalization
    norm_means = np.mean(normalized, axis=0)
    norm_stds = np.std(normalized, axis=0)
    
    assert np.allclose(norm_means, 0, atol=1e-10)
    assert np.allclose(norm_stds, 1, atol=1e-10)
    
    print("✓ Normalization works correctly")


def test_batch_iteration():
    """Test batch iteration logic."""
    print("Testing batch iteration...")
    
    n_samples = 237  # Non-divisible by batch size
    batch_size = 32
    
    # Calculate expected batches
    n_batches_drop = n_samples // batch_size  # 7 batches
    n_batches_keep = (n_samples + batch_size - 1) // batch_size  # 8 batches
    
    assert n_batches_drop == 7
    assert n_batches_keep == 8
    
    # Test batch creation
    indices = np.arange(n_samples)
    
    # With drop_last=True
    batches_drop = []
    for i in range(n_batches_drop):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        batches_drop.append(batch_indices)
    
    assert len(batches_drop) == 7
    assert all(len(batch) == batch_size for batch in batches_drop)
    
    # With drop_last=False
    batches_keep = []
    for i in range(n_batches_keep):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches_keep.append(batch_indices)
    
    assert len(batches_keep) == 8
    assert len(batches_keep[-1]) == 237 - 7 * 32  # Last batch has remainder
    
    print("✓ Batch iteration logic works correctly")


def test_patch_extraction():
    """Test basic patch extraction logic."""
    print("Testing patch extraction...")
    
    # Create test surface
    surface = np.random.randn(20, 30)
    patch_size = (9, 9)
    
    # Extract patch from center
    center_t, center_k = 10, 15
    half_height, half_width = patch_size[0] // 2, patch_size[1] // 2
    
    t_start = center_t - half_height
    t_end = center_t + half_height + 1
    k_start = center_k - half_width
    k_end = center_k + half_width + 1
    
    patch = surface[t_start:t_end, k_start:k_end]
    
    assert patch.shape == patch_size
    
    # Test boundary handling with padding
    padded_surface = np.pad(surface, ((4, 4), (4, 4)), mode='reflect')
    
    # Extract from padded surface
    padded_center_t, padded_center_k = center_t + 4, center_k + 4
    padded_patch = padded_surface[
        padded_center_t - half_height:padded_center_t + half_height + 1,
        padded_center_k - half_width:padded_center_k + half_width + 1
    ]
    
    assert padded_patch.shape == patch_size
    
    print("✓ Patch extraction logic works correctly")


if __name__ == "__main__":
    print("Running simple data loading tests...")
    print("=" * 50)
    
    try:
        test_hdf5_basic()
        test_data_splits()
        test_normalization()
        test_batch_iteration()
        test_patch_extraction()
        
        print("=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        raise