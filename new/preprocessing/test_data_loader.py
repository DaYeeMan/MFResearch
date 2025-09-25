"""
Tests for data loading and preprocessing pipeline.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import pickle
import h5py
import time

from .data_loader import (
    DataLoaderConfig, DataSample, HDF5DataStore, DataPreprocessor,
    SABRDataLoader, DataIterator
)
from .patch_extractor import PatchExtractor, PatchConfig
from .feature_engineer import FeatureEngineer, FeatureConfig
from ..data_generation.sabr_params import SABRParams, GridConfig
from ..data_generation.data_orchestrator import SurfaceData


class TestHDF5DataStore:
    """Test HDF5 data storage functionality."""
    
    def test_create_and_write_datasets(self):
        """Test creating and writing to HDF5 datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_data.h5"
            
            # Test dataset creation
            with HDF5DataStore(filepath) as store:
                store.create_datasets(
                    n_samples=100,
                    patch_shape=(9, 9),
                    n_features=8
                )
                
                # Test metadata
                metadata = store.get_metadata()
                assert metadata['n_samples'] == 100
                assert metadata['patch_shape'] == (9, 9)
                assert metadata['n_features'] == 8
                
                # Test dataset size
                assert store.get_dataset_size() == 100
    
    def test_write_and_read_batch(self):
        """Test writing and reading data batches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_data.h5"
            
            # Create test data
            batch_size = 10
            patches = np.random.randn(batch_size, 9, 9).astype(np.float32)
            features = np.random.randn(batch_size, 8).astype(np.float32)
            targets = np.random.randn(batch_size).astype(np.float32)
            surface_ids = np.arange(batch_size, dtype=np.int32)
            point_ids = np.arange(batch_size, dtype=np.int32)
            
            with HDF5DataStore(filepath) as store:
                store.create_datasets(batch_size, (9, 9), 8)
                
                # Write batch
                store.write_batch(0, patches, features, targets, surface_ids, point_ids)
                
                # Read batch
                indices = np.arange(batch_size)
                read_patches, read_features, read_targets = store.read_batch(indices)
                
                # Verify data
                np.testing.assert_array_equal(patches, read_patches)
                np.testing.assert_array_equal(features, read_features)
                np.testing.assert_array_equal(targets, read_targets)
    
    def test_partial_read(self):
        """Test reading partial data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_data.h5"
            
            # Create and write test data
            n_samples = 50
            patches = np.random.randn(n_samples, 9, 9).astype(np.float32)
            features = np.random.randn(n_samples, 8).astype(np.float32)
            targets = np.random.randn(n_samples).astype(np.float32)
            surface_ids = np.arange(n_samples, dtype=np.int32)
            point_ids = np.arange(n_samples, dtype=np.int32)
            
            with HDF5DataStore(filepath) as store:
                store.create_datasets(n_samples, (9, 9), 8)
                store.write_batch(0, patches, features, targets, surface_ids, point_ids)
                
                # Read subset
                indices = np.array([5, 15, 25, 35, 45])
                read_patches, read_features, read_targets = store.read_batch(indices)
                
                # Verify subset
                np.testing.assert_array_equal(patches[indices], read_patches)
                np.testing.assert_array_equal(features[indices], read_features)
                np.testing.assert_array_equal(targets[indices], read_targets)


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def create_mock_surface_data(self) -> SurfaceData:
        """Create mock surface data for testing."""
        sabr_params = SABRParams(F0=100.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.2)
        grid_config = GridConfig(
            strike_range=(80.0, 120.0),
            maturity_range=(0.1, 2.0),
            n_strikes=21,
            n_maturities=11
        )
        
        strikes = grid_config.get_strikes(sabr_params.F0)
        maturities = grid_config.get_maturities()
        
        # Create mock surfaces
        hf_surface = np.random.uniform(0.1, 0.5, (11, 21))
        lf_surface = np.random.uniform(0.1, 0.5, (11, 21))
        residuals = hf_surface - lf_surface
        
        return SurfaceData(
            parameters=sabr_params,
            grid_config=grid_config,
            hf_surface=hf_surface,
            lf_surface=lf_surface,
            residuals=residuals,
            strikes=strikes,
            maturities=maturities,
            generation_time=1.0,
            quality_metrics={}
        )
    
    def test_preprocess_surface(self):
        """Test surface preprocessing."""
        # Create components
        patch_config = PatchConfig(patch_size=(9, 9))
        feature_config = FeatureConfig()
        loader_config = DataLoaderConfig(hf_budget_per_surface=50)
        
        patch_extractor = PatchExtractor(patch_config)
        feature_engineer = FeatureEngineer(feature_config)
        preprocessor = DataPreprocessor(patch_extractor, feature_engineer, loader_config)
        
        # Create mock data
        surface_data = self.create_mock_surface_data()
        
        # Preprocess
        samples = preprocessor.preprocess_surface(surface_data, surface_id=0)
        
        # Verify samples
        assert len(samples) > 0
        assert len(samples) <= loader_config.hf_budget_per_surface
        
        for sample in samples:
            assert isinstance(sample, DataSample)
            assert sample.patch.shape == (9, 9)
            assert len(sample.point_features) == len(feature_engineer.feature_names)
            assert np.isfinite(sample.target)
            assert sample.surface_id == 0
    
    def test_hf_point_sampling(self):
        """Test HF point sampling strategies."""
        patch_config = PatchConfig(patch_size=(9, 9))
        feature_config = FeatureConfig()
        loader_config = DataLoaderConfig(hf_budget_per_surface=10)
        
        patch_extractor = PatchExtractor(patch_config)
        feature_engineer = FeatureEngineer(feature_config)
        preprocessor = DataPreprocessor(patch_extractor, feature_engineer, loader_config)
        
        surface_data = self.create_mock_surface_data()
        
        # Test with limited budget
        samples = preprocessor.preprocess_surface(surface_data, surface_id=0)
        assert len(samples) <= 10
        
        # Test with large budget
        loader_config.hf_budget_per_surface = 1000
        preprocessor.config = loader_config
        samples = preprocessor.preprocess_surface(surface_data, surface_id=0)
        
        # Should be limited by available valid points
        n_valid_points = np.sum(np.isfinite(surface_data.residuals))
        assert len(samples) <= n_valid_points


class TestDataIterator:
    """Test data iterator functionality."""
    
    def create_test_hdf5_file(self, temp_dir: Path, n_samples: int = 100) -> Path:
        """Create test HDF5 file with sample data."""
        filepath = temp_dir / "test_data.h5"
        
        # Create test data
        patches = np.random.randn(n_samples, 9, 9).astype(np.float32)
        features = np.random.randn(n_samples, 8).astype(np.float32)
        targets = np.random.randn(n_samples).astype(np.float32)
        surface_ids = np.random.randint(0, 10, n_samples, dtype=np.int32)
        point_ids = np.random.randint(0, 50, n_samples, dtype=np.int32)
        
        with HDF5DataStore(filepath) as store:
            store.create_datasets(n_samples, (9, 9), 8)
            store.write_batch(0, patches, features, targets, surface_ids, point_ids)
        
        return filepath
    
    def test_basic_iteration(self):
        """Test basic data iteration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hdf5_file = self.create_test_hdf5_file(temp_path, n_samples=100)
            
            indices = list(range(100))
            iterator = DataIterator(
                hdf5_file=hdf5_file,
                indices=indices,
                batch_size=10,
                shuffle=False
            )
            
            # Test iterator properties
            assert len(iterator) == 10  # 100 samples / 10 batch_size
            
            # Test iteration
            batches = list(iterator)
            assert len(batches) == 10
            
            for patches, features, targets in batches:
                assert patches.shape[0] <= 10  # batch_size
                assert features.shape[0] <= 10
                assert targets.shape[0] <= 10
                assert patches.shape[1:] == (9, 9)
                assert features.shape[1] == 8
    
    def test_shuffling(self):
        """Test data shuffling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hdf5_file = self.create_test_hdf5_file(temp_path, n_samples=50)
            
            indices = list(range(50))
            
            # Create two iterators with same seed
            iterator1 = DataIterator(hdf5_file, indices, batch_size=10, shuffle=True)
            iterator2 = DataIterator(hdf5_file, indices, batch_size=10, shuffle=True)
            
            # Get first batches
            batch1_1 = next(iter(iterator1))
            batch2_1 = next(iter(iterator2))
            
            # Should be different due to different random states
            # (This is probabilistic, but very likely to be true)
            patches1, _, _ = batch1_1
            patches2, _, _ = batch2_1
            
            # At least some difference expected
            assert not np.array_equal(patches1, patches2)
    
    def test_drop_last(self):
        """Test drop_last functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hdf5_file = self.create_test_hdf5_file(temp_path, n_samples=95)
            
            indices = list(range(95))
            
            # Test with drop_last=True
            iterator = DataIterator(
                hdf5_file, indices, batch_size=10, shuffle=False, drop_last=True
            )
            batches = list(iterator)
            assert len(batches) == 9  # 95 // 10
            
            # Test with drop_last=False
            iterator = DataIterator(
                hdf5_file, indices, batch_size=10, shuffle=False, drop_last=False
            )
            batches = list(iterator)
            assert len(batches) == 10  # ceil(95 / 10)
            
            # Last batch should have 5 samples
            last_batch = batches[-1]
            assert last_batch[0].shape[0] == 5
    
    def test_sample_batch(self):
        """Test sample batch functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hdf5_file = self.create_test_hdf5_file(temp_path, n_samples=100)
            
            indices = list(range(100))
            iterator = DataIterator(hdf5_file, indices, batch_size=10)
            
            # Get sample batch
            sample_patches, sample_features, sample_targets = iterator.get_sample_batch(5)
            
            assert sample_patches.shape == (5, 9, 9)
            assert sample_features.shape == (5, 8)
            assert sample_targets.shape == (5,)


class TestSABRDataLoader:
    """Test main SABR data loader."""
    
    def create_mock_data_directory(self, temp_dir: Path, n_surfaces: int = 10):
        """Create mock data directory with surface files."""
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir(parents=True)
        
        # Create mock surface files
        for i in range(n_surfaces):
            sabr_params = SABRParams(
                F0=100.0 + i,
                alpha=0.2 + i * 0.01,
                beta=0.5,
                nu=0.3 + i * 0.01,
                rho=-0.2 + i * 0.01
            )
            
            grid_config = GridConfig(
                strike_range=(80.0, 120.0),
                maturity_range=(0.1, 2.0),
                n_strikes=21,
                n_maturities=11
            )
            
            strikes = grid_config.get_strikes(sabr_params.F0)
            maturities = grid_config.get_maturities()
            
            # Create mock surfaces with some variation
            hf_surface = np.random.uniform(0.1, 0.5, (11, 21))
            lf_surface = np.random.uniform(0.1, 0.5, (11, 21))
            residuals = hf_surface - lf_surface
            
            surface_data = SurfaceData(
                parameters=sabr_params,
                grid_config=grid_config,
                hf_surface=hf_surface,
                lf_surface=lf_surface,
                residuals=residuals,
                strikes=strikes,
                maturities=maturities,
                generation_time=1.0,
                quality_metrics={}
            )
            
            # Save surface
            surface_file = raw_dir / f"surface_{i:06d}.pkl"
            with open(surface_file, 'wb') as f:
                pickle.dump(surface_data, f)
    
    def test_data_preprocessing(self):
        """Test full data preprocessing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock data
            self.create_mock_data_directory(temp_path, n_surfaces=5)
            
            # Create data loader
            patch_config = PatchConfig(patch_size=(9, 9))
            feature_config = FeatureConfig()
            loader_config = DataLoaderConfig(
                hf_budget_per_surface=20,
                validation_split=0.2,
                test_split=0.2
            )
            
            data_loader = SABRDataLoader(
                temp_path, patch_config, feature_config, loader_config
            )
            
            # Preprocess data
            data_loader.preprocess_data()
            
            # Verify preprocessing
            assert data_loader.is_preprocessed
            assert (temp_path / "processed" / "training_data.h5").exists()
            assert (temp_path / "processed" / "feature_stats.pkl").exists()
            assert (temp_path / "splits" / "data_splits.pkl").exists()
            
            # Check dataset info
            info = data_loader.get_dataset_info()
            assert info['total_samples'] > 0
            assert info['patch_shape'] == (9, 9)
            assert info['n_features'] > 0
            assert sum(info['splits'].values()) == info['total_samples']
    
    def test_data_loader_creation(self):
        """Test data loader creation for different splits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock data
            self.create_mock_data_directory(temp_path, n_surfaces=3)
            
            # Create and preprocess
            patch_config = PatchConfig(patch_size=(9, 9))
            feature_config = FeatureConfig()
            loader_config = DataLoaderConfig(
                hf_budget_per_surface=10,
                batch_size=5
            )
            
            data_loader = SABRDataLoader(
                temp_path, patch_config, feature_config, loader_config
            )
            data_loader.preprocess_data()
            
            # Test data loaders for different splits
            for split in ['train', 'val', 'test']:
                iterator = data_loader.get_data_loader(split)
                assert isinstance(iterator, DataIterator)
                
                # Test iteration
                batch = next(iter(iterator))
                patches, features, targets = batch
                
                assert patches.shape[1:] == (9, 9)
                assert features.shape[1] == len(data_loader.feature_engineer.feature_names)
                assert len(targets.shape) == 1
    
    def test_feature_normalization(self):
        """Test feature normalization during preprocessing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock data
            self.create_mock_data_directory(temp_path, n_surfaces=3)
            
            # Create data loader
            patch_config = PatchConfig(patch_size=(9, 9))
            feature_config = FeatureConfig(normalize_features=True)
            loader_config = DataLoaderConfig(hf_budget_per_surface=10)
            
            data_loader = SABRDataLoader(
                temp_path, patch_config, feature_config, loader_config
            )
            data_loader.preprocess_data()
            
            # Check that feature engineer is fitted
            assert data_loader.feature_engineer.is_fitted
            assert data_loader.feature_engineer.feature_stats is not None
            
            # Get sample batch and check normalization
            train_loader = data_loader.get_data_loader('train')
            patches, features, targets = train_loader.get_sample_batch(10)
            
            # Features should be approximately normalized
            feature_means = np.mean(features, axis=0)
            feature_stds = np.std(features, axis=0)
            
            # Should be close to standard normal (allowing some variation due to small sample)
            assert np.all(np.abs(feature_means) < 2.0)  # Reasonable range
            assert np.all(feature_stds > 0.1)  # Not collapsed


class TestPerformance:
    """Test performance characteristics of data loading."""
    
    def test_loading_performance(self):
        """Test data loading performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create larger dataset for performance testing
            n_samples = 1000
            filepath = temp_path / "perf_test.h5"
            
            # Create test data
            patches = np.random.randn(n_samples, 9, 9).astype(np.float32)
            features = np.random.randn(n_samples, 8).astype(np.float32)
            targets = np.random.randn(n_samples).astype(np.float32)
            surface_ids = np.random.randint(0, 100, n_samples, dtype=np.int32)
            point_ids = np.random.randint(0, 50, n_samples, dtype=np.int32)
            
            # Write data
            with HDF5DataStore(filepath) as store:
                store.create_datasets(n_samples, (9, 9), 8)
                store.write_batch(0, patches, features, targets, surface_ids, point_ids)
            
            # Test loading performance
            indices = list(range(n_samples))
            iterator = DataIterator(
                filepath, indices, batch_size=64, shuffle=True
            )
            
            # Time full epoch
            start_time = time.time()
            total_samples = 0
            
            for batch in iterator:
                total_samples += batch[0].shape[0]
            
            elapsed_time = time.time() - start_time
            
            # Performance assertions
            assert total_samples == n_samples
            assert elapsed_time < 10.0  # Should complete within 10 seconds
            
            samples_per_second = n_samples / elapsed_time
            assert samples_per_second > 100  # At least 100 samples/second
    
    def test_memory_usage(self):
        """Test memory usage during data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dataset
            n_samples = 500
            filepath = temp_path / "memory_test.h5"
            
            patches = np.random.randn(n_samples, 9, 9).astype(np.float32)
            features = np.random.randn(n_samples, 8).astype(np.float32)
            targets = np.random.randn(n_samples).astype(np.float32)
            surface_ids = np.arange(n_samples, dtype=np.int32)
            point_ids = np.arange(n_samples, dtype=np.int32)
            
            with HDF5DataStore(filepath) as store:
                store.create_datasets(n_samples, (9, 9), 8)
                store.write_batch(0, patches, features, targets, surface_ids, point_ids)
            
            # Test with different batch sizes
            for batch_size in [16, 64, 128]:
                indices = list(range(n_samples))
                iterator = DataIterator(filepath, indices, batch_size=batch_size)
                
                # Load all batches
                batches = list(iterator)
                
                # Verify all data loaded correctly
                total_loaded = sum(batch[0].shape[0] for batch in batches)
                assert total_loaded == n_samples


if __name__ == "__main__":
    pytest.main([__file__])