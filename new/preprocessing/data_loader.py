"""
Data loading and preprocessing pipeline for SABR volatility surface modeling.

This module provides efficient data loading with batching, shuffling, and
HDF5-based storage for preprocessed training data.
"""

import numpy as np
import h5py
from typing import List, Tuple, Optional, Union, Dict, Any, Iterator
from dataclasses import dataclass
from pathlib import Path
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from ..data_generation.sabr_params import SABRParams, GridConfig
from ..data_generation.data_orchestrator import SurfaceData
from .patch_extractor import PatchExtractor, PatchConfig
from .feature_engineer import FeatureEngineer, FeatureConfig, FeatureStats
from ..utils.logging_utils import get_logger
from ..utils.common import ensure_directory

logger = get_logger(__name__)


@dataclass
class DataLoaderConfig:
    """
    Configuration for data loader.
    
    Attributes:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker threads for data loading
        prefetch_batches: Number of batches to prefetch
        drop_last: Whether to drop last incomplete batch
        pin_memory: Whether to pin memory for faster GPU transfer
        cache_preprocessed: Whether to cache preprocessed data
        hf_budget_per_surface: Number of HF points to sample per surface
        validation_split: Fraction of data for validation
        test_split: Fraction of data for test
        random_seed: Random seed for reproducibility
    """
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4
    prefetch_batches: int = 2
    drop_last: bool = False
    pin_memory: bool = False
    cache_preprocessed: bool = True
    hf_budget_per_surface: int = 200
    validation_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


@dataclass
class DataSample:
    """
    Single training sample for MDA-CNN.
    
    Attributes:
        patch: LF surface patch around HF point
        point_features: Point features for MLP branch
        target: Target residual value
        surface_id: ID of source surface
        point_id: ID of point within surface
        metadata: Additional metadata
    """
    patch: np.ndarray
    point_features: np.ndarray
    target: float
    surface_id: int
    point_id: int
    metadata: Dict[str, Any]


class HDF5DataStore:
    """
    HDF5-based storage for preprocessed training data.
    """
    
    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize HDF5 data store.
        
        Args:
            filepath: Path to HDF5 file
        """
        self.filepath = Path(filepath)
        self.is_open = False
        self._file = None
        self._lock = threading.Lock()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def open(self, mode: str = 'r'):
        """Open HDF5 file."""
        with self._lock:
            if not self.is_open:
                ensure_directory(self.filepath.parent)
                self._file = h5py.File(self.filepath, mode)
                self.is_open = True
    
    def close(self):
        """Close HDF5 file."""
        with self._lock:
            if self.is_open and self._file is not None:
                self._file.close()
                self._file = None
                self.is_open = False
    
    def create_datasets(self, 
                       n_samples: int,
                       patch_shape: Tuple[int, int],
                       n_features: int,
                       compression: str = 'gzip'):
        """
        Create HDF5 datasets for training data.
        
        Args:
            n_samples: Total number of training samples
            patch_shape: Shape of surface patches
            n_features: Number of point features
            compression: Compression algorithm
        """
        if not self.is_open:
            self.open('w')
        
        # Create datasets
        self._file.create_dataset(
            'patches', 
            shape=(n_samples, patch_shape[0], patch_shape[1]),
            dtype=np.float32,
            compression=compression,
            chunks=True
        )
        
        self._file.create_dataset(
            'point_features',
            shape=(n_samples, n_features),
            dtype=np.float32,
            compression=compression,
            chunks=True
        )
        
        self._file.create_dataset(
            'targets',
            shape=(n_samples,),
            dtype=np.float32,
            compression=compression,
            chunks=True
        )
        
        self._file.create_dataset(
            'surface_ids',
            shape=(n_samples,),
            dtype=np.int32,
            compression=compression,
            chunks=True
        )
        
        self._file.create_dataset(
            'point_ids',
            shape=(n_samples,),
            dtype=np.int32,
            compression=compression,
            chunks=True
        )
        
        # Create metadata group
        metadata_group = self._file.create_group('metadata')
        metadata_group.attrs['n_samples'] = n_samples
        metadata_group.attrs['patch_shape'] = patch_shape
        metadata_group.attrs['n_features'] = n_features
        
        logger.info(f"Created HDF5 datasets for {n_samples} samples")
    
    def write_batch(self, 
                   start_idx: int,
                   patches: np.ndarray,
                   point_features: np.ndarray,
                   targets: np.ndarray,
                   surface_ids: np.ndarray,
                   point_ids: np.ndarray):
        """
        Write a batch of data to HDF5 file.
        
        Args:
            start_idx: Starting index for batch
            patches: Batch of surface patches
            point_features: Batch of point features
            targets: Batch of target values
            surface_ids: Batch of surface IDs
            point_ids: Batch of point IDs
        """
        if not self.is_open:
            raise ValueError("HDF5 file not open")
        
        batch_size = len(patches)
        end_idx = start_idx + batch_size
        
        with self._lock:
            self._file['patches'][start_idx:end_idx] = patches
            self._file['point_features'][start_idx:end_idx] = point_features
            self._file['targets'][start_idx:end_idx] = targets
            self._file['surface_ids'][start_idx:end_idx] = surface_ids
            self._file['point_ids'][start_idx:end_idx] = point_ids
    
    def read_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read a batch of data from HDF5 file.
        
        Args:
            indices: Indices to read
            
        Returns:
            Tuple of (patches, point_features, targets)
        """
        if not self.is_open:
            raise ValueError("HDF5 file not open")
        
        with self._lock:
            patches = self._file['patches'][indices]
            point_features = self._file['point_features'][indices]
            targets = self._file['targets'][indices]
        
        return patches, point_features, targets
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        if not self.is_open:
            raise ValueError("HDF5 file not open")
        
        metadata = dict(self._file['metadata'].attrs)
        return metadata
    
    def get_dataset_size(self) -> int:
        """Get total number of samples."""
        if not self.is_open:
            raise ValueError("HDF5 file not open")
        
        return self._file['patches'].shape[0]


class DataPreprocessor:
    """
    Preprocesses raw surface data into training samples.
    """
    
    def __init__(self,
                 patch_extractor: PatchExtractor,
                 feature_engineer: FeatureEngineer,
                 config: DataLoaderConfig):
        """
        Initialize data preprocessor.
        
        Args:
            patch_extractor: Patch extraction utility
            feature_engineer: Feature engineering utility
            config: Data loader configuration
        """
        self.patch_extractor = patch_extractor
        self.feature_engineer = feature_engineer
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
    
    def preprocess_surface(self, surface_data: SurfaceData, surface_id: int) -> List[DataSample]:
        """
        Preprocess a single surface into training samples.
        
        Args:
            surface_data: Raw surface data
            surface_id: Surface identifier
            
        Returns:
            List of training samples
        """
        # Sample HF points from the surface
        hf_points = self._sample_hf_points(surface_data)
        
        if len(hf_points) == 0:
            logger.warning(f"No valid HF points found for surface {surface_id}")
            return []
        
        samples = []
        
        for point_id, (t_idx, k_idx) in enumerate(hf_points):
            try:
                # Extract patch around HF point
                patch = self.patch_extractor.extract_patch(
                    surface_data.lf_surface, (t_idx, k_idx)
                )
                
                # Create point features
                strike = surface_data.strikes[k_idx]
                maturity = surface_data.maturities[t_idx]
                hagan_vol = surface_data.lf_surface[t_idx, k_idx]
                
                point_features = self.feature_engineer.create_point_features(
                    surface_data.parameters,
                    strike,
                    maturity,
                    hagan_vol
                )
                
                # Get target residual
                target = surface_data.residuals[t_idx, k_idx]
                
                # Skip if target is not finite
                if not np.isfinite(target):
                    continue
                
                # Create sample
                sample = DataSample(
                    patch=patch,
                    point_features=point_features,
                    target=target,
                    surface_id=surface_id,
                    point_id=point_id,
                    metadata={
                        'strike': strike,
                        'maturity': maturity,
                        'hagan_vol': hagan_vol,
                        'grid_coords': (t_idx, k_idx)
                    }
                )
                
                samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Failed to process point {point_id} in surface {surface_id}: {e}")
                continue
        
        return samples
    
    def _sample_hf_points(self, surface_data: SurfaceData) -> List[Tuple[int, int]]:
        """
        Sample high-fidelity points from a surface.
        
        Args:
            surface_data: Surface data
            
        Returns:
            List of (maturity_idx, strike_idx) coordinates
        """
        n_maturities, n_strikes = surface_data.hf_surface.shape
        
        # Find all valid points (finite residuals)
        finite_mask = np.isfinite(surface_data.residuals)
        valid_coords = np.where(finite_mask)
        
        if len(valid_coords[0]) == 0:
            return []
        
        # Create list of valid coordinates
        all_points = list(zip(valid_coords[0], valid_coords[1]))
        
        # Sample points up to budget
        n_sample = min(self.config.hf_budget_per_surface, len(all_points))
        
        if n_sample == len(all_points):
            return all_points
        else:
            # Random sampling
            sampled_indices = self.rng.choice(len(all_points), n_sample, replace=False)
            return [all_points[i] for i in sampled_indices]


class SABRDataLoader:
    """
    Main data loader for SABR volatility surface training data.
    """
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 patch_config: PatchConfig,
                 feature_config: FeatureConfig,
                 loader_config: DataLoaderConfig):
        """
        Initialize SABR data loader.
        
        Args:
            data_dir: Directory containing raw surface data
            patch_config: Patch extraction configuration
            feature_config: Feature engineering configuration
            loader_config: Data loader configuration
        """
        self.data_dir = Path(data_dir)
        self.patch_config = patch_config
        self.feature_config = feature_config
        self.config = loader_config
        
        # Initialize components
        self.patch_extractor = PatchExtractor(patch_config)
        self.feature_engineer = FeatureEngineer(feature_config)
        self.preprocessor = DataPreprocessor(
            self.patch_extractor, 
            self.feature_engineer, 
            loader_config
        )
        
        # Data storage
        self.processed_dir = self.data_dir / "processed"
        self.splits_dir = self.data_dir / "splits"
        ensure_directory(self.processed_dir)
        ensure_directory(self.splits_dir)
        
        # State
        self.is_preprocessed = False
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        
        logger.info(f"Initialized SABR data loader for {self.data_dir}")
    
    def preprocess_data(self, force_reprocess: bool = False) -> None:
        """
        Preprocess raw surface data into training format.
        
        Args:
            force_reprocess: Whether to force reprocessing even if data exists
        """
        processed_file = self.processed_dir / "training_data.h5"
        
        if processed_file.exists() and not force_reprocess:
            logger.info("Preprocessed data already exists, skipping preprocessing")
            self.is_preprocessed = True
            return
        
        logger.info("Starting data preprocessing...")
        
        # Load raw surface data
        raw_dir = self.data_dir / "raw"
        surface_files = list(raw_dir.glob("surface_*.pkl"))
        
        if not surface_files:
            raise FileNotFoundError(f"No surface files found in {raw_dir}")
        
        logger.info(f"Found {len(surface_files)} surface files")
        
        # First pass: collect all samples to determine dataset size
        all_samples = []
        
        for i, surface_file in enumerate(surface_files):
            try:
                with open(surface_file, 'rb') as f:
                    surface_data = pickle.load(f)
                
                samples = self.preprocessor.preprocess_surface(surface_data, i)
                all_samples.extend(samples)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(surface_files)} surfaces, "
                               f"{len(all_samples)} samples so far")
                    
            except Exception as e:
                logger.warning(f"Failed to process {surface_file}: {e}")
                continue
        
        if not all_samples:
            raise ValueError("No valid training samples generated")
        
        logger.info(f"Generated {len(all_samples)} training samples")
        
        # Fit feature normalization
        logger.info("Fitting feature normalization...")
        all_features = np.array([sample.point_features for sample in all_samples])
        self.feature_engineer.fit_normalization(all_features)
        
        # Save feature statistics
        feature_stats_file = self.processed_dir / "feature_stats.pkl"
        with open(feature_stats_file, 'wb') as f:
            pickle.dump(self.feature_engineer.feature_stats, f)
        
        # Create HDF5 dataset
        logger.info("Creating HDF5 dataset...")
        patch_shape = self.patch_config.patch_size
        n_features = len(self.feature_engineer.feature_names)
        
        with HDF5DataStore(processed_file) as store:
            store.create_datasets(len(all_samples), patch_shape, n_features)
            
            # Write data in batches
            batch_size = 1000
            for start_idx in range(0, len(all_samples), batch_size):
                end_idx = min(start_idx + batch_size, len(all_samples))
                batch_samples = all_samples[start_idx:end_idx]
                
                # Prepare batch data
                patches = np.array([s.patch for s in batch_samples])
                features = np.array([s.point_features for s in batch_samples])
                targets = np.array([s.target for s in batch_samples])
                surface_ids = np.array([s.surface_id for s in batch_samples])
                point_ids = np.array([s.point_id for s in batch_samples])
                
                # Normalize features
                features = self.feature_engineer.normalize_features(features)
                
                # Write batch
                store.write_batch(start_idx, patches, features, targets, surface_ids, point_ids)
                
                if (end_idx) % 5000 == 0:
                    logger.info(f"Written {end_idx}/{len(all_samples)} samples to HDF5")
        
        # Create data splits
        self._create_data_splits(len(all_samples))
        
        self.is_preprocessed = True
        logger.info("Data preprocessing completed")
    
    def _create_data_splits(self, n_samples: int) -> None:
        """
        Create train/validation/test splits.
        
        Args:
            n_samples: Total number of samples
        """
        logger.info("Creating data splits...")
        
        # Create indices
        indices = np.arange(n_samples)
        rng = np.random.RandomState(self.config.random_seed)
        rng.shuffle(indices)
        
        # Calculate split sizes
        n_test = int(n_samples * self.config.test_split)
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_test - n_val
        
        # Split indices
        self.test_indices = indices[:n_test].tolist()
        self.val_indices = indices[n_test:n_test + n_val].tolist()
        self.train_indices = indices[n_test + n_val:].tolist()
        
        # Save splits
        splits_data = {
            'train_indices': self.train_indices,
            'val_indices': self.val_indices,
            'test_indices': self.test_indices,
            'random_seed': self.config.random_seed
        }
        
        splits_file = self.splits_dir / "data_splits.pkl"
        with open(splits_file, 'wb') as f:
            pickle.dump(splits_data, f)
        
        logger.info(f"Created splits: {n_train} train, {n_val} val, {n_test} test")
    
    def load_data_splits(self) -> None:
        """Load existing data splits."""
        splits_file = self.splits_dir / "data_splits.pkl"
        
        if not splits_file.exists():
            raise FileNotFoundError(f"Data splits not found: {splits_file}")
        
        with open(splits_file, 'rb') as f:
            splits_data = pickle.load(f)
        
        self.train_indices = splits_data['train_indices']
        self.val_indices = splits_data['val_indices']
        self.test_indices = splits_data['test_indices']
        
        logger.info(f"Loaded splits: {len(self.train_indices)} train, "
                   f"{len(self.val_indices)} val, {len(self.test_indices)} test")
    
    def load_feature_stats(self) -> None:
        """Load feature normalization statistics."""
        feature_stats_file = self.processed_dir / "feature_stats.pkl"
        
        if not feature_stats_file.exists():
            raise FileNotFoundError(f"Feature stats not found: {feature_stats_file}")
        
        with open(feature_stats_file, 'rb') as f:
            self.feature_engineer.feature_stats = pickle.load(f)
            self.feature_engineer.is_fitted = True
        
        logger.info("Loaded feature normalization statistics")
    
    def get_data_loader(self, split: str = 'train') -> 'DataIterator':
        """
        Get data iterator for specified split.
        
        Args:
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Data iterator
        """
        if not self.is_preprocessed:
            # Try to load existing preprocessed data
            processed_file = self.processed_dir / "training_data.h5"
            if not processed_file.exists():
                raise ValueError("Data not preprocessed. Call preprocess_data() first.")
            self.is_preprocessed = True
        
        # Load splits if not already loaded
        if self.train_indices is None:
            self.load_data_splits()
        
        # Load feature stats if not already loaded
        if not self.feature_engineer.is_fitted:
            self.load_feature_stats()
        
        # Get indices for split
        if split == 'train':
            indices = self.train_indices
            shuffle = self.config.shuffle
        elif split == 'val':
            indices = self.val_indices
            shuffle = False
        elif split == 'test':
            indices = self.test_indices
            shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}")
        
        return DataIterator(
            hdf5_file=self.processed_dir / "training_data.h5",
            indices=indices,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=self.config.drop_last,
            num_workers=self.config.num_workers,
            prefetch_batches=self.config.prefetch_batches
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        if not self.is_preprocessed:
            return {}
        
        processed_file = self.processed_dir / "training_data.h5"
        
        with HDF5DataStore(processed_file) as store:
            metadata = store.get_metadata()
            n_samples = store.get_dataset_size()
        
        info = {
            'total_samples': n_samples,
            'patch_shape': metadata['patch_shape'],
            'n_features': metadata['n_features'],
            'feature_names': self.feature_engineer.feature_names,
            'splits': {
                'train': len(self.train_indices) if self.train_indices else 0,
                'val': len(self.val_indices) if self.val_indices else 0,
                'test': len(self.test_indices) if self.test_indices else 0
            }
        }
        
        return info


class DataIterator:
    """
    Iterator for batched data loading from HDF5 storage.
    """
    
    def __init__(self,
                 hdf5_file: Path,
                 indices: List[int],
                 batch_size: int = 64,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 num_workers: int = 1,
                 prefetch_batches: int = 2):
        """
        Initialize data iterator.
        
        Args:
            hdf5_file: Path to HDF5 data file
            indices: Indices to iterate over
            batch_size: Batch size
            shuffle: Whether to shuffle indices
            drop_last: Whether to drop last incomplete batch
            num_workers: Number of worker threads
            prefetch_batches: Number of batches to prefetch
        """
        self.hdf5_file = hdf5_file
        self.indices = np.array(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        
        self.n_samples = len(indices)
        self.n_batches = self._calculate_n_batches()
        
        # State
        self.current_epoch = 0
        self.current_batch = 0
        self._epoch_indices = None
        
        logger.debug(f"Created data iterator: {self.n_samples} samples, {self.n_batches} batches")
    
    def _calculate_n_batches(self) -> int:
        """Calculate number of batches per epoch."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def _shuffle_indices(self) -> None:
        """Shuffle indices for new epoch."""
        if self.shuffle:
            rng = np.random.RandomState(self.current_epoch)
            self._epoch_indices = rng.permutation(self.indices)
        else:
            self._epoch_indices = self.indices.copy()
    
    def __iter__(self):
        """Start new epoch."""
        self.current_batch = 0
        self._shuffle_indices()
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get next batch."""
        if self.current_batch >= self.n_batches:
            self.current_epoch += 1
            raise StopIteration
        
        # Calculate batch indices
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        if self.drop_last and end_idx - start_idx < self.batch_size:
            self.current_epoch += 1
            raise StopIteration
        
        batch_indices = self._epoch_indices[start_idx:end_idx]
        
        # Load batch data
        with HDF5DataStore(self.hdf5_file) as store:
            patches, point_features, targets = store.read_batch(batch_indices)
        
        self.current_batch += 1
        
        return patches, point_features, targets
    
    def __len__(self) -> int:
        """Get number of batches per epoch."""
        return self.n_batches
    
    def get_sample_batch(self, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a small sample batch for testing.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Tuple of (patches, point_features, targets)
        """
        sample_indices = self.indices[:n_samples]
        
        with HDF5DataStore(self.hdf5_file) as store:
            return store.read_batch(sample_indices)