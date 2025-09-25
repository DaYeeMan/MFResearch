"""
Example usage of the data preprocessing and loading pipeline.

This script demonstrates how to use the SABR data loader to preprocess
raw surface data and create efficient data loaders for training.
"""

import numpy as np
from pathlib import Path
import time

from .data_loader import DataLoaderConfig, SABRDataLoader
from .patch_extractor import PatchConfig
from .feature_engineer import FeatureConfig
from .normalization import PatchNormalizer, FeatureScaler
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def demonstrate_data_loading():
    """Demonstrate the complete data loading pipeline."""
    
    # Configuration
    data_dir = Path("new/data")  # Assumes data has been generated
    
    # Check if data exists
    if not (data_dir / "raw").exists():
        logger.error(f"Raw data directory not found: {data_dir / 'raw'}")
        logger.info("Please run data generation first")
        return
    
    # Configure components
    patch_config = PatchConfig(
        patch_size=(9, 9),
        boundary_mode='reflect',
        normalize_patches=True
    )
    
    feature_config = FeatureConfig(
        include_sabr_params=True,
        include_market_features=True,
        include_hagan_vol=True,
        include_derived_features=True,
        normalize_features=True,
        standardize_features=True
    )
    
    loader_config = DataLoaderConfig(
        batch_size=64,
        shuffle=True,
        num_workers=2,
        hf_budget_per_surface=200,
        validation_split=0.15,
        test_split=0.15,
        random_seed=42
    )
    
    # Create data loader
    logger.info("Creating SABR data loader...")
    data_loader = SABRDataLoader(
        data_dir=data_dir,
        patch_config=patch_config,
        feature_config=feature_config,
        loader_config=loader_config
    )
    
    # Preprocess data (this may take some time)
    logger.info("Starting data preprocessing...")
    start_time = time.time()
    
    data_loader.preprocess_data(force_reprocess=False)
    
    preprocessing_time = time.time() - start_time
    logger.info(f"Data preprocessing completed in {preprocessing_time:.2f} seconds")
    
    # Get dataset information
    info = data_loader.get_dataset_info()
    logger.info("Dataset Information:")
    logger.info(f"  Total samples: {info['total_samples']}")
    logger.info(f"  Patch shape: {info['patch_shape']}")
    logger.info(f"  Number of features: {info['n_features']}")
    logger.info(f"  Feature names: {info['feature_names']}")
    logger.info(f"  Data splits: {info['splits']}")
    
    # Create data loaders for each split
    logger.info("\nCreating data iterators...")
    
    train_loader = data_loader.get_data_loader('train')
    val_loader = data_loader.get_data_loader('val')
    test_loader = data_loader.get_data_loader('test')
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Validation loader: {len(val_loader)} batches")
    logger.info(f"Test loader: {len(test_loader)} batches")
    
    # Demonstrate data loading
    logger.info("\nDemonstrating data loading...")
    
    # Get sample batches
    train_patches, train_features, train_targets = train_loader.get_sample_batch(10)
    logger.info(f"Sample train batch shapes:")
    logger.info(f"  Patches: {train_patches.shape}")
    logger.info(f"  Features: {train_features.shape}")
    logger.info(f"  Targets: {train_targets.shape}")
    
    # Check data statistics
    logger.info(f"\nSample data statistics:")
    logger.info(f"  Patch range: [{np.min(train_patches):.4f}, {np.max(train_patches):.4f}]")
    logger.info(f"  Feature range: [{np.min(train_features):.4f}, {np.max(train_features):.4f}]")
    logger.info(f"  Target range: [{np.min(train_targets):.4f}, {np.max(train_targets):.4f}]")
    
    # Demonstrate iteration
    logger.info("\nDemonstrating batch iteration...")
    
    start_time = time.time()
    batch_count = 0
    sample_count = 0
    
    for patches, features, targets in train_loader:
        batch_count += 1
        sample_count += patches.shape[0]
        
        if batch_count >= 5:  # Just demonstrate first 5 batches
            break
    
    iteration_time = time.time() - start_time
    logger.info(f"Loaded {batch_count} batches ({sample_count} samples) in {iteration_time:.3f} seconds")
    logger.info(f"Loading rate: {sample_count / iteration_time:.1f} samples/second")
    
    # Demonstrate feature analysis
    logger.info("\nFeature analysis:")
    
    # Get larger sample for analysis
    analysis_patches, analysis_features, analysis_targets = train_loader.get_sample_batch(100)
    
    # Feature statistics
    feature_means = np.mean(analysis_features, axis=0)
    feature_stds = np.std(analysis_features, axis=0)
    
    logger.info("Feature statistics (mean ± std):")
    for i, name in enumerate(info['feature_names']):
        logger.info(f"  {name}: {feature_means[i]:.4f} ± {feature_stds[i]:.4f}")
    
    # Patch statistics
    patch_mean = np.mean(analysis_patches)
    patch_std = np.std(analysis_patches)
    logger.info(f"\nPatch statistics:")
    logger.info(f"  Mean: {patch_mean:.4f}")
    logger.info(f"  Std: {patch_std:.4f}")
    logger.info(f"  Shape: {analysis_patches.shape}")
    
    # Target statistics
    target_mean = np.mean(analysis_targets)
    target_std = np.std(analysis_targets)
    logger.info(f"\nTarget statistics:")
    logger.info(f"  Mean: {target_mean:.4f}")
    logger.info(f"  Std: {target_std:.4f}")
    logger.info(f"  Range: [{np.min(analysis_targets):.4f}, {np.max(analysis_targets):.4f}]")


def demonstrate_normalization():
    """Demonstrate normalization utilities."""
    
    logger.info("\n" + "="*50)
    logger.info("NORMALIZATION DEMONSTRATION")
    logger.info("="*50)
    
    # Create sample data
    n_samples = 1000
    
    # Sample patches with different characteristics
    patches = np.zeros((n_samples, 9, 9))
    for i in range(n_samples):
        # Varying means and scales
        mean_val = np.random.uniform(-0.5, 0.5)
        scale_val = np.random.uniform(0.1, 2.0)
        patches[i] = np.random.normal(mean_val, scale_val, (9, 9))
    
    # Sample features with different scales
    features = np.zeros((n_samples, 8))
    features[:, 0] = np.random.uniform(80, 120, n_samples)  # Strike
    features[:, 1] = np.random.uniform(0.1, 2.0, n_samples)  # Maturity
    features[:, 2] = np.random.uniform(0.1, 0.5, n_samples)  # Alpha
    features[:, 3] = np.random.uniform(0, 1, n_samples)  # Beta
    features[:, 4] = np.random.uniform(0.1, 0.8, n_samples)  # Nu
    features[:, 5] = np.random.uniform(-0.9, 0.9, n_samples)  # Rho
    features[:, 6] = np.random.uniform(0.8, 1.2, n_samples)  # Moneyness
    features[:, 7] = np.random.uniform(0.1, 0.4, n_samples)  # Hagan vol
    
    logger.info(f"Created sample data: {n_samples} samples")
    logger.info(f"  Patches shape: {patches.shape}")
    logger.info(f"  Features shape: {features.shape}")
    
    # Demonstrate patch normalization
    logger.info("\nPatch Normalization:")
    
    for method in ['local', 'global', 'standardize']:
        logger.info(f"\n  {method.upper()} normalization:")
        
        normalizer = PatchNormalizer(method=method, robust=False)
        
        if method != 'local':
            normalizer.fit(patches[:800])  # Fit on training subset
        
        normalized_patches = normalizer.transform(patches[:100])  # Transform test subset
        
        # Statistics
        original_mean = np.mean(patches[:100])
        original_std = np.std(patches[:100])
        normalized_mean = np.mean(normalized_patches)
        normalized_std = np.std(normalized_patches)
        
        logger.info(f"    Original: mean={original_mean:.4f}, std={original_std:.4f}")
        logger.info(f"    Normalized: mean={normalized_mean:.4f}, std={normalized_std:.4f}")
    
    # Demonstrate feature scaling
    logger.info("\nFeature Scaling:")
    
    for method in ['standard', 'minmax', 'robust']:
        logger.info(f"\n  {method.upper()} scaling:")
        
        scaler = FeatureScaler(method=method)
        scaler.fit(features[:800])  # Fit on training subset
        scaled_features = scaler.transform(features[:100])  # Transform test subset
        
        # Statistics per feature
        logger.info("    Feature statistics (first 4 features):")
        for i in range(4):
            orig_mean = np.mean(features[:100, i])
            orig_std = np.std(features[:100, i])
            scaled_mean = np.mean(scaled_features[:, i])
            scaled_std = np.std(scaled_features[:, i])
            
            logger.info(f"      Feature {i}: {orig_mean:.3f}±{orig_std:.3f} → {scaled_mean:.3f}±{scaled_std:.3f}")


def demonstrate_performance():
    """Demonstrate performance characteristics."""
    
    logger.info("\n" + "="*50)
    logger.info("PERFORMANCE DEMONSTRATION")
    logger.info("="*50)
    
    # This would require actual data to be meaningful
    data_dir = Path("new/data")
    
    if not (data_dir / "processed" / "training_data.h5").exists():
        logger.info("No preprocessed data found for performance testing")
        return
    
    # Configure for performance testing
    loader_config = DataLoaderConfig(
        batch_size=128,
        shuffle=True,
        num_workers=4,
        prefetch_batches=3
    )
    
    patch_config = PatchConfig(patch_size=(9, 9))
    feature_config = FeatureConfig()
    
    data_loader = SABRDataLoader(data_dir, patch_config, feature_config, loader_config)
    
    # Performance test
    train_loader = data_loader.get_data_loader('train')
    
    logger.info(f"Performance test with {len(train_loader)} batches")
    
    # Time full epoch
    start_time = time.time()
    total_samples = 0
    batch_count = 0
    
    for patches, features, targets in train_loader:
        total_samples += patches.shape[0]
        batch_count += 1
        
        if batch_count % 50 == 0:
            elapsed = time.time() - start_time
            rate = total_samples / elapsed
            logger.info(f"  Processed {batch_count} batches, {total_samples} samples, {rate:.1f} samples/sec")
    
    total_time = time.time() - start_time
    final_rate = total_samples / total_time
    
    logger.info(f"\nPerformance Results:")
    logger.info(f"  Total time: {total_time:.2f} seconds")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Average rate: {final_rate:.1f} samples/second")
    logger.info(f"  Batches per second: {batch_count / total_time:.1f}")


if __name__ == "__main__":
    # Run demonstrations
    try:
        demonstrate_data_loading()
        demonstrate_normalization()
        demonstrate_performance()
        
        logger.info("\n" + "="*50)
        logger.info("DATA LOADING DEMONSTRATION COMPLETED")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise