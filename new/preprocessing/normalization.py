"""
Data normalization and scaling utilities for SABR volatility surface modeling.

This module provides comprehensive normalization and scaling utilities for
both surface patches and point features, with support for different scaling
strategies and robust handling of non-finite values.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import warnings
import pickle
from pathlib import Path

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class NormalizationStats:
    """
    Statistics for data normalization.
    
    Attributes:
        means: Feature/patch means
        stds: Feature/patch standard deviations
        medians: Feature/patch medians (for robust scaling)
        iqrs: Feature/patch interquartile ranges (for robust scaling)
        mins: Feature/patch minimums
        maxs: Feature/patch maximums
        percentiles: Additional percentiles for robust scaling
        n_samples: Number of samples used to compute statistics
    """
    means: np.ndarray
    stds: np.ndarray
    medians: np.ndarray
    iqrs: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    percentiles: Dict[float, np.ndarray]
    n_samples: int


class PatchNormalizer:
    """
    Normalizer for surface patches.
    """
    
    def __init__(self, 
                 method: str = 'local',
                 robust: bool = False,
                 clip_outliers: bool = True,
                 outlier_percentiles: Tuple[float, float] = (1.0, 99.0)):
        """
        Initialize patch normalizer.
        
        Args:
            method: Normalization method ('local', 'global', 'standardize')
            robust: Whether to use robust statistics (median, IQR)
            clip_outliers: Whether to clip outliers
            outlier_percentiles: Percentiles for outlier clipping
        """
        self.method = method
        self.robust = robust
        self.clip_outliers = clip_outliers
        self.outlier_percentiles = outlier_percentiles
        
        self.stats: Optional[NormalizationStats] = None
        self.is_fitted = False
        
        if method not in ['local', 'global', 'standardize']:
            raise ValueError(f"Invalid normalization method: {method}")
    
    def fit(self, patches: np.ndarray) -> None:
        """
        Fit normalization parameters on training patches.
        
        Args:
            patches: Training patches of shape (n_samples, height, width)
        """
        if len(patches.shape) != 3:
            raise ValueError(f"Expected 3D patches array, got {len(patches.shape)}D")
        
        n_samples, height, width = patches.shape
        logger.info(f"Fitting patch normalizer on {n_samples} patches of size {height}x{width}")
        
        if self.method == 'local':
            # Local normalization doesn't need global statistics
            self.is_fitted = True
            return
        
        # Flatten patches for global statistics
        if self.method == 'global':
            # Compute statistics across all patches and spatial dimensions
            flat_patches = patches.reshape(-1)
        else:  # standardize
            # Compute statistics per spatial location
            flat_patches = patches.reshape(n_samples, -1)
        
        # Remove non-finite values
        finite_mask = np.isfinite(flat_patches)
        
        if self.method == 'global':
            finite_values = flat_patches[finite_mask]
            
            if len(finite_values) == 0:
                raise ValueError("No finite values found in patches")
            
            # Compute statistics
            stats_dict = self._compute_statistics(finite_values)
            
            # Broadcast to patch shape for consistent interface
            self.stats = NormalizationStats(
                means=np.full((height, width), stats_dict['mean']),
                stds=np.full((height, width), stats_dict['std']),
                medians=np.full((height, width), stats_dict['median']),
                iqrs=np.full((height, width), stats_dict['iqr']),
                mins=np.full((height, width), stats_dict['min']),
                maxs=np.full((height, width), stats_dict['max']),
                percentiles={p: np.full((height, width), v) for p, v in stats_dict['percentiles'].items()},
                n_samples=n_samples
            )
            
        else:  # standardize
            # Compute statistics per spatial location
            means = np.zeros((height, width))
            stds = np.ones((height, width))
            medians = np.zeros((height, width))
            iqrs = np.ones((height, width))
            mins = np.zeros((height, width))
            maxs = np.ones((height, width))
            percentiles = {p: np.zeros((height, width)) for p in [1, 5, 95, 99]}
            
            for i in range(height):
                for j in range(width):
                    pixel_values = patches[:, i, j]
                    finite_pixel_mask = np.isfinite(pixel_values)
                    finite_pixel_values = pixel_values[finite_pixel_mask]
                    
                    if len(finite_pixel_values) > 0:
                        stats_dict = self._compute_statistics(finite_pixel_values)
                        
                        means[i, j] = stats_dict['mean']
                        stds[i, j] = stats_dict['std']
                        medians[i, j] = stats_dict['median']
                        iqrs[i, j] = stats_dict['iqr']
                        mins[i, j] = stats_dict['min']
                        maxs[i, j] = stats_dict['max']
                        
                        for p, v in stats_dict['percentiles'].items():
                            percentiles[p][i, j] = v
            
            self.stats = NormalizationStats(
                means=means,
                stds=stds,
                medians=medians,
                iqrs=iqrs,
                mins=mins,
                maxs=maxs,
                percentiles=percentiles,
                n_samples=n_samples
            )
        
        self.is_fitted = True
        logger.info("Patch normalizer fitted successfully")
    
    def _compute_statistics(self, values: np.ndarray) -> Dict[str, Any]:
        """Compute normalization statistics for values."""
        if len(values) == 0:
            return {
                'mean': 0.0, 'std': 1.0, 'median': 0.0, 'iqr': 1.0,
                'min': 0.0, 'max': 1.0, 'percentiles': {1: 0.0, 5: 0.0, 95: 1.0, 99: 1.0}
            }
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        
        q25, q75 = np.percentile(values, [25, 75])
        iqr_val = q75 - q25
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        percentiles = {}
        for p in [1, 5, 95, 99]:
            percentiles[p] = np.percentile(values, p)
        
        # Avoid division by zero
        if std_val == 0:
            std_val = 1.0
        if iqr_val == 0:
            iqr_val = 1.0
        
        return {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'iqr': iqr_val,
            'min': min_val,
            'max': max_val,
            'percentiles': percentiles
        }
    
    def transform(self, patches: np.ndarray) -> np.ndarray:
        """
        Normalize patches using fitted parameters.
        
        Args:
            patches: Patches to normalize
            
        Returns:
            Normalized patches
        """
        if self.method == 'local':
            return self._normalize_local(patches)
        
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        if self.method == 'global':
            return self._normalize_global(patches)
        else:  # standardize
            return self._normalize_standardize(patches)
    
    def _normalize_local(self, patches: np.ndarray) -> np.ndarray:
        """Apply local normalization to each patch."""
        normalized_patches = np.zeros_like(patches)
        
        for i in range(patches.shape[0]):
            patch = patches[i]
            finite_mask = np.isfinite(patch)
            
            if not np.any(finite_mask):
                # All non-finite values
                normalized_patches[i] = np.zeros_like(patch)
                continue
            
            finite_values = patch[finite_mask]
            
            if len(finite_values) <= 1:
                # Constant or single value
                normalized_patches[i] = np.zeros_like(patch)
                continue
            
            if self.robust:
                center = np.median(finite_values)
                q25, q75 = np.percentile(finite_values, [25, 75])
                scale = q75 - q25
                if scale == 0:
                    scale = 1.0
            else:
                center = np.mean(finite_values)
                scale = np.std(finite_values)
                if scale == 0:
                    scale = 1.0
            
            # Normalize
            normalized_patch = (patch - center) / scale
            
            # Handle non-finite values
            normalized_patch[~finite_mask] = 0.0
            
            # Clip outliers if requested
            if self.clip_outliers:
                p_low, p_high = self.outlier_percentiles
                finite_normalized = normalized_patch[finite_mask]
                
                if len(finite_normalized) > 0:
                    low_clip = np.percentile(finite_normalized, p_low)
                    high_clip = np.percentile(finite_normalized, p_high)
                    normalized_patch = np.clip(normalized_patch, low_clip, high_clip)
            
            normalized_patches[i] = normalized_patch
        
        return normalized_patches
    
    def _normalize_global(self, patches: np.ndarray) -> np.ndarray:
        """Apply global normalization using fitted statistics."""
        if self.robust:
            center = self.stats.medians[0, 0]  # Global value
            scale = self.stats.iqrs[0, 0]
        else:
            center = self.stats.means[0, 0]
            scale = self.stats.stds[0, 0]
        
        normalized_patches = (patches - center) / scale
        
        # Handle non-finite values
        finite_mask = np.isfinite(normalized_patches)
        normalized_patches[~finite_mask] = 0.0
        
        # Clip outliers if requested
        if self.clip_outliers:
            p_low, p_high = self.outlier_percentiles
            low_clip = self.stats.percentiles[p_low][0, 0]
            high_clip = self.stats.percentiles[p_high][0, 0]
            
            # Transform clip values
            low_clip = (low_clip - center) / scale
            high_clip = (high_clip - center) / scale
            
            normalized_patches = np.clip(normalized_patches, low_clip, high_clip)
        
        return normalized_patches
    
    def _normalize_standardize(self, patches: np.ndarray) -> np.ndarray:
        """Apply per-pixel standardization."""
        if self.robust:
            center = self.stats.medians
            scale = self.stats.iqrs
        else:
            center = self.stats.means
            scale = self.stats.stds
        
        normalized_patches = (patches - center) / scale
        
        # Handle non-finite values
        finite_mask = np.isfinite(normalized_patches)
        normalized_patches[~finite_mask] = 0.0
        
        return normalized_patches
    
    def inverse_transform(self, normalized_patches: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized patches back to original scale.
        
        Args:
            normalized_patches: Normalized patches
            
        Returns:
            Patches in original scale
        """
        if self.method == 'local':
            warnings.warn("Cannot inverse transform locally normalized patches")
            return normalized_patches
        
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        
        if self.method == 'global':
            if self.robust:
                center = self.stats.medians[0, 0]
                scale = self.stats.iqrs[0, 0]
            else:
                center = self.stats.means[0, 0]
                scale = self.stats.stds[0, 0]
            
            return normalized_patches * scale + center
        
        else:  # standardize
            if self.robust:
                center = self.stats.medians
                scale = self.stats.iqrs
            else:
                center = self.stats.means
                scale = self.stats.stds
            
            return normalized_patches * scale + center
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save normalizer state."""
        if not self.is_fitted and self.method != 'local':
            warnings.warn("Saving unfitted normalizer")
        
        state = {
            'method': self.method,
            'robust': self.robust,
            'clip_outliers': self.clip_outliers,
            'outlier_percentiles': self.outlier_percentiles,
            'stats': self.stats,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load normalizer state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.method = state['method']
        self.robust = state['robust']
        self.clip_outliers = state['clip_outliers']
        self.outlier_percentiles = state['outlier_percentiles']
        self.stats = state['stats']
        self.is_fitted = state['is_fitted']


class FeatureScaler:
    """
    Scaler for point features with multiple scaling strategies.
    """
    
    def __init__(self,
                 method: str = 'standard',
                 robust: bool = False,
                 feature_range: Tuple[float, float] = (0, 1),
                 clip_outliers: bool = True,
                 outlier_std_threshold: float = 3.0):
        """
        Initialize feature scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust', 'quantile')
            robust: Whether to use robust statistics
            feature_range: Target range for minmax scaling
            clip_outliers: Whether to clip outliers
            outlier_std_threshold: Standard deviation threshold for outlier clipping
        """
        self.method = method
        self.robust = robust
        self.feature_range = feature_range
        self.clip_outliers = clip_outliers
        self.outlier_std_threshold = outlier_std_threshold
        
        self.stats: Optional[NormalizationStats] = None
        self.is_fitted = False
        
        if method not in ['standard', 'minmax', 'robust', 'quantile']:
            raise ValueError(f"Invalid scaling method: {method}")
    
    def fit(self, features: np.ndarray) -> None:
        """
        Fit scaling parameters on training features.
        
        Args:
            features: Training features of shape (n_samples, n_features)
        """
        if len(features.shape) != 2:
            raise ValueError(f"Expected 2D features array, got {len(features.shape)}D")
        
        n_samples, n_features = features.shape
        logger.info(f"Fitting feature scaler on {n_samples} samples with {n_features} features")
        
        # Compute statistics per feature
        means = np.zeros(n_features)
        stds = np.ones(n_features)
        medians = np.zeros(n_features)
        iqrs = np.ones(n_features)
        mins = np.zeros(n_features)
        maxs = np.ones(n_features)
        percentiles = {p: np.zeros(n_features) for p in [1, 5, 10, 25, 75, 90, 95, 99]}
        
        for i in range(n_features):
            feature_values = features[:, i]
            finite_mask = np.isfinite(feature_values)
            finite_values = feature_values[finite_mask]
            
            if len(finite_values) > 0:
                stats_dict = self._compute_feature_statistics(finite_values)
                
                means[i] = stats_dict['mean']
                stds[i] = stats_dict['std']
                medians[i] = stats_dict['median']
                iqrs[i] = stats_dict['iqr']
                mins[i] = stats_dict['min']
                maxs[i] = stats_dict['max']
                
                for p, v in stats_dict['percentiles'].items():
                    percentiles[p][i] = v
        
        self.stats = NormalizationStats(
            means=means,
            stds=stds,
            medians=medians,
            iqrs=iqrs,
            mins=mins,
            maxs=maxs,
            percentiles=percentiles,
            n_samples=n_samples
        )
        
        self.is_fitted = True
        logger.info("Feature scaler fitted successfully")
    
    def _compute_feature_statistics(self, values: np.ndarray) -> Dict[str, Any]:
        """Compute scaling statistics for feature values."""
        if len(values) == 0:
            return {
                'mean': 0.0, 'std': 1.0, 'median': 0.0, 'iqr': 1.0,
                'min': 0.0, 'max': 1.0, 
                'percentiles': {p: 0.0 if p < 50 else 1.0 for p in [1, 5, 10, 25, 75, 90, 95, 99]}
            }
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        
        q25, q75 = np.percentile(values, [25, 75])
        iqr_val = q75 - q25
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        percentiles = {}
        for p in [1, 5, 10, 25, 75, 90, 95, 99]:
            percentiles[p] = np.percentile(values, p)
        
        # Avoid division by zero
        if std_val == 0:
            std_val = 1.0
        if iqr_val == 0:
            iqr_val = 1.0
        
        return {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'iqr': iqr_val,
            'min': min_val,
            'max': max_val,
            'percentiles': percentiles
        }
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Scale features using fitted parameters.
        
        Args:
            features: Features to scale
            
        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        if self.method == 'standard':
            return self._scale_standard(features)
        elif self.method == 'minmax':
            return self._scale_minmax(features)
        elif self.method == 'robust':
            return self._scale_robust(features)
        else:  # quantile
            return self._scale_quantile(features)
    
    def _scale_standard(self, features: np.ndarray) -> np.ndarray:
        """Apply standard scaling (z-score normalization)."""
        if self.robust:
            center = self.stats.medians
            scale = self.stats.iqrs
        else:
            center = self.stats.means
            scale = self.stats.stds
        
        scaled_features = (features - center) / scale
        
        # Handle non-finite values
        finite_mask = np.isfinite(scaled_features)
        scaled_features[~finite_mask] = 0.0
        
        # Clip outliers if requested
        if self.clip_outliers:
            scaled_features = np.clip(
                scaled_features, 
                -self.outlier_std_threshold, 
                self.outlier_std_threshold
            )
        
        return scaled_features
    
    def _scale_minmax(self, features: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to specified range."""
        min_val, max_val = self.feature_range
        
        # Use robust percentiles if requested
        if self.robust:
            data_min = self.stats.percentiles[5]
            data_max = self.stats.percentiles[95]
        else:
            data_min = self.stats.mins
            data_max = self.stats.maxs
        
        # Avoid division by zero
        data_range = data_max - data_min
        data_range[data_range == 0] = 1.0
        
        # Scale to [0, 1] then to target range
        scaled_features = (features - data_min) / data_range
        scaled_features = scaled_features * (max_val - min_val) + min_val
        
        # Handle non-finite values
        finite_mask = np.isfinite(scaled_features)
        scaled_features[~finite_mask] = min_val
        
        # Clip to target range
        scaled_features = np.clip(scaled_features, min_val, max_val)
        
        return scaled_features
    
    def _scale_robust(self, features: np.ndarray) -> np.ndarray:
        """Apply robust scaling using median and IQR."""
        center = self.stats.medians
        scale = self.stats.iqrs
        
        scaled_features = (features - center) / scale
        
        # Handle non-finite values
        finite_mask = np.isfinite(scaled_features)
        scaled_features[~finite_mask] = 0.0
        
        return scaled_features
    
    def _scale_quantile(self, features: np.ndarray) -> np.ndarray:
        """Apply quantile-based scaling."""
        # Scale to [0, 1] using 5th and 95th percentiles
        p5 = self.stats.percentiles[5]
        p95 = self.stats.percentiles[95]
        
        scale_range = p95 - p5
        scale_range[scale_range == 0] = 1.0
        
        scaled_features = (features - p5) / scale_range
        
        # Handle non-finite values
        finite_mask = np.isfinite(scaled_features)
        scaled_features[~finite_mask] = 0.0
        
        # Clip outliers
        scaled_features = np.clip(scaled_features, 0, 1)
        
        return scaled_features
    
    def inverse_transform(self, scaled_features: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            scaled_features: Scaled features
            
        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted")
        
        if self.method == 'standard':
            if self.robust:
                center = self.stats.medians
                scale = self.stats.iqrs
            else:
                center = self.stats.means
                scale = self.stats.stds
            
            return scaled_features * scale + center
        
        elif self.method == 'minmax':
            min_val, max_val = self.feature_range
            
            if self.robust:
                data_min = self.stats.percentiles[5]
                data_max = self.stats.percentiles[95]
            else:
                data_min = self.stats.mins
                data_max = self.stats.maxs
            
            data_range = data_max - data_min
            
            # Scale from target range to [0, 1] then to original range
            features = (scaled_features - min_val) / (max_val - min_val)
            features = features * data_range + data_min
            
            return features
        
        elif self.method == 'robust':
            center = self.stats.medians
            scale = self.stats.iqrs
            
            return scaled_features * scale + center
        
        else:  # quantile
            p5 = self.stats.percentiles[5]
            p95 = self.stats.percentiles[95]
            scale_range = p95 - p5
            
            return scaled_features * scale_range + p5
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save scaler state."""
        if not self.is_fitted:
            warnings.warn("Saving unfitted scaler")
        
        state = {
            'method': self.method,
            'robust': self.robust,
            'feature_range': self.feature_range,
            'clip_outliers': self.clip_outliers,
            'outlier_std_threshold': self.outlier_std_threshold,
            'stats': self.stats,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load scaler state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.method = state['method']
        self.robust = state['robust']
        self.feature_range = state['feature_range']
        self.clip_outliers = state['clip_outliers']
        self.outlier_std_threshold = state['outlier_std_threshold']
        self.stats = state['stats']
        self.is_fitted = state['is_fitted']


def create_data_splits(n_samples: int,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_seed: int = 42,
                      stratify_by: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/validation/test splits with proper indexing.
    
    Args:
        n_samples: Total number of samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility
        stratify_by: Optional array for stratified splitting
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    rng = np.random.RandomState(random_seed)
    indices = np.arange(n_samples)
    
    if stratify_by is not None:
        # Stratified splitting (simplified implementation)
        unique_labels = np.unique(stratify_by)
        train_indices = []
        val_indices = []
        test_indices = []
        
        for label in unique_labels:
            label_indices = indices[stratify_by == label]
            rng.shuffle(label_indices)
            
            n_label = len(label_indices)
            n_train = int(n_label * train_ratio)
            n_val = int(n_label * val_ratio)
            
            train_indices.extend(label_indices[:n_train])
            val_indices.extend(label_indices[n_train:n_train + n_val])
            test_indices.extend(label_indices[n_train + n_val:])
        
        # Convert to arrays and shuffle
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)
        
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)
        
    else:
        # Random splitting
        rng.shuffle(indices)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
    
    logger.info(f"Created data splits: {len(train_indices)} train, "
               f"{len(val_indices)} val, {len(test_indices)} test")
    
    return train_indices, val_indices, test_indices