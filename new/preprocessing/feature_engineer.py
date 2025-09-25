"""
Feature engineering utilities for MDA-CNN volatility surface modeling.

This module provides functionality to create and normalize point features
for the MLP branch of the MDA-CNN model, including SABR parameters,
market conditions, and derived features.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import warnings

from ..data_generation.sabr_params import SABRParams


@dataclass
class FeatureConfig:
    """
    Configuration for feature engineering.
    
    Attributes:
        include_sabr_params: Whether to include raw SABR parameters
        include_market_features: Whether to include strike/maturity features
        include_hagan_vol: Whether to include Hagan volatility as feature
        include_derived_features: Whether to include derived/engineered features
        normalize_features: Whether to apply feature normalization
        log_transform_features: List of feature names to log-transform
        standardize_features: Whether to standardize features (mean=0, std=1)
        robust_scaling: Whether to use robust scaling (median, IQR)
    """
    include_sabr_params: bool = True
    include_market_features: bool = True
    include_hagan_vol: bool = True
    include_derived_features: bool = True
    normalize_features: bool = True
    log_transform_features: List[str] = None
    standardize_features: bool = True
    robust_scaling: bool = False
    
    def __post_init__(self):
        if self.log_transform_features is None:
            self.log_transform_features = ['F0', 'alpha', 'nu']


@dataclass
class FeatureStats:
    """
    Statistics for feature normalization.
    
    Attributes:
        means: Feature means
        stds: Feature standard deviations
        medians: Feature medians (for robust scaling)
        iqrs: Feature interquartile ranges (for robust scaling)
        mins: Feature minimums
        maxs: Feature maximums
        feature_names: Names of features
    """
    means: np.ndarray
    stds: np.ndarray
    medians: np.ndarray
    iqrs: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    feature_names: List[str]


class FeatureEngineer:
    """
    Create and normalize point features for MLP input.
    
    This class handles the creation of point features from SABR parameters,
    market conditions (strike, maturity), and Hagan volatility values,
    with proper normalization and scaling.
    """
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize feature engineer.
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        self.feature_stats: Optional[FeatureStats] = None
        self.is_fitted = False
        
        # Define feature names
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names based on configuration."""
        names = []
        
        if self.config.include_sabr_params:
            names.extend(['F0', 'alpha', 'beta', 'nu', 'rho'])
        
        if self.config.include_market_features:
            names.extend(['strike', 'maturity', 'moneyness', 'log_moneyness'])
        
        if self.config.include_hagan_vol:
            names.append('hagan_vol')
        
        if self.config.include_derived_features:
            names.extend([
                'alpha_beta_interaction',
                'nu_rho_interaction', 
                'time_to_expiry_sqrt',
                'vol_of_vol_scaled'
            ])
        
        return names
    
    def create_point_features(self, 
                            sabr_params: SABRParams,
                            strike: float,
                            maturity: float,
                            hagan_vol: Optional[float] = None) -> np.ndarray:
        """
        Create point features for a single data point.
        
        Args:
            sabr_params: SABR model parameters
            strike: Strike price
            maturity: Time to maturity
            hagan_vol: Hagan volatility value (optional)
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # SABR parameters
        if self.config.include_sabr_params:
            features.extend([
                sabr_params.F0,
                sabr_params.alpha,
                sabr_params.beta,
                sabr_params.nu,
                sabr_params.rho
            ])
        
        # Market features
        if self.config.include_market_features:
            moneyness = strike / sabr_params.F0
            log_moneyness = np.log(moneyness)
            
            features.extend([
                strike,
                maturity,
                moneyness,
                log_moneyness
            ])
        
        # Hagan volatility
        if self.config.include_hagan_vol:
            if hagan_vol is not None:
                features.append(hagan_vol)
            else:
                features.append(0.0)  # Placeholder
        
        # Derived features
        if self.config.include_derived_features:
            # Interaction terms
            alpha_beta_interaction = sabr_params.alpha * (sabr_params.beta ** 2)
            nu_rho_interaction = sabr_params.nu * abs(sabr_params.rho)
            
            # Time-related features
            time_to_expiry_sqrt = np.sqrt(maturity)
            
            # Scaled vol-of-vol
            vol_of_vol_scaled = sabr_params.nu * sabr_params.alpha
            
            features.extend([
                alpha_beta_interaction,
                nu_rho_interaction,
                time_to_expiry_sqrt,
                vol_of_vol_scaled
            ])
        
        return np.array(features, dtype=np.float32)
    
    def create_features_batch(self,
                            sabr_params_list: List[SABRParams],
                            strikes: np.ndarray,
                            maturities: np.ndarray,
                            hagan_vols: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create point features for multiple data points.
        
        Args:
            sabr_params_list: List of SABR parameters
            strikes: Array of strike prices
            maturities: Array of maturities
            hagan_vols: Array of Hagan volatilities (optional)
            
        Returns:
            Feature matrix of shape (n_points, n_features)
        """
        n_points = len(strikes)
        
        if len(sabr_params_list) != n_points:
            raise ValueError("Number of SABR params must match number of points")
        
        if len(maturities) != n_points:
            raise ValueError("Number of maturities must match number of points")
        
        if hagan_vols is not None and len(hagan_vols) != n_points:
            raise ValueError("Number of Hagan vols must match number of points")
        
        # Create features for each point
        features_list = []
        for i in range(n_points):
            hagan_vol = hagan_vols[i] if hagan_vols is not None else None
            
            point_features = self.create_point_features(
                sabr_params_list[i],
                strikes[i],
                maturities[i],
                hagan_vol
            )
            features_list.append(point_features)
        
        return np.array(features_list, dtype=np.float32)
    
    def fit_normalization(self, features: np.ndarray) -> None:
        """
        Fit normalization parameters on training data.
        
        Args:
            features: Training feature matrix of shape (n_samples, n_features)
        """
        if features.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {features.shape[1]}")
        
        # Handle non-finite values
        finite_mask = np.isfinite(features)
        
        # Compute statistics
        means = np.zeros(features.shape[1])
        stds = np.ones(features.shape[1])
        medians = np.zeros(features.shape[1])
        iqrs = np.ones(features.shape[1])
        mins = np.zeros(features.shape[1])
        maxs = np.ones(features.shape[1])
        
        for i in range(features.shape[1]):
            finite_values = features[finite_mask[:, i], i]
            
            if len(finite_values) > 0:
                means[i] = np.mean(finite_values)
                stds[i] = np.std(finite_values)
                medians[i] = np.median(finite_values)
                
                q25, q75 = np.percentile(finite_values, [25, 75])
                iqrs[i] = q75 - q25
                
                mins[i] = np.min(finite_values)
                maxs[i] = np.max(finite_values)
                
                # Avoid division by zero
                if stds[i] == 0:
                    stds[i] = 1.0
                if iqrs[i] == 0:
                    iqrs[i] = 1.0
        
        self.feature_stats = FeatureStats(
            means=means,
            stds=stds,
            medians=medians,
            iqrs=iqrs,
            mins=mins,
            maxs=maxs,
            feature_names=self.feature_names.copy()
        )
        
        self.is_fitted = True
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using fitted statistics.
        
        Args:
            features: Feature matrix to normalize
            
        Returns:
            Normalized feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_normalization() before normalizing")
        
        if not self.config.normalize_features:
            return features.copy()
        
        normalized_features = features.copy()
        
        # Apply log transforms first
        for feature_name in self.config.log_transform_features:
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                
                # Only log-transform positive values
                positive_mask = normalized_features[:, feature_idx] > 0
                normalized_features[positive_mask, feature_idx] = np.log(
                    normalized_features[positive_mask, feature_idx]
                )
        
        # Apply scaling
        if self.config.robust_scaling:
            # Robust scaling using median and IQR
            normalized_features = (normalized_features - self.feature_stats.medians) / self.feature_stats.iqrs
        elif self.config.standardize_features:
            # Standard scaling using mean and std
            normalized_features = (normalized_features - self.feature_stats.means) / self.feature_stats.stds
        else:
            # Min-max scaling
            ranges = self.feature_stats.maxs - self.feature_stats.mins
            ranges[ranges == 0] = 1.0  # Avoid division by zero
            normalized_features = (normalized_features - self.feature_stats.mins) / ranges
        
        # Handle non-finite values
        finite_mask = np.isfinite(normalized_features)
        normalized_features[~finite_mask] = 0.0
        
        return normalized_features
    
    def inverse_normalize_features(self, normalized_features: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized features back to original scale.
        
        Args:
            normalized_features: Normalized feature matrix
            
        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_normalization() before inverse normalizing")
        
        if not self.config.normalize_features:
            return normalized_features.copy()
        
        features = normalized_features.copy()
        
        # Inverse scaling
        if self.config.robust_scaling:
            features = features * self.feature_stats.iqrs + self.feature_stats.medians
        elif self.config.standardize_features:
            features = features * self.feature_stats.stds + self.feature_stats.means
        else:
            ranges = self.feature_stats.maxs - self.feature_stats.mins
            features = features * ranges + self.feature_stats.mins
        
        # Inverse log transforms
        for feature_name in self.config.log_transform_features:
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                features[:, feature_idx] = np.exp(features[:, feature_idx])
        
        return features
    
    def get_feature_importance_names(self) -> List[str]:
        """Get feature names for importance analysis."""
        return self.feature_names.copy()
    
    def validate_features(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Validate feature matrix and return diagnostics.
        
        Args:
            features: Feature matrix to validate
            
        Returns:
            Dictionary with validation results and diagnostics
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'diagnostics': {
                'shape': features.shape,
                'expected_features': len(self.feature_names),
                'feature_names': self.feature_names
            }
        }
        
        # Check shape
        if len(features.shape) != 2:
            validation_results['errors'].append(f"Expected 2D array, got {len(features.shape)}D")
            validation_results['is_valid'] = False
            return validation_results
        
        n_samples, n_features = features.shape
        
        if n_features != len(self.feature_names):
            validation_results['errors'].append(
                f"Expected {len(self.feature_names)} features, got {n_features}"
            )
            validation_results['is_valid'] = False
        
        # Check for non-finite values
        finite_mask = np.isfinite(features)
        finite_fraction = np.sum(finite_mask) / features.size
        validation_results['diagnostics']['finite_fraction'] = finite_fraction
        
        if finite_fraction < 0.9:
            validation_results['warnings'].append(f"Only {finite_fraction:.1%} values are finite")
        
        # Check feature ranges
        for i, feature_name in enumerate(self.feature_names):
            if i < n_features:
                finite_values = features[finite_mask[:, i], i]
                
                if len(finite_values) > 0:
                    feature_min = np.min(finite_values)
                    feature_max = np.max(finite_values)
                    feature_std = np.std(finite_values)
                    
                    validation_results['diagnostics'][f'{feature_name}_range'] = (feature_min, feature_max)
                    validation_results['diagnostics'][f'{feature_name}_std'] = feature_std
                    
                    # Check for suspicious values
                    if feature_name in ['F0', 'alpha', 'strike'] and feature_min <= 0:
                        validation_results['warnings'].append(f"{feature_name} has non-positive values")
                    
                    if feature_name == 'beta' and (feature_min < 0 or feature_max > 1):
                        validation_results['warnings'].append(f"Beta outside [0,1] range: [{feature_min:.3f}, {feature_max:.3f}]")
                    
                    if feature_name == 'rho' and (feature_min < -1 or feature_max > 1):
                        validation_results['warnings'].append(f"Rho outside [-1,1] range: [{feature_min:.3f}, {feature_max:.3f}]")
                    
                    if feature_name == 'nu' and feature_min < 0:
                        validation_results['warnings'].append(f"Nu has negative values")
                    
                    if feature_std == 0:
                        validation_results['warnings'].append(f"{feature_name} has zero variance")
        
        return validation_results
    
    def get_feature_correlations(self, features: np.ndarray) -> np.ndarray:
        """
        Compute feature correlation matrix.
        
        Args:
            features: Feature matrix
            
        Returns:
            Correlation matrix
        """
        finite_mask = np.all(np.isfinite(features), axis=1)
        finite_features = features[finite_mask]
        
        if len(finite_features) < 2:
            return np.eye(features.shape[1])
        
        return np.corrcoef(finite_features.T)
    
    def detect_feature_outliers(self, features: np.ndarray, 
                              threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers in features using z-score method.
        
        Args:
            features: Feature matrix
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean mask indicating outliers
        """
        if not self.is_fitted:
            warnings.warn("Feature normalization not fitted, using sample statistics")
            means = np.nanmean(features, axis=0)
            stds = np.nanstd(features, axis=0)
        else:
            means = self.feature_stats.means
            stds = self.feature_stats.stds
        
        # Compute z-scores
        z_scores = np.abs((features - means) / stds)
        
        # Handle non-finite values
        z_scores[~np.isfinite(z_scores)] = 0
        
        # Outlier if any feature has z-score > threshold
        outlier_mask = np.any(z_scores > threshold, axis=1)
        
        return outlier_mask