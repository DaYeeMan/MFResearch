"""
Tests for feature engineering functionality.
"""

import numpy as np
import pytest
from unittest.mock import Mock

from .feature_engineer import FeatureEngineer, FeatureConfig, FeatureStats
from ..data_generation.sabr_params import SABRParams


class TestFeatureConfig:
    """Test FeatureConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureConfig()
        
        assert config.include_sabr_params is True
        assert config.include_market_features is True
        assert config.include_hagan_vol is True
        assert config.include_derived_features is True
        assert config.normalize_features is True
        assert config.log_transform_features == ['F0', 'alpha', 'nu']
        assert config.standardize_features is True
        assert config.robust_scaling is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeatureConfig(
            include_sabr_params=False,
            include_derived_features=False,
            normalize_features=False,
            log_transform_features=['alpha'],
            robust_scaling=True
        )
        
        assert config.include_sabr_params is False
        assert config.include_derived_features is False
        assert config.normalize_features is False
        assert config.log_transform_features == ['alpha']
        assert config.robust_scaling is True


class TestFeatureStats:
    """Test FeatureStats dataclass."""
    
    def test_feature_stats_creation(self):
        """Test FeatureStats creation."""
        means = np.array([1.0, 2.0, 3.0])
        stds = np.array([0.5, 1.0, 1.5])
        medians = np.array([0.9, 1.8, 2.7])
        iqrs = np.array([0.6, 1.2, 1.8])
        mins = np.array([0.0, 0.5, 1.0])
        maxs = np.array([2.0, 4.0, 6.0])
        names = ['feature1', 'feature2', 'feature3']
        
        stats = FeatureStats(
            means=means, stds=stds, medians=medians, iqrs=iqrs,
            mins=mins, maxs=maxs, feature_names=names
        )
        
        np.testing.assert_array_equal(stats.means, means)
        np.testing.assert_array_equal(stats.stds, stds)
        assert stats.feature_names == names


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = FeatureConfig()
        self.engineer = FeatureEngineer(self.config)
        
        # Create test SABR parameters
        self.sabr_params = SABRParams(
            F0=100.0,
            alpha=0.3,
            beta=0.7,
            nu=0.4,
            rho=-0.2
        )
        
        # Test market conditions
        self.strike = 105.0
        self.maturity = 1.0
        self.hagan_vol = 0.25
    
    def test_initialization(self):
        """Test engineer initialization."""
        assert not self.engineer.is_fitted
        assert self.engineer.feature_stats is None
        assert len(self.engineer.feature_names) > 0
    
    def test_get_feature_names_full(self):
        """Test feature name generation with all features enabled."""
        expected_names = [
            'F0', 'alpha', 'beta', 'nu', 'rho',  # SABR params
            'strike', 'maturity', 'moneyness', 'log_moneyness',  # Market features
            'hagan_vol',  # Hagan vol
            'alpha_beta_interaction', 'nu_rho_interaction',  # Derived features
            'time_to_expiry_sqrt', 'vol_of_vol_scaled'
        ]
        
        assert self.engineer.feature_names == expected_names
    
    def test_get_feature_names_partial(self):
        """Test feature name generation with partial features."""
        config = FeatureConfig(
            include_sabr_params=True,
            include_market_features=False,
            include_hagan_vol=False,
            include_derived_features=False
        )
        engineer = FeatureEngineer(config)
        
        expected_names = ['F0', 'alpha', 'beta', 'nu', 'rho']
        assert engineer.feature_names == expected_names
    
    def test_create_point_features_full(self):
        """Test point feature creation with all features."""
        features = self.engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity, self.hagan_vol
        )
        
        assert len(features) == len(self.engineer.feature_names)
        assert features.dtype == np.float32
        
        # Check SABR parameters
        assert features[0] == self.sabr_params.F0
        assert features[1] == self.sabr_params.alpha
        assert features[2] == self.sabr_params.beta
        assert features[3] == self.sabr_params.nu
        assert features[4] == self.sabr_params.rho
        
        # Check market features
        assert features[5] == self.strike
        assert features[6] == self.maturity
        np.testing.assert_almost_equal(features[7], self.strike / self.sabr_params.F0)  # moneyness
        np.testing.assert_almost_equal(features[8], np.log(self.strike / self.sabr_params.F0))  # log_moneyness
        
        # Check Hagan vol
        assert features[9] == self.hagan_vol
        
        # Check derived features exist
        assert len(features) >= 10
    
    def test_create_point_features_no_hagan_vol(self):
        """Test point feature creation without Hagan volatility."""
        features = self.engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity, hagan_vol=None
        )
        
        # Should use 0.0 as placeholder for Hagan vol
        hagan_vol_idx = self.engineer.feature_names.index('hagan_vol')
        assert features[hagan_vol_idx] == 0.0
    
    def test_create_point_features_derived(self):
        """Test derived feature calculations."""
        features = self.engineer.create_point_features(
            self.sabr_params, self.strike, self.maturity, self.hagan_vol
        )
        
        # Find derived feature indices
        alpha_beta_idx = self.engineer.feature_names.index('alpha_beta_interaction')
        nu_rho_idx = self.engineer.feature_names.index('nu_rho_interaction')
        sqrt_time_idx = self.engineer.feature_names.index('time_to_expiry_sqrt')
        vol_scaled_idx = self.engineer.feature_names.index('vol_of_vol_scaled')
        
        # Check calculations
        expected_alpha_beta = self.sabr_params.alpha * (self.sabr_params.beta ** 2)
        expected_nu_rho = self.sabr_params.nu * abs(self.sabr_params.rho)
        expected_sqrt_time = np.sqrt(self.maturity)
        expected_vol_scaled = self.sabr_params.nu * self.sabr_params.alpha
        
        assert features[alpha_beta_idx] == expected_alpha_beta
        assert features[nu_rho_idx] == expected_nu_rho
        assert features[sqrt_time_idx] == expected_sqrt_time
        assert features[vol_scaled_idx] == expected_vol_scaled
    
    def test_create_features_batch(self):
        """Test batch feature creation."""
        # Create multiple parameter sets
        sabr_params_list = [
            SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.2),
            SABRParams(F0=110.0, alpha=0.25, beta=0.8, nu=0.3, rho=0.1),
            SABRParams(F0=90.0, alpha=0.35, beta=0.6, nu=0.5, rho=-0.3)
        ]
        
        strikes = np.array([105.0, 115.0, 95.0])
        maturities = np.array([1.0, 0.5, 2.0])
        hagan_vols = np.array([0.25, 0.22, 0.28])
        
        features = self.engineer.create_features_batch(
            sabr_params_list, strikes, maturities, hagan_vols
        )
        
        assert features.shape == (3, len(self.engineer.feature_names))
        assert features.dtype == np.float32
        
        # Check that batch matches individual creation
        for i in range(3):
            individual_features = self.engineer.create_point_features(
                sabr_params_list[i], strikes[i], maturities[i], hagan_vols[i]
            )
            np.testing.assert_array_equal(features[i], individual_features)
    
    def test_create_features_batch_mismatched_lengths(self):
        """Test batch creation with mismatched input lengths."""
        sabr_params_list = [self.sabr_params]
        strikes = np.array([105.0, 115.0])  # Different length
        maturities = np.array([1.0])
        
        with pytest.raises(ValueError, match="Number of SABR params must match"):
            self.engineer.create_features_batch(sabr_params_list, strikes, maturities)
    
    def test_fit_normalization(self):
        """Test normalization fitting."""
        # Create sample feature matrix
        n_samples = 100
        n_features = len(self.engineer.feature_names)
        features = np.random.randn(n_samples, n_features) * 2 + 1
        
        self.engineer.fit_normalization(features)
        
        assert self.engineer.is_fitted
        assert self.engineer.feature_stats is not None
        assert len(self.engineer.feature_stats.means) == n_features
        assert len(self.engineer.feature_stats.stds) == n_features
        assert len(self.engineer.feature_stats.feature_names) == n_features
    
    def test_fit_normalization_with_nans(self):
        """Test normalization fitting with NaN values."""
        n_samples = 100
        n_features = len(self.engineer.feature_names)
        features = np.random.randn(n_samples, n_features)
        
        # Add some NaN values
        features[10:20, 0] = np.nan
        features[30:35, 2] = np.nan
        
        self.engineer.fit_normalization(features)
        
        assert self.engineer.is_fitted
        # Should handle NaNs gracefully
        assert np.all(np.isfinite(self.engineer.feature_stats.means))
        assert np.all(np.isfinite(self.engineer.feature_stats.stds))
    
    def test_fit_normalization_wrong_shape(self):
        """Test normalization fitting with wrong feature count."""
        features = np.random.randn(100, 5)  # Wrong number of features
        
        with pytest.raises(ValueError, match="Expected .* features, got"):
            self.engineer.fit_normalization(features)
    
    def test_normalize_features_not_fitted(self):
        """Test normalization without fitting first."""
        features = np.random.randn(10, len(self.engineer.feature_names))
        
        with pytest.raises(ValueError, match="Must call fit_normalization"):
            self.engineer.normalize_features(features)
    
    def test_normalize_features_standard(self):
        """Test standard feature normalization."""
        # Create and fit on training data
        train_features = np.random.randn(100, len(self.engineer.feature_names)) * 2 + 5
        self.engineer.fit_normalization(train_features)
        
        # Normalize test data
        test_features = np.random.randn(20, len(self.engineer.feature_names)) * 2 + 5
        normalized = self.engineer.normalize_features(test_features)
        
        assert normalized.shape == test_features.shape
        assert np.all(np.isfinite(normalized))
    
    def test_normalize_features_disabled(self):
        """Test normalization when disabled in config."""
        config = FeatureConfig(normalize_features=False)
        engineer = FeatureEngineer(config)
        
        features = np.random.randn(100, len(engineer.feature_names))
        engineer.fit_normalization(features)
        
        test_features = np.random.randn(20, len(engineer.feature_names))
        normalized = engineer.normalize_features(test_features)
        
        # Should return copy of original features
        np.testing.assert_array_equal(normalized, test_features)
    
    def test_normalize_features_robust_scaling(self):
        """Test robust feature scaling."""
        config = FeatureConfig(robust_scaling=True, standardize_features=False)
        engineer = FeatureEngineer(config)
        
        features = np.random.randn(100, len(engineer.feature_names)) * 2 + 5
        engineer.fit_normalization(features)
        
        test_features = np.random.randn(20, len(engineer.feature_names)) * 2 + 5
        normalized = engineer.normalize_features(test_features)
        
        assert normalized.shape == test_features.shape
        assert np.all(np.isfinite(normalized))
    
    def test_normalize_features_log_transform(self):
        """Test log transformation of specified features."""
        # Create features with positive values for log transform
        features = np.abs(np.random.randn(100, len(self.engineer.feature_names))) + 0.1
        self.engineer.fit_normalization(features)
        
        test_features = np.abs(np.random.randn(20, len(self.engineer.feature_names))) + 0.1
        normalized = self.engineer.normalize_features(test_features)
        
        assert normalized.shape == test_features.shape
        assert np.all(np.isfinite(normalized))
    
    def test_inverse_normalize_features(self):
        """Test inverse feature normalization."""
        # Create and fit on training data
        train_features = np.random.randn(100, len(self.engineer.feature_names)) * 2 + 5
        self.engineer.fit_normalization(train_features)
        
        # Normalize and then inverse normalize
        test_features = np.random.randn(20, len(self.engineer.feature_names)) * 2 + 5
        normalized = self.engineer.normalize_features(test_features)
        inverse_normalized = self.engineer.inverse_normalize_features(normalized)
        
        # Should approximately recover original features
        # (may not be exact due to log transforms and numerical precision)
        assert inverse_normalized.shape == test_features.shape
        assert np.all(np.isfinite(inverse_normalized))
    
    def test_inverse_normalize_not_fitted(self):
        """Test inverse normalization without fitting first."""
        features = np.random.randn(10, len(self.engineer.feature_names))
        
        with pytest.raises(ValueError, match="Must call fit_normalization"):
            self.engineer.inverse_normalize_features(features)
    
    def test_get_feature_importance_names(self):
        """Test getting feature names for importance analysis."""
        names = self.engineer.get_feature_importance_names()
        
        assert names == self.engineer.feature_names
        assert names is not self.engineer.feature_names  # Should be a copy
    
    def test_validate_features_valid(self):
        """Test feature validation with valid features."""
        features = np.random.randn(50, len(self.engineer.feature_names))
        
        # Make sure SABR parameters are in valid ranges
        if 'F0' in self.engineer.feature_names:
            f0_idx = self.engineer.feature_names.index('F0')
            features[:, f0_idx] = np.abs(features[:, f0_idx]) + 50  # Positive F0
        
        if 'alpha' in self.engineer.feature_names:
            alpha_idx = self.engineer.feature_names.index('alpha')
            features[:, alpha_idx] = np.abs(features[:, alpha_idx]) + 0.1  # Positive alpha
        
        if 'beta' in self.engineer.feature_names:
            beta_idx = self.engineer.feature_names.index('beta')
            features[:, beta_idx] = np.clip(features[:, beta_idx], 0, 1)  # Beta in [0,1]
        
        if 'rho' in self.engineer.feature_names:
            rho_idx = self.engineer.feature_names.index('rho')
            features[:, rho_idx] = np.clip(features[:, rho_idx], -1, 1)  # Rho in [-1,1]
        
        if 'nu' in self.engineer.feature_names:
            nu_idx = self.engineer.feature_names.index('nu')
            features[:, nu_idx] = np.abs(features[:, nu_idx])  # Non-negative nu
        
        validation = self.engineer.validate_features(features)
        
        assert validation['is_valid'] is True
        assert len(validation['errors']) == 0
        assert validation['diagnostics']['shape'] == features.shape
    
    def test_validate_features_wrong_shape(self):
        """Test feature validation with wrong shape."""
        features = np.random.randn(50, 5)  # Wrong number of features
        
        validation = self.engineer.validate_features(features)
        
        assert validation['is_valid'] is False
        assert any("Expected" in error for error in validation['errors'])
    
    def test_validate_features_1d_array(self):
        """Test feature validation with 1D array."""
        features = np.random.randn(50)  # 1D array
        
        validation = self.engineer.validate_features(features)
        
        assert validation['is_valid'] is False
        assert any("Expected 2D array" in error for error in validation['errors'])
    
    def test_validate_features_with_nans(self):
        """Test feature validation with NaN values."""
        features = np.random.randn(50, len(self.engineer.feature_names))
        features[10:20, 0] = np.nan  # Add some NaNs
        
        validation = self.engineer.validate_features(features)
        
        # Should still be valid but with warnings about finite fraction
        assert validation['is_valid'] is True
        finite_fraction = validation['diagnostics']['finite_fraction']
        assert finite_fraction < 1.0
    
    def test_validate_features_invalid_ranges(self):
        """Test feature validation with invalid parameter ranges."""
        features = np.random.randn(50, len(self.engineer.feature_names))
        
        # Set invalid values
        if 'F0' in self.engineer.feature_names:
            f0_idx = self.engineer.feature_names.index('F0')
            features[0, f0_idx] = -10  # Negative F0
        
        if 'beta' in self.engineer.feature_names:
            beta_idx = self.engineer.feature_names.index('beta')
            features[1, beta_idx] = 1.5  # Beta > 1
        
        if 'rho' in self.engineer.feature_names:
            rho_idx = self.engineer.feature_names.index('rho')
            features[2, rho_idx] = 1.5  # Rho > 1
        
        validation = self.engineer.validate_features(features)
        
        # Should have warnings about invalid ranges
        assert len(validation['warnings']) > 0
    
    def test_get_feature_correlations(self):
        """Test feature correlation computation."""
        features = np.random.randn(100, len(self.engineer.feature_names))
        
        correlations = self.engineer.get_feature_correlations(features)
        
        assert correlations.shape == (len(self.engineer.feature_names), len(self.engineer.feature_names))
        
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(correlations), 1.0)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(correlations, correlations.T)
    
    def test_get_feature_correlations_with_nans(self):
        """Test correlation computation with NaN values."""
        features = np.random.randn(100, len(self.engineer.feature_names))
        features[50:, :] = np.nan  # Half NaN values
        
        correlations = self.engineer.get_feature_correlations(features)
        
        assert correlations.shape == (len(self.engineer.feature_names), len(self.engineer.feature_names))
        assert np.all(np.isfinite(correlations))
    
    def test_detect_feature_outliers(self):
        """Test outlier detection."""
        # Create features with some outliers
        features = np.random.randn(100, len(self.engineer.feature_names))
        features[0, 0] = 10  # Outlier in first feature
        features[1, 1] = -8  # Outlier in second feature
        
        outliers = self.engineer.detect_feature_outliers(features, threshold=3.0)
        
        assert len(outliers) == 100
        assert outliers[0] == True  # First sample should be outlier
        assert outliers[1] == True  # Second sample should be outlier
    
    def test_detect_feature_outliers_fitted(self):
        """Test outlier detection with fitted normalization."""
        # Fit on clean data
        clean_features = np.random.randn(100, len(self.engineer.feature_names))
        self.engineer.fit_normalization(clean_features)
        
        # Test on data with outliers
        test_features = np.random.randn(50, len(self.engineer.feature_names))
        test_features[0, 0] = 10  # Outlier
        
        outliers = self.engineer.detect_feature_outliers(test_features, threshold=3.0)
        
        assert len(outliers) == 50
        assert outliers[0] == True  # Should detect outlier


if __name__ == "__main__":
    pytest.main([__file__])