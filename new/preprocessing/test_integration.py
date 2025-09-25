"""
Integration tests for patch extraction and feature engineering.

This module tests the integration between PatchExtractor and FeatureEngineer
to ensure they work together correctly for the MDA-CNN pipeline.
"""

import numpy as np
import pytest

from .patch_extractor import PatchExtractor, PatchConfig
from .feature_engineer import FeatureEngineer, FeatureConfig
from ..data_generation.sabr_params import SABRParams, GridConfig


class TestIntegration:
    """Test integration between patch extraction and feature engineering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create configurations
        self.patch_config = PatchConfig(patch_size=(7, 7), normalize_patches=True)
        self.feature_config = FeatureConfig()
        
        # Create extractors
        self.patch_extractor = PatchExtractor(self.patch_config)
        self.feature_engineer = FeatureEngineer(self.feature_config)
        
        # Create test data
        self.grid_config = GridConfig(
            strike_range=(0.8, 1.2),
            maturity_range=(0.1, 2.0),
            n_strikes=15,
            n_maturities=10
        )
        
        # Create test SABR parameters
        self.sabr_params = SABRParams(
            F0=100.0,
            alpha=0.3,
            beta=0.7,
            nu=0.4,
            rho=-0.2
        )
        
        # Create synthetic surface data
        strikes = self.grid_config.get_strikes(self.sabr_params.F0)
        maturities = self.grid_config.get_maturities()
        
        # Create synthetic LF surface (simple volatility smile)
        self.lf_surface = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                moneyness = K / self.sabr_params.F0
                # Simple volatility smile: higher vol for OTM options
                vol = 0.2 + 0.1 * abs(np.log(moneyness)) + 0.05 * np.sqrt(T)
                self.lf_surface[i, j] = vol
        
        # Create HF points (subset of grid)
        self.hf_strikes = strikes[::3]  # Every 3rd strike
        self.hf_maturities = maturities[::2]  # Every 2nd maturity
        
        # Create HF volatilities (LF + some noise/correction)
        self.hf_vols = []
        self.hf_coordinates = []
        
        for T in self.hf_maturities:
            for K in self.hf_strikes:
                # Find closest grid point
                t_idx = np.argmin(np.abs(maturities - T))
                k_idx = np.argmin(np.abs(strikes - K))
                
                # HF vol = LF vol + small correction
                lf_vol = self.lf_surface[t_idx, k_idx]
                hf_vol = lf_vol + np.random.normal(0, 0.01)  # Small correction
                
                self.hf_vols.append(hf_vol)
                self.hf_coordinates.append((t_idx, k_idx))
        
        self.hf_vols = np.array(self.hf_vols)
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from surface to model inputs."""
        # Step 1: Align HF points to grid
        grid_strikes = self.grid_config.get_strikes(self.sabr_params.F0)
        grid_maturities = self.grid_config.get_maturities()
        
        # Create all combinations of HF strikes and maturities
        hf_strikes_flat = []
        hf_maturities_flat = []
        for T in self.hf_maturities:
            for K in self.hf_strikes:
                hf_strikes_flat.append(K)
                hf_maturities_flat.append(T)
        
        coordinates = self.patch_extractor.align_hf_to_grid(
            np.array(hf_strikes_flat), np.array(hf_maturities_flat),
            grid_strikes, grid_maturities
        )
        
        assert len(coordinates) == len(self.hf_strikes) * len(self.hf_maturities)
        
        # Step 2: Extract patches around HF points
        patches = self.patch_extractor.extract_patches_batch(
            self.lf_surface, coordinates
        )
        
        assert patches.shape == (len(coordinates), 7, 7)
        assert np.all(np.isfinite(patches))
        
        # Step 3: Create point features
        sabr_params_list = [self.sabr_params] * len(coordinates)
        
        # Flatten HF strikes and maturities for all combinations
        strikes_flat = []
        maturities_flat = []
        hagan_vols_flat = []
        
        for T in self.hf_maturities:
            for K in self.hf_strikes:
                strikes_flat.append(K)
                maturities_flat.append(T)
                
                # Get Hagan vol from LF surface
                t_idx = np.argmin(np.abs(grid_maturities - T))
                k_idx = np.argmin(np.abs(grid_strikes - K))
                hagan_vols_flat.append(self.lf_surface[t_idx, k_idx])
        
        features = self.feature_engineer.create_features_batch(
            sabr_params_list,
            np.array(strikes_flat),
            np.array(maturities_flat),
            np.array(hagan_vols_flat)
        )
        
        assert features.shape == (len(coordinates), len(self.feature_engineer.feature_names))
        assert np.all(np.isfinite(features))
        
        # Step 4: Normalize features
        self.feature_engineer.fit_normalization(features)
        normalized_features = self.feature_engineer.normalize_features(features)
        
        assert normalized_features.shape == features.shape
        assert np.all(np.isfinite(normalized_features))
        
        # Step 5: Validate pipeline outputs
        validation_patches = self.patch_extractor.validate_extraction(
            self.lf_surface, coordinates
        )
        assert validation_patches['is_valid']
        
        validation_features = self.feature_engineer.validate_features(features)
        assert validation_features['is_valid']
        
        print(f"Pipeline completed successfully:")
        print(f"  - Extracted {patches.shape[0]} patches of size {patches.shape[1:]}") 
        print(f"  - Created {features.shape[0]} feature vectors with {features.shape[1]} features")
        print(f"  - All data validated successfully")
    
    def test_different_patch_sizes(self):
        """Test pipeline with different patch sizes."""
        patch_sizes = [(3, 3), (5, 5), (9, 9), (11, 11)]
        
        for patch_size in patch_sizes:
            config = PatchConfig(patch_size=patch_size, normalize_patches=True)
            extractor = PatchExtractor(config)
            
            # Use subset of coordinates to avoid boundary issues with large patches
            coordinates = self.hf_coordinates[:5]  # Use first 5 points
            
            patches = extractor.extract_patches_batch(self.lf_surface, coordinates)
            
            assert patches.shape == (len(coordinates), patch_size[0], patch_size[1])
            assert np.all(np.isfinite(patches))
    
    def test_feature_configurations(self):
        """Test pipeline with different feature configurations."""
        configs = [
            FeatureConfig(include_derived_features=False),
            FeatureConfig(include_hagan_vol=False),
            FeatureConfig(normalize_features=False),
            FeatureConfig(robust_scaling=True, standardize_features=False)
        ]
        
        for config in configs:
            engineer = FeatureEngineer(config)
            
            features = engineer.create_point_features(
                self.sabr_params,
                self.hf_strikes[0],
                self.hf_maturities[0],
                self.hf_vols[0]
            )
            
            assert len(features) == len(engineer.feature_names)
            assert np.all(np.isfinite(features))
    
    def test_batch_processing_consistency(self):
        """Test that batch processing gives same results as individual processing."""
        # Process individually
        individual_patches = []
        individual_features = []
        
        for i, coord in enumerate(self.hf_coordinates[:5]):
            # Extract patch
            patch = self.patch_extractor.extract_patch(self.lf_surface, coord)
            individual_patches.append(patch)
            
            # Create features
            strike_idx = coord[1]
            maturity_idx = coord[0]
            
            grid_strikes = self.grid_config.get_strikes(self.sabr_params.F0)
            grid_maturities = self.grid_config.get_maturities()
            
            strike = grid_strikes[strike_idx]
            maturity = grid_maturities[maturity_idx]
            hagan_vol = self.lf_surface[coord]
            
            features = self.feature_engineer.create_point_features(
                self.sabr_params, strike, maturity, hagan_vol
            )
            individual_features.append(features)
        
        individual_patches = np.array(individual_patches)
        individual_features = np.array(individual_features)
        
        # Process in batch
        batch_patches = self.patch_extractor.extract_patches_batch(
            self.lf_surface, self.hf_coordinates[:5]
        )
        
        # Create batch features
        sabr_params_list = [self.sabr_params] * 5
        strikes = []
        maturities = []
        hagan_vols = []
        
        grid_strikes = self.grid_config.get_strikes(self.sabr_params.F0)
        grid_maturities = self.grid_config.get_maturities()
        
        for coord in self.hf_coordinates[:5]:
            strikes.append(grid_strikes[coord[1]])
            maturities.append(grid_maturities[coord[0]])
            hagan_vols.append(self.lf_surface[coord])
        
        batch_features = self.feature_engineer.create_features_batch(
            sabr_params_list, np.array(strikes), np.array(maturities), np.array(hagan_vols)
        )
        
        # Compare results
        np.testing.assert_array_almost_equal(individual_patches, batch_patches)
        np.testing.assert_array_almost_equal(individual_features, batch_features)
    
    def test_error_handling(self):
        """Test error handling in integrated pipeline."""
        # Test with mismatched dimensions
        with pytest.raises(ValueError):
            self.feature_engineer.create_features_batch(
                [self.sabr_params],  # 1 param set
                np.array([100, 105]),  # 2 strikes
                np.array([1.0])  # 1 maturity
            )
        
        # Test with invalid surface
        empty_surface = np.array([]).reshape(0, 0)
        validation = self.patch_extractor.validate_extraction(empty_surface, [(0, 0)])
        assert not validation['is_valid']
        
        # Test with invalid features
        invalid_features = np.random.randn(10, 5)  # Wrong number of features
        validation = self.feature_engineer.validate_features(invalid_features)
        assert not validation['is_valid']


if __name__ == "__main__":
    pytest.main([__file__])