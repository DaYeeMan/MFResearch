"""
Tests for patch extraction functionality.
"""

import numpy as np
import pytest
from unittest.mock import Mock

from .patch_extractor import PatchExtractor, PatchConfig


class TestPatchConfig:
    """Test PatchConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PatchConfig()
        
        assert config.patch_size == (9, 9)
        assert config.boundary_mode == 'reflect'
        assert config.pad_value == 0.0
        assert config.normalize_patches is True
        assert config.center_on_hf is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PatchConfig(
            patch_size=(5, 7),
            boundary_mode='constant',
            pad_value=1.0,
            normalize_patches=False
        )
        
        assert config.patch_size == (5, 7)
        assert config.boundary_mode == 'constant'
        assert config.pad_value == 1.0
        assert config.normalize_patches is False


class TestPatchExtractor:
    """Test PatchExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PatchConfig(patch_size=(5, 5), boundary_mode='reflect')
        self.extractor = PatchExtractor(self.config)
        
        # Create test surface
        self.surface = np.random.rand(10, 12) * 0.5 + 0.2  # Volatilities between 0.2 and 0.7
        
        # Create test grid coordinates
        self.grid_strikes = np.linspace(80, 120, 12)
        self.grid_maturities = np.linspace(0.1, 2.0, 10)
    
    def test_initialization(self):
        """Test extractor initialization."""
        assert self.extractor.patch_height == 5
        assert self.extractor.patch_width == 5
        assert self.extractor.config.boundary_mode == 'reflect'
    
    def test_initialization_invalid_patch_size(self):
        """Test initialization with invalid patch size."""
        with pytest.raises(ValueError, match="Patch size must be positive"):
            PatchExtractor(PatchConfig(patch_size=(0, 5)))
        
        with pytest.raises(ValueError, match="Patch size must be positive"):
            PatchExtractor(PatchConfig(patch_size=(5, -1)))
    
    def test_initialization_invalid_boundary_mode(self):
        """Test initialization with invalid boundary mode."""
        with pytest.raises(ValueError, match="Invalid boundary mode"):
            PatchExtractor(PatchConfig(boundary_mode='invalid'))
    
    def test_align_hf_to_grid_basic(self):
        """Test basic HF to grid alignment."""
        hf_strikes = np.array([85, 100, 115])
        hf_maturities = np.array([0.5, 1.0, 1.5])
        
        coordinates = self.extractor.align_hf_to_grid(
            hf_strikes, hf_maturities,
            self.grid_strikes, self.grid_maturities
        )
        
        assert len(coordinates) == 3
        
        # Check that coordinates are within bounds
        for t_idx, k_idx in coordinates:
            assert 0 <= t_idx < len(self.grid_maturities)
            assert 0 <= k_idx < len(self.grid_strikes)
    
    def test_align_hf_to_grid_mismatched_lengths(self):
        """Test alignment with mismatched input lengths."""
        hf_strikes = np.array([85, 100])
        hf_maturities = np.array([0.5, 1.0, 1.5])
        
        with pytest.raises(ValueError, match="HF strikes and maturities must have same length"):
            self.extractor.align_hf_to_grid(
                hf_strikes, hf_maturities,
                self.grid_strikes, self.grid_maturities
            )
    
    def test_align_hf_to_grid_outside_bounds(self):
        """Test alignment with points outside grid bounds."""
        # Use points that are definitely outside the grid bounds
        hf_strikes = np.array([10, 100, 200])  # 10 and 200 well outside [80, 120]
        hf_maturities = np.array([0.01, 1.0, 5.0])  # 0.01 and 5.0 well outside [0.1, 2.0]
        
        with pytest.warns(UserWarning):
            coordinates = self.extractor.align_hf_to_grid(
                hf_strikes, hf_maturities,
                self.grid_strikes, self.grid_maturities
            )
        
        # Should still return valid coordinates (clipped)
        assert len(coordinates) == 3
        for t_idx, k_idx in coordinates:
            assert 0 <= t_idx < len(self.grid_maturities)
            assert 0 <= k_idx < len(self.grid_strikes)
    
    def test_extract_patch_center(self):
        """Test patch extraction from center of surface."""
        center_idx = (5, 6)  # Center of 10x12 surface
        
        patch = self.extractor.extract_patch(self.surface, center_idx)
        
        assert patch.shape == (5, 5)
        assert np.all(np.isfinite(patch))
        
        # For normalized patches, we can't directly compare values
        # Instead, test with non-normalized extractor
        config_no_norm = PatchConfig(patch_size=(5, 5), normalize_patches=False)
        extractor_no_norm = PatchExtractor(config_no_norm)
        patch_no_norm = extractor_no_norm.extract_patch(self.surface, center_idx)
        
        # Check that center value matches for non-normalized patch
        expected_center = self.surface[center_idx]
        actual_center = patch_no_norm[2, 2]  # Center of 5x5 patch
        assert actual_center == expected_center
    
    def test_extract_patch_corner(self):
        """Test patch extraction from corner (boundary handling)."""
        center_idx = (0, 0)  # Top-left corner
        
        patch = self.extractor.extract_patch(self.surface, center_idx)
        
        assert patch.shape == (5, 5)
        assert np.all(np.isfinite(patch))
    
    def test_extract_patch_different_sizes(self):
        """Test patch extraction with different patch sizes."""
        center_idx = (5, 6)
        
        # Test 3x3 patch
        patch_3x3 = self.extractor.extract_patch(self.surface, center_idx, patch_size=(3, 3))
        assert patch_3x3.shape == (3, 3)
        
        # Test 7x9 patch
        patch_7x9 = self.extractor.extract_patch(self.surface, center_idx, patch_size=(7, 9))
        assert patch_7x9.shape == (7, 9)
    
    def test_extract_patch_boundary_modes(self):
        """Test different boundary handling modes."""
        center_idx = (0, 0)  # Corner position
        
        # Test reflect mode
        config_reflect = PatchConfig(patch_size=(5, 5), boundary_mode='reflect')
        extractor_reflect = PatchExtractor(config_reflect)
        patch_reflect = extractor_reflect.extract_patch(self.surface, center_idx)
        assert patch_reflect.shape == (5, 5)
        
        # Test constant mode
        config_constant = PatchConfig(patch_size=(5, 5), boundary_mode='constant', pad_value=0.5)
        extractor_constant = PatchExtractor(config_constant)
        patch_constant = extractor_constant.extract_patch(self.surface, center_idx)
        assert patch_constant.shape == (5, 5)
        
        # Test wrap mode
        config_wrap = PatchConfig(patch_size=(5, 5), boundary_mode='wrap')
        extractor_wrap = PatchExtractor(config_wrap)
        patch_wrap = extractor_wrap.extract_patch(self.surface, center_idx)
        assert patch_wrap.shape == (5, 5)
    
    def test_extract_patch_normalization(self):
        """Test patch normalization."""
        center_idx = (5, 6)
        
        # Test with normalization
        config_norm = PatchConfig(patch_size=(5, 5), normalize_patches=True)
        extractor_norm = PatchExtractor(config_norm)
        patch_norm = extractor_norm.extract_patch(self.surface, center_idx)
        
        # Should have approximately zero mean and unit std (for non-constant patches)
        if np.std(patch_norm) > 0:
            assert abs(np.mean(patch_norm)) < 1e-10
            assert abs(np.std(patch_norm) - 1.0) < 1e-10
        
        # Test without normalization
        config_no_norm = PatchConfig(patch_size=(5, 5), normalize_patches=False)
        extractor_no_norm = PatchExtractor(config_no_norm)
        patch_no_norm = extractor_no_norm.extract_patch(self.surface, center_idx)
        
        # Should preserve original values
        assert patch_no_norm[2, 2] == self.surface[center_idx]
    
    def test_extract_patches_batch(self):
        """Test batch patch extraction."""
        centers = [(2, 3), (5, 6), (7, 8)]
        
        patches = self.extractor.extract_patches_batch(self.surface, centers)
        
        assert patches.shape == (3, 5, 5)
        
        # Check individual patches match single extraction
        for i, center in enumerate(centers):
            single_patch = self.extractor.extract_patch(self.surface, center)
            np.testing.assert_array_equal(patches[i], single_patch)
    
    def test_extract_patches_batch_empty(self):
        """Test batch extraction with empty centers list."""
        patches = self.extractor.extract_patches_batch(self.surface, [])
        
        assert patches.shape == (0, 5, 5)
    
    def test_validate_extraction_valid(self):
        """Test validation with valid setup."""
        centers = [(2, 3), (5, 6), (7, 8)]
        
        validation = self.extractor.validate_extraction(self.surface, centers)
        
        assert validation['is_valid'] is True
        assert len(validation['errors']) == 0
        assert validation['diagnostics']['surface_shape'] == self.surface.shape
        assert validation['diagnostics']['n_centers'] == 3
    
    def test_validate_extraction_empty_surface(self):
        """Test validation with empty surface."""
        empty_surface = np.array([]).reshape(0, 0)
        centers = [(0, 0)]
        
        validation = self.extractor.validate_extraction(empty_surface, centers)
        
        assert validation['is_valid'] is False
        assert any("Empty surface" in error for error in validation['errors'])
    
    def test_validate_extraction_invalid_centers(self):
        """Test validation with invalid center coordinates."""
        centers = [(-1, 0), (5, 6), (15, 20)]  # Invalid coordinates
        
        validation = self.extractor.validate_extraction(self.surface, centers)
        
        assert validation['is_valid'] is False
        assert validation['diagnostics']['invalid_centers'] > 0
    
    def test_validate_extraction_boundary_centers(self):
        """Test validation with centers near boundaries."""
        centers = [(0, 0), (9, 11)]  # Corner positions
        
        validation = self.extractor.validate_extraction(self.surface, centers)
        
        # Should be valid but with warnings
        assert validation['is_valid'] is True
        assert validation['diagnostics']['boundary_centers'] > 0
        assert len(validation['warnings']) > 0
    
    def test_patch_extraction_with_nans(self):
        """Test patch extraction with NaN values in surface."""
        # Create surface with some NaN values
        surface_with_nans = self.surface.copy()
        surface_with_nans[2:4, 3:6] = np.nan
        
        center_idx = (3, 4)  # Center in NaN region
        
        patch = self.extractor.extract_patch(surface_with_nans, center_idx)
        
        assert patch.shape == (5, 5)
        # After normalization, NaNs should be replaced with zeros
        assert np.all(np.isfinite(patch))
    
    def test_patch_extraction_constant_surface(self):
        """Test patch extraction from constant surface."""
        constant_surface = np.full((10, 12), 0.3)
        center_idx = (5, 6)
        
        patch = self.extractor.extract_patch(constant_surface, center_idx)
        
        assert patch.shape == (5, 5)
        # Constant surface should normalize to zeros
        if self.config.normalize_patches:
            np.testing.assert_array_almost_equal(patch, np.zeros((5, 5)))
        else:
            np.testing.assert_array_almost_equal(patch, np.full((5, 5), 0.3))
    
    def test_resize_patch_crop(self):
        """Test patch resizing by cropping."""
        large_patch = np.random.rand(7, 9)
        target_size = (5, 5)
        
        resized = self.extractor._resize_patch(large_patch, target_size)
        
        assert resized.shape == target_size
    
    def test_resize_patch_pad(self):
        """Test patch resizing by padding."""
        small_patch = np.random.rand(3, 3)
        target_size = (5, 5)
        
        resized = self.extractor._resize_patch(small_patch, target_size)
        
        assert resized.shape == target_size
    
    def test_normalize_patch_edge_cases(self):
        """Test patch normalization edge cases."""
        # All NaN patch
        nan_patch = np.full((3, 3), np.nan)
        normalized = self.extractor._normalize_patch(nan_patch)
        np.testing.assert_array_equal(normalized, np.zeros((3, 3)))
        
        # Constant patch
        constant_patch = np.full((3, 3), 0.5)
        normalized = self.extractor._normalize_patch(constant_patch)
        np.testing.assert_array_equal(normalized, np.zeros((3, 3)))
        
        # Mixed finite/non-finite patch
        mixed_patch = np.array([[1.0, 2.0, np.nan],
                               [3.0, np.inf, 4.0],
                               [5.0, 6.0, 7.0]])
        normalized = self.extractor._normalize_patch(mixed_patch)
        assert normalized.shape == (3, 3)
        assert np.all(np.isfinite(normalized))


if __name__ == "__main__":
    pytest.main([__file__])