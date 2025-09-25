"""
Patch extraction utilities for MDA-CNN volatility surface modeling.

This module provides functionality to extract local surface patches around
high-fidelity points and align HF points to LF surface grids for training
the MDA-CNN model.
"""

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import warnings

from ..data_generation.sabr_params import GridConfig


@dataclass
class PatchConfig:
    """
    Configuration for patch extraction.
    
    Attributes:
        patch_size: (height, width) of extracted patches
        boundary_mode: How to handle boundary conditions ('pad', 'reflect', 'wrap', 'constant')
        pad_value: Value to use for constant padding
        normalize_patches: Whether to normalize patches locally
        center_on_hf: Whether to center patches exactly on HF points
    """
    patch_size: Tuple[int, int] = (9, 9)
    boundary_mode: str = 'reflect'
    pad_value: float = 0.0
    normalize_patches: bool = True
    center_on_hf: bool = True


class PatchExtractor:
    """
    Extract local surface patches around high-fidelity points for CNN input.
    
    This class handles the extraction of local patches from low-fidelity surfaces
    around high-fidelity sample points, with proper boundary handling and
    grid alignment.
    """
    
    def __init__(self, config: PatchConfig):
        """
        Initialize patch extractor.
        
        Args:
            config: Patch extraction configuration
        """
        self.config = config
        self.patch_height, self.patch_width = config.patch_size
        
        # Validate configuration
        if self.patch_height <= 0 or self.patch_width <= 0:
            raise ValueError("Patch size must be positive")
        
        if config.boundary_mode not in ['pad', 'reflect', 'wrap', 'constant']:
            raise ValueError(f"Invalid boundary mode: {config.boundary_mode}")
    
    def align_hf_to_grid(self, 
                        hf_strikes: np.ndarray, 
                        hf_maturities: np.ndarray,
                        grid_strikes: np.ndarray, 
                        grid_maturities: np.ndarray) -> List[Tuple[int, int]]:
        """
        Map high-fidelity points to low-fidelity surface grid coordinates.
        
        Args:
            hf_strikes: High-fidelity strike prices
            hf_maturities: High-fidelity maturity times
            grid_strikes: LF surface strike grid
            grid_maturities: LF surface maturity grid
            
        Returns:
            List of (maturity_idx, strike_idx) grid coordinates for each HF point
        """
        if len(hf_strikes) != len(hf_maturities):
            raise ValueError("HF strikes and maturities must have same length")
        
        grid_coordinates = []
        
        for strike, maturity in zip(hf_strikes, hf_maturities):
            # Check if values are outside grid ranges before finding closest points
            if maturity < grid_maturities[0] or maturity > grid_maturities[-1]:
                warnings.warn(f"Maturity {maturity} outside grid range [{grid_maturities[0]}, {grid_maturities[-1]}]", UserWarning)
            
            if strike < grid_strikes[0] or strike > grid_strikes[-1]:
                warnings.warn(f"Strike {strike} outside grid range [{grid_strikes[0]}, {grid_strikes[-1]}]", UserWarning)
            
            # Find closest grid points
            strike_idx = np.argmin(np.abs(grid_strikes - strike))
            maturity_idx = np.argmin(np.abs(grid_maturities - maturity))
            
            grid_coordinates.append((maturity_idx, strike_idx))
        
        return grid_coordinates
    
    def extract_patch(self, 
                     surface: np.ndarray, 
                     center_idx: Tuple[int, int],
                     patch_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract a local patch from a surface around a center point.
        
        Args:
            surface: 2D surface array (maturities x strikes)
            center_idx: (maturity_idx, strike_idx) center coordinates
            patch_size: Optional override for patch size
            
        Returns:
            Extracted patch array
        """
        if patch_size is None:
            patch_height, patch_width = self.patch_height, self.patch_width
        else:
            patch_height, patch_width = patch_size
        
        center_t, center_k = center_idx
        surface_height, surface_width = surface.shape
        
        # Calculate patch boundaries
        half_height = patch_height // 2
        half_width = patch_width // 2
        
        # Define extraction region
        t_start = center_t - half_height
        t_end = center_t + half_height + 1
        k_start = center_k - half_width
        k_end = center_k + half_width + 1
        
        # Handle boundary conditions
        if (t_start >= 0 and t_end <= surface_height and 
            k_start >= 0 and k_end <= surface_width):
            # Simple case: patch fits entirely within surface
            patch = surface[t_start:t_end, k_start:k_end]
        else:
            # Need boundary handling
            patch = self._extract_patch_with_boundaries(
                surface, center_t, center_k, patch_height, patch_width
            )
        
        # Ensure patch has correct size
        if patch.shape != (patch_height, patch_width):
            patch = self._resize_patch(patch, (patch_height, patch_width))
        
        # Optional local normalization
        if self.config.normalize_patches:
            patch = self._normalize_patch(patch)
        
        return patch
    
    def _extract_patch_with_boundaries(self, 
                                     surface: np.ndarray,
                                     center_t: int, 
                                     center_k: int,
                                     patch_height: int, 
                                     patch_width: int) -> np.ndarray:
        """
        Extract patch with boundary condition handling.
        
        Args:
            surface: Surface array
            center_t: Center maturity index
            center_k: Center strike index
            patch_height: Patch height
            patch_width: Patch width
            
        Returns:
            Extracted patch with boundary handling
        """
        surface_height, surface_width = surface.shape
        half_height = patch_height // 2
        half_width = patch_width // 2
        
        if self.config.boundary_mode == 'pad':
            # Pad surface and extract
            pad_t = max(half_height, 0)
            pad_k = max(half_width, 0)
            
            if self.config.boundary_mode == 'constant':
                padded_surface = np.pad(
                    surface, 
                    ((pad_t, pad_t), (pad_k, pad_k)), 
                    mode='constant', 
                    constant_values=self.config.pad_value
                )
            else:
                padded_surface = np.pad(
                    surface, 
                    ((pad_t, pad_t), (pad_k, pad_k)), 
                    mode='reflect'
                )
            
            # Adjust center coordinates for padded surface
            padded_center_t = center_t + pad_t
            padded_center_k = center_k + pad_k
            
            # Extract patch
            t_start = padded_center_t - half_height
            t_end = padded_center_t + half_height + 1
            k_start = padded_center_k - half_width
            k_end = padded_center_k + half_width + 1
            
            patch = padded_surface[t_start:t_end, k_start:k_end]
            
        elif self.config.boundary_mode == 'reflect':
            # Use numpy's reflection padding
            pad_t = half_height
            pad_k = half_width
            
            padded_surface = np.pad(
                surface, 
                ((pad_t, pad_t), (pad_k, pad_k)), 
                mode='reflect'
            )
            
            padded_center_t = center_t + pad_t
            padded_center_k = center_k + pad_k
            
            t_start = padded_center_t - half_height
            t_end = padded_center_t + half_height + 1
            k_start = padded_center_k - half_width
            k_end = padded_center_k + half_width + 1
            
            patch = padded_surface[t_start:t_end, k_start:k_end]
            
        elif self.config.boundary_mode == 'wrap':
            # Periodic boundary conditions
            patch = np.zeros((patch_height, patch_width))
            
            for i in range(patch_height):
                for j in range(patch_width):
                    t_idx = (center_t - half_height + i) % surface_height
                    k_idx = (center_k - half_width + j) % surface_width
                    patch[i, j] = surface[t_idx, k_idx]
                    
        else:  # constant
            # Fill with constant value outside boundaries
            patch = np.full((patch_height, patch_width), self.config.pad_value)
            
            # Fill valid region
            for i in range(patch_height):
                for j in range(patch_width):
                    t_idx = center_t - half_height + i
                    k_idx = center_k - half_width + j
                    
                    if 0 <= t_idx < surface_height and 0 <= k_idx < surface_width:
                        patch[i, j] = surface[t_idx, k_idx]
        
        return patch
    
    def _resize_patch(self, patch: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize patch to target size if needed.
        
        Args:
            patch: Input patch
            target_size: Target (height, width)
            
        Returns:
            Resized patch
        """
        current_height, current_width = patch.shape
        target_height, target_width = target_size
        
        if (current_height, current_width) == (target_height, target_width):
            return patch
        
        # Simple resize by cropping or padding
        if current_height >= target_height and current_width >= target_width:
            # Crop to center
            h_start = (current_height - target_height) // 2
            w_start = (current_width - target_width) // 2
            return patch[h_start:h_start + target_height, w_start:w_start + target_width]
        else:
            # Pad to target size
            h_pad = max(0, target_height - current_height)
            w_pad = max(0, target_width - current_width)
            
            h_pad_before = h_pad // 2
            h_pad_after = h_pad - h_pad_before
            w_pad_before = w_pad // 2
            w_pad_after = w_pad - w_pad_before
            
            return np.pad(patch, 
                         ((h_pad_before, h_pad_after), (w_pad_before, w_pad_after)),
                         mode='constant', constant_values=self.config.pad_value)
    
    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Normalize patch locally.
        
        Args:
            patch: Input patch
            
        Returns:
            Normalized patch
        """
        # Handle NaN values
        finite_mask = np.isfinite(patch)
        
        if not np.any(finite_mask):
            # All NaN/inf values
            return np.zeros_like(patch)
        
        finite_values = patch[finite_mask]
        
        if len(finite_values) == 0:
            return np.zeros_like(patch)
        
        # Compute statistics on finite values
        mean_val = np.mean(finite_values)
        std_val = np.std(finite_values)
        
        if std_val == 0:
            # Constant patch
            normalized_patch = np.zeros_like(patch)
        else:
            # Standard normalization
            normalized_patch = (patch - mean_val) / std_val
        
        # Handle non-finite values
        normalized_patch[~finite_mask] = 0.0
        
        return normalized_patch
    
    def extract_patches_batch(self, 
                             surface: np.ndarray,
                             center_coordinates: List[Tuple[int, int]],
                             patch_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract multiple patches from a surface.
        
        Args:
            surface: 2D surface array
            center_coordinates: List of (maturity_idx, strike_idx) centers
            patch_size: Optional override for patch size
            
        Returns:
            Array of shape (n_patches, patch_height, patch_width)
        """
        if not center_coordinates:
            return np.array([]).reshape(0, self.patch_height, self.patch_width)
        
        patches = []
        for center_idx in center_coordinates:
            patch = self.extract_patch(surface, center_idx, patch_size)
            patches.append(patch)
        
        return np.array(patches)
    
    def validate_extraction(self, 
                          surface: np.ndarray,
                          center_coordinates: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Validate patch extraction setup and return diagnostics.
        
        Args:
            surface: Surface to extract from
            center_coordinates: Center coordinates for patches
            
        Returns:
            Dictionary with validation results and diagnostics
        """
        surface_height, surface_width = surface.shape
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'diagnostics': {
                'surface_shape': surface.shape,
                'patch_size': self.config.patch_size,
                'n_centers': len(center_coordinates),
                'boundary_mode': self.config.boundary_mode
            }
        }
        
        # Check surface properties
        if surface.size == 0:
            validation_results['errors'].append("Empty surface")
            validation_results['is_valid'] = False
        
        finite_fraction = np.sum(np.isfinite(surface)) / surface.size
        validation_results['diagnostics']['finite_fraction'] = finite_fraction
        
        if finite_fraction < 0.5:
            validation_results['warnings'].append(f"Surface has {finite_fraction:.1%} finite values")
        
        # Check center coordinates
        invalid_centers = []
        boundary_centers = []
        
        half_height = self.patch_height // 2
        half_width = self.patch_width // 2
        
        for i, (t_idx, k_idx) in enumerate(center_coordinates):
            # Check if center is within surface bounds
            if not (0 <= t_idx < surface_height and 0 <= k_idx < surface_width):
                invalid_centers.append(i)
                continue
            
            # Check if patch would extend beyond boundaries
            if (t_idx - half_height < 0 or t_idx + half_height >= surface_height or
                k_idx - half_width < 0 or k_idx + half_width >= surface_width):
                boundary_centers.append(i)
        
        validation_results['diagnostics']['invalid_centers'] = len(invalid_centers)
        validation_results['diagnostics']['boundary_centers'] = len(boundary_centers)
        
        if invalid_centers:
            validation_results['errors'].append(f"{len(invalid_centers)} centers outside surface bounds")
            validation_results['is_valid'] = False
        
        if boundary_centers:
            validation_results['warnings'].append(f"{len(boundary_centers)} centers near boundaries")
        
        # Test extraction on a sample
        if validation_results['is_valid'] and center_coordinates:
            try:
                sample_patch = self.extract_patch(surface, center_coordinates[0])
                validation_results['diagnostics']['sample_patch_shape'] = sample_patch.shape
                validation_results['diagnostics']['sample_patch_finite'] = np.sum(np.isfinite(sample_patch))
            except Exception as e:
                validation_results['errors'].append(f"Sample extraction failed: {e}")
                validation_results['is_valid'] = False
        
        return validation_results