"""
Example usage of patch extraction and feature engineering for MDA-CNN.

This script demonstrates how to use the PatchExtractor and FeatureEngineer
classes to prepare data for training the MDA-CNN model.
"""

import numpy as np
import matplotlib.pyplot as plt

from patch_extractor import PatchExtractor, PatchConfig
from feature_engineer import FeatureEngineer, FeatureConfig
from ..data_generation.sabr_params import SABRParams, GridConfig


def create_example_surface(sabr_params: SABRParams, grid_config: GridConfig) -> np.ndarray:
    """Create a simple example volatility surface."""
    strikes = grid_config.get_strikes(sabr_params.F0)
    maturities = grid_config.get_maturities()
    
    surface = np.zeros((len(maturities), len(strikes)))
    
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            moneyness = K / sabr_params.F0
            # Simple volatility smile: higher vol for OTM options
            vol = 0.2 + 0.1 * abs(np.log(moneyness)) + 0.05 * np.sqrt(T)
            surface[i, j] = vol
    
    return surface


def main():
    """Demonstrate patch extraction and feature engineering."""
    print("MDA-CNN Preprocessing Example")
    print("=" * 40)
    
    # 1. Set up configurations
    print("\n1. Setting up configurations...")
    
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
        normalize_features=True
    )
    
    grid_config = GridConfig(
        strike_range=(0.7, 1.3),
        maturity_range=(0.1, 2.0),
        n_strikes=20,
        n_maturities=15
    )
    
    print(f"  - Patch size: {patch_config.patch_size}")
    print(f"  - Feature count: {len(FeatureEngineer(feature_config).feature_names)}")
    print(f"  - Grid size: {grid_config.n_maturities} x {grid_config.n_strikes}")
    
    # 2. Create example data
    print("\n2. Creating example data...")
    
    sabr_params = SABRParams(
        F0=100.0,
        alpha=0.3,
        beta=0.7,
        nu=0.4,
        rho=-0.2
    )
    
    # Create LF surface
    lf_surface = create_example_surface(sabr_params, grid_config)
    
    # Create HF sample points (subset of grid)
    grid_strikes = grid_config.get_strikes(sabr_params.F0)
    grid_maturities = grid_config.get_maturities()
    
    # Sample every 3rd strike and every 2nd maturity
    hf_strikes = grid_strikes[::3]
    hf_maturities = grid_maturities[::2]
    
    print(f"  - LF surface shape: {lf_surface.shape}")
    print(f"  - HF sample points: {len(hf_strikes)} strikes x {len(hf_maturities)} maturities")
    print(f"  - Total HF points: {len(hf_strikes) * len(hf_maturities)}")
    
    # 3. Initialize extractors
    print("\n3. Initializing extractors...")
    
    patch_extractor = PatchExtractor(patch_config)
    feature_engineer = FeatureEngineer(feature_config)
    
    print(f"  - Patch extractor ready (boundary mode: {patch_config.boundary_mode})")
    print(f"  - Feature engineer ready ({len(feature_engineer.feature_names)} features)")
    
    # 4. Align HF points to grid
    print("\n4. Aligning HF points to grid...")
    
    # Create all combinations of HF points
    hf_strikes_flat = []
    hf_maturities_flat = []
    for T in hf_maturities:
        for K in hf_strikes:
            hf_strikes_flat.append(K)
            hf_maturities_flat.append(T)
    
    coordinates = patch_extractor.align_hf_to_grid(
        np.array(hf_strikes_flat),
        np.array(hf_maturities_flat),
        grid_strikes,
        grid_maturities
    )
    
    print(f"  - Aligned {len(coordinates)} HF points to grid coordinates")
    
    # 5. Extract patches
    print("\n5. Extracting patches...")
    
    patches = patch_extractor.extract_patches_batch(lf_surface, coordinates)
    
    print(f"  - Extracted patches shape: {patches.shape}")
    print(f"  - Patch statistics: mean={np.mean(patches):.4f}, std={np.std(patches):.4f}")
    
    # Validate patch extraction
    validation = patch_extractor.validate_extraction(lf_surface, coordinates)
    print(f"  - Patch extraction valid: {validation['is_valid']}")
    if validation['warnings']:
        print(f"  - Warnings: {len(validation['warnings'])}")
    
    # 6. Create point features
    print("\n6. Creating point features...")
    
    # Create SABR parameter list for all points
    sabr_params_list = [sabr_params] * len(coordinates)
    
    # Get Hagan volatilities from LF surface
    hagan_vols = []
    for coord in coordinates:
        hagan_vols.append(lf_surface[coord])
    
    features = feature_engineer.create_features_batch(
        sabr_params_list,
        np.array(hf_strikes_flat),
        np.array(hf_maturities_flat),
        np.array(hagan_vols)
    )
    
    print(f"  - Created features shape: {features.shape}")
    print(f"  - Feature names: {feature_engineer.feature_names}")
    
    # Validate features
    validation = feature_engineer.validate_features(features)
    print(f"  - Feature validation valid: {validation['is_valid']}")
    
    # 7. Normalize features
    print("\n7. Normalizing features...")
    
    feature_engineer.fit_normalization(features)
    normalized_features = feature_engineer.normalize_features(features)
    
    print(f"  - Normalized features shape: {normalized_features.shape}")
    print(f"  - Normalized statistics: mean={np.mean(normalized_features):.4f}, std={np.std(normalized_features):.4f}")
    
    # 8. Display results summary
    print("\n8. Results Summary")
    print("-" * 20)
    print(f"Input LF surface: {lf_surface.shape}")
    print(f"HF sample points: {len(coordinates)}")
    print(f"Extracted patches: {patches.shape}")
    print(f"Point features: {features.shape}")
    print(f"Normalized features: {normalized_features.shape}")
    
    # 9. Visualize a sample patch
    print("\n9. Sample patch visualization...")
    
    if len(patches) > 0:
        sample_patch = patches[0]
        
        plt.figure(figsize=(12, 4))
        
        # Plot LF surface
        plt.subplot(1, 3, 1)
        plt.imshow(lf_surface, aspect='auto', cmap='viridis')
        plt.title('LF Volatility Surface')
        plt.xlabel('Strike Index')
        plt.ylabel('Maturity Index')
        plt.colorbar()
        
        # Plot sample patch
        plt.subplot(1, 3, 2)
        plt.imshow(sample_patch, aspect='auto', cmap='viridis')
        plt.title(f'Sample Patch ({patch_config.patch_size[0]}x{patch_config.patch_size[1]})')
        plt.xlabel('Strike Index')
        plt.ylabel('Maturity Index')
        plt.colorbar()
        
        # Plot feature values
        plt.subplot(1, 3, 3)
        sample_features = normalized_features[0]
        plt.bar(range(len(sample_features)), sample_features)
        plt.title('Sample Normalized Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Normalized Value')
        plt.xticks(range(len(sample_features)), 
                  [name[:8] + '...' if len(name) > 8 else name 
                   for name in feature_engineer.feature_names], 
                  rotation=45)
        
        plt.tight_layout()
        plt.savefig('preprocessing_example.png', dpi=150, bbox_inches='tight')
        print("  - Saved visualization to 'preprocessing_example.png'")
    
    print("\nExample completed successfully!")
    print("The extracted patches and features are ready for MDA-CNN training.")


if __name__ == "__main__":
    main()