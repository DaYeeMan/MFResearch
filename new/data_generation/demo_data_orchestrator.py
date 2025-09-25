"""
Demo script for SABR data generation orchestrator.

This script demonstrates how to use the DataGenerationOrchestrator to generate
a complete dataset of SABR volatility surfaces with both high-fidelity Monte Carlo
and low-fidelity Hagan analytical surfaces.
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List

from data_orchestrator import (
    DataGenerationOrchestrator, DataGenerationConfig, GenerationProgress,
    create_default_generation_config
)
from sabr_params import GridConfig, create_default_grid_config
from sabr_mc_generator import MCConfig, create_default_mc_config
from hagan_surface_generator import HaganConfig, create_default_hagan_config
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_utils import setup_logging, get_logger
from utils.common import Timer

# Set up logging
setup_logging(log_level="INFO", console_output=True)
logger = get_logger(__name__)


def progress_callback(progress: GenerationProgress):
    """Progress callback function to display generation progress."""
    if progress.completed_surfaces % 10 == 0 or progress.get_progress_percentage() >= 100:
        print(f"Progress: {progress.completed_surfaces}/{progress.total_surfaces} "
              f"({progress.get_progress_percentage():.1f}%) - "
              f"Stage: {progress.current_stage} - "
              f"ETA: {progress.get_eta_string()}")


def create_demo_config() -> DataGenerationConfig:
    """Create configuration for demo."""
    config = create_default_generation_config()
    
    # Adjust for demo (smaller dataset)
    config.n_parameter_sets = 50
    config.sampling_strategy = "latin_hypercube"
    config.hf_budget = 100
    config.output_dir = "new/data/demo"
    config.parallel_generation = True
    config.max_workers = 2
    config.quality_checks = True
    config.outlier_detection = True
    config.random_seed = 42
    
    return config


def create_demo_grid_config() -> GridConfig:
    """Create grid configuration for demo."""
    grid_config = create_default_grid_config()
    
    # Smaller grid for faster generation
    grid_config.strike_range = (0.7, 1.5)
    grid_config.maturity_range = (0.25, 3.0)
    grid_config.n_strikes = 15
    grid_config.n_maturities = 8
    grid_config.log_strikes = True
    grid_config.log_maturities = False
    
    return grid_config


def create_demo_mc_config() -> MCConfig:
    """Create MC configuration for demo."""
    mc_config = create_default_mc_config()
    
    # Faster settings for demo
    mc_config.n_paths = 25000
    mc_config.n_steps = 100
    mc_config.convergence_check = True
    mc_config.convergence_tolerance = 1e-3
    mc_config.max_iterations = 3
    mc_config.parallel = True
    mc_config.random_seed = 42
    
    return mc_config


def visualize_sample_surfaces(orchestrator: DataGenerationOrchestrator, surfaces: List, n_samples: int = 3):
    """Visualize a few sample surfaces from the generated dataset."""
    if len(surfaces) < n_samples:
        n_samples = len(surfaces)
    
    # Select random surfaces to visualize
    np.random.seed(42)
    sample_indices = np.random.choice(len(surfaces), n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_indices):
        surface = surfaces[idx]
        
        # Create meshgrid for plotting
        K, T = np.meshgrid(surface.strikes, surface.maturities)
        
        # Plot HF surface
        im1 = axes[i, 0].contourf(K, T, surface.hf_surface, levels=20, cmap='viridis')
        axes[i, 0].set_title(f'HF Surface {idx}\n(F0={surface.parameters.F0:.1f}, α={surface.parameters.alpha:.3f})')
        axes[i, 0].set_xlabel('Strike')
        axes[i, 0].set_ylabel('Maturity')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Plot LF surface
        im2 = axes[i, 1].contourf(K, T, surface.lf_surface, levels=20, cmap='viridis')
        axes[i, 1].set_title(f'LF Surface {idx}\n(β={surface.parameters.beta:.3f}, ν={surface.parameters.nu:.3f})')
        axes[i, 1].set_xlabel('Strike')
        axes[i, 1].set_ylabel('Maturity')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Plot residuals
        im3 = axes[i, 2].contourf(K, T, surface.residuals, levels=20, cmap='RdBu_r')
        axes[i, 2].set_title(f'Residuals {idx}\n(ρ={surface.parameters.rho:.3f})')
        axes[i, 2].set_xlabel('Strike')
        axes[i, 2].set_ylabel('Maturity')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(orchestrator.config.output_dir)
    plot_path = output_dir / "sample_surfaces.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved sample surfaces plot to {plot_path}")
    plt.show()


def analyze_dataset_quality(result: dict):
    """Analyze and report on dataset quality."""
    surfaces = result['surfaces']
    metadata = result['metadata']
    
    logger.info("=== Dataset Quality Analysis ===")
    
    # Basic statistics
    logger.info(f"Total surfaces generated: {len(surfaces)}")
    logger.info(f"Success rate: {metadata['success_rate']:.2%}")
    logger.info(f"Failed surfaces: {metadata['n_surfaces_failed']}")
    
    # Quality scores
    quality_scores = [s.quality_metrics.get('quality_score', 0) for s in surfaces if s.quality_metrics]
    if quality_scores:
        logger.info(f"Quality scores - Mean: {np.mean(quality_scores):.3f}, "
                   f"Std: {np.std(quality_scores):.3f}, "
                   f"Min: {np.min(quality_scores):.3f}, "
                   f"Max: {np.max(quality_scores):.3f}")
    
    # Generation time statistics
    gen_times = [s.generation_time for s in surfaces]
    if gen_times:
        logger.info(f"Generation times - Mean: {np.mean(gen_times):.2f}s, "
                   f"Total: {np.sum(gen_times):.1f}s")
    
    # Parameter space coverage
    if 'parameter_statistics' in metadata['dataset_statistics']:
        param_stats = metadata['dataset_statistics']['parameter_statistics']
        logger.info("Parameter space coverage:")
        for param, stats in param_stats.items():
            logger.info(f"  {param}: [{stats['min']:.3f}, {stats['max']:.3f}] "
                       f"(mean: {stats['mean']:.3f}, std: {stats['std']:.3f})")
    
    # Surface statistics
    if 'surface_statistics' in metadata['dataset_statistics']:
        surf_stats = metadata['dataset_statistics']['surface_statistics']
        logger.info("Surface value ranges:")
        for surf_type, stats in surf_stats.items():
            logger.info(f"  {surf_type}: [{stats['min']:.4f}, {stats['max']:.4f}] "
                       f"(mean: {stats['mean']:.4f}, count: {stats['count']})")
    
    # Data splits
    splits = metadata['data_splits']
    logger.info(f"Data splits - Train: {splits['train']}, "
               f"Val: {splits['val']}, Test: {splits['test']}")


def main():
    """Main demo function."""
    logger.info("Starting SABR Data Generation Orchestrator Demo")
    
    # Create configurations
    config = create_demo_config()
    grid_config = create_demo_grid_config()
    mc_config = create_demo_mc_config()
    hagan_config = create_default_hagan_config()
    
    logger.info(f"Demo configuration:")
    logger.info(f"  Parameter sets: {config.n_parameter_sets}")
    logger.info(f"  Grid size: {grid_config.n_maturities} x {grid_config.n_strikes}")
    logger.info(f"  MC paths: {mc_config.n_paths}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Parallel generation: {config.parallel_generation}")
    
    # Create orchestrator
    orchestrator = DataGenerationOrchestrator(
        config=config,
        grid_config=grid_config,
        mc_config=mc_config,
        hagan_config=hagan_config
    )
    
    # Add progress callback
    orchestrator.add_progress_callback(progress_callback)
    
    # Generate complete dataset
    logger.info("Starting dataset generation...")
    
    with Timer("Complete dataset generation"):
        result = orchestrator.generate_complete_dataset()
    
    # Analyze results
    analyze_dataset_quality(result)
    
    # Visualize sample surfaces
    if result['surfaces']:
        logger.info("Creating sample surface visualizations...")
        try:
            visualize_sample_surfaces(orchestrator, result['surfaces'])
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
    
    # Report file locations
    output_dir = Path(config.output_dir)
    logger.info(f"Generated data saved to: {output_dir}")
    logger.info(f"  Raw data: {output_dir / 'raw'}")
    logger.info(f"  Data splits: {output_dir / 'splits'}")
    logger.info(f"  Metadata: {output_dir / 'raw' / 'generation_metadata.json'}")
    
    logger.info("Demo completed successfully!")
    
    return result


if __name__ == "__main__":
    try:
        result = main()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise