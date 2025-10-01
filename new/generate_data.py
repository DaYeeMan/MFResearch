#!/usr/bin/env python3
"""
Main data generation script for SABR volatility surface modeling.

This script provides a command-line interface for generating comprehensive
SABR volatility surface datasets using both Monte Carlo simulations and
Hagan analytical formulas.

Usage:
    python generate_data.py --config configs/default_config.yaml
    python generate_data.py --n-surfaces 1000 --hf-budget 200 --output-dir data/experiment1
    python generate_data.py --quick-test  # Generate small test dataset
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config import ConfigManager, ExperimentConfig
from utils.logging_utils import setup_logging, get_logger
from utils.reproducibility import set_random_seed
from data_generation.data_orchestrator import DataOrchestrator
from data_generation.sabr_params import SABRParams, GridConfig

logger = get_logger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate SABR volatility surface data for MDA-CNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data using default configuration
  python generate_data.py

  # Generate data with custom parameters
  python generate_data.py --n-surfaces 500 --hf-budget 100 --mc-paths 50000

  # Generate quick test dataset
  python generate_data.py --quick-test

  # Use custom configuration file
  python generate_data.py --config configs/custom_config.yaml

  # Generate data with specific output directory
  python generate_data.py --output-dir data/my_experiment --n-surfaces 200
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file (default: configs/default_config.yaml)'
    )
    
    # Data generation parameters
    parser.add_argument(
        '--n-surfaces', '-n',
        type=int,
        help='Number of surface parameter sets to generate'
    )
    
    parser.add_argument(
        '--hf-budget',
        type=int,
        help='Number of high-fidelity Monte Carlo points per surface'
    )
    
    parser.add_argument(
        '--mc-paths',
        type=int,
        help='Number of Monte Carlo simulation paths'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for generated data'
    )
    
    # Grid configuration
    parser.add_argument(
        '--n-strikes',
        type=int,
        help='Number of strike points in grid'
    )
    
    parser.add_argument(
        '--n-maturities',
        type=int,
        help='Number of maturity points in grid'
    )
    
    parser.add_argument(
        '--strike-range',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='Strike range as two values: min max'
    )
    
    parser.add_argument(
        '--maturity-range',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='Maturity range as two values: min max'
    )
    
    # SABR parameter ranges
    parser.add_argument(
        '--alpha-range',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='Alpha parameter range'
    )
    
    parser.add_argument(
        '--beta-range',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='Beta parameter range'
    )
    
    parser.add_argument(
        '--nu-range',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='Nu parameter range'
    )
    
    parser.add_argument(
        '--rho-range',
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        help='Rho parameter range'
    )
    
    # Execution options
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Generate small test dataset for quick validation'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing for data generation'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit without generating data'
    )
    
    return parser


def override_config_from_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Override configuration with command-line arguments."""
    
    # Data generation parameters
    if args.n_surfaces is not None:
        config.data_gen_config.n_parameter_sets = args.n_surfaces
    
    if args.hf_budget is not None:
        config.data_gen_config.hf_budget = args.hf_budget
    
    if args.mc_paths is not None:
        config.data_gen_config.mc_paths = args.mc_paths
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    # Grid configuration
    if args.n_strikes is not None:
        config.grid_config.n_strikes = args.n_strikes
    
    if args.n_maturities is not None:
        config.grid_config.n_maturities = args.n_maturities
    
    if args.strike_range is not None:
        config.grid_config.strike_range = tuple(args.strike_range)
    
    if args.maturity_range is not None:
        config.grid_config.maturity_range = tuple(args.maturity_range)
    
    # SABR parameter ranges
    if args.alpha_range is not None:
        config.sabr_params.alpha_range = tuple(args.alpha_range)
    
    if args.beta_range is not None:
        config.sabr_params.beta_range = tuple(args.beta_range)
    
    if args.nu_range is not None:
        config.sabr_params.nu_range = tuple(args.nu_range)
    
    if args.rho_range is not None:
        config.sabr_params.rho_range = tuple(args.rho_range)
    
    return config


def create_quick_test_config() -> ExperimentConfig:
    """Create configuration for quick test dataset."""
    config_manager = ConfigManager()
    config = config_manager.create_default_config("quick_test")
    
    # Small dataset for quick testing
    config.data_gen_config.n_parameter_sets = 10
    config.data_gen_config.hf_budget = 20
    config.data_gen_config.mc_paths = 1000
    
    # Smaller grid
    config.grid_config.n_strikes = 15
    config.grid_config.n_maturities = 8
    
    # Quick output directory
    config.output_dir = "data/quick_test"
    
    return config


def print_configuration_summary(config: ExperimentConfig):
    """Print a summary of the configuration."""
    print("\n" + "="*60)
    print("SABR DATA GENERATION CONFIGURATION")
    print("="*60)
    
    print(f"\nExperiment: {config.name}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Random Seed: {config.data_gen_config.random_seed}")
    
    print(f"\nData Generation:")
    print(f"  Parameter Sets: {config.data_gen_config.n_parameter_sets}")
    print(f"  HF Budget: {config.data_gen_config.hf_budget}")
    print(f"  MC Paths: {config.data_gen_config.mc_paths}")
    
    print(f"\nGrid Configuration:")
    print(f"  Strikes: {config.grid_config.n_strikes} points")
    print(f"  Strike Range: {config.grid_config.strike_range}")
    print(f"  Maturities: {config.grid_config.n_maturities} points")
    print(f"  Maturity Range: {config.grid_config.maturity_range}")
    
    print(f"\nSABR Parameter Ranges:")
    print(f"  Alpha: {config.sabr_params.alpha_range}")
    print(f"  Beta: {config.sabr_params.beta_range}")
    print(f"  Nu: {config.sabr_params.nu_range}")
    print(f"  Rho: {config.sabr_params.rho_range}")
    print(f"  Forward: {config.sabr_params.forward_range}")
    
    # Estimate data size and time
    total_surfaces = config.data_gen_config.n_parameter_sets
    grid_size = config.grid_config.n_strikes * config.grid_config.n_maturities
    hf_points = config.data_gen_config.hf_budget
    
    print(f"\nEstimated Output:")
    print(f"  Total Surfaces: {total_surfaces}")
    print(f"  Grid Points per Surface: {grid_size}")
    print(f"  HF Points per Surface: {hf_points}")
    print(f"  Total HF Simulations: {total_surfaces * hf_points}")
    
    print("="*60)


def main():
    """Main data generation function."""
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging()
    
    logger.info("Starting SABR data generation")
    
    try:
        # Set random seed
        set_random_seed(args.seed)
        logger.info(f"Set random seed to {args.seed}")
        
        # Load or create configuration
        if args.quick_test:
            logger.info("Creating quick test configuration")
            config = create_quick_test_config()
        else:
            logger.info(f"Loading configuration from {args.config}")
            config_manager = ConfigManager()
            
            if Path(args.config).exists():
                config = config_manager.load_config(args.config)
            else:
                logger.warning(f"Configuration file {args.config} not found, using default")
                config = config_manager.create_default_config("data_generation")
        
        # Override with command-line arguments
        config = override_config_from_args(config, args)
        
        # Print configuration summary
        print_configuration_summary(config)
        
        # Dry run - just show configuration and exit
        if args.dry_run:
            print("\nDry run completed. No data generated.")
            return
        
        # Confirm before proceeding with large datasets
        if not args.quick_test and config.data_gen_config.n_parameter_sets > 100:
            response = input(f"\nGenerate {config.data_gen_config.n_parameter_sets} surfaces? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Data generation cancelled.")
                return
        
        # Create data orchestrator
        logger.info("Initializing data orchestrator")
        orchestrator = DataOrchestrator(config)
        
        # Generate data
        start_time = time.time()
        logger.info("Starting data generation...")
        
        orchestrator.generate_training_data()
        
        generation_time = time.time() - start_time
        
        # Success message
        print(f"\n" + "="*60)
        print("DATA GENERATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Generation Time: {generation_time:.2f} seconds")
        print(f"Output Directory: {config.output_dir}")
        print(f"Surfaces Generated: {config.data_gen_config.n_parameter_sets}")
        
        # Show output structure
        output_path = Path(config.output_dir)
        if output_path.exists():
            print(f"\nGenerated Files:")
            for file_path in sorted(output_path.rglob("*")):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  {file_path.relative_to(output_path)} ({size_mb:.1f} MB)")
        
        print(f"\nNext Steps:")
        print(f"  1. Train models: python train_model.py --data-dir {config.output_dir}")
        print(f"  2. Evaluate results: python evaluate_model.py --results-dir results/")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Data generation interrupted by user")
        print("\nData generation cancelled by user.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        print(f"\nError: {e}")
        print("Check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()