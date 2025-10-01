#!/usr/bin/env python3
"""
Main launcher script for SABR MDA-CNN system.

This script provides a unified interface to all main functionality
and handles import path issues.

Usage:
    python run_sabr.py generate --quick-test
    python run_sabr.py train --data-dir data/test --dry-run
    python run_sabr.py evaluate --results-dir results/test --dry-run
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to Python path to handle imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def create_argument_parser():
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        description="SABR MDA-CNN System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  generate    Generate SABR volatility surface data
  train       Train MDA-CNN and baseline models
  evaluate    Evaluate and analyze model results
  test        Run system tests

Examples:
  # Generate test data
  python run_sabr.py generate --quick-test

  # Train MDA-CNN model
  python run_sabr.py train --data-dir data/experiment1 --model mda_cnn

  # Evaluate results
  python run_sabr.py evaluate --results-dir results/training_20241201_120000

  # Run system tests
  python run_sabr.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate training data')
    generate_parser.add_argument('--quick-test', action='store_true', help='Generate small test dataset')
    generate_parser.add_argument('--n-surfaces', type=int, help='Number of surfaces to generate')
    generate_parser.add_argument('--hf-budget', type=int, help='HF budget per surface')
    generate_parser.add_argument('--output-dir', type=str, help='Output directory')
    generate_parser.add_argument('--dry-run', action='store_true', help='Show config without generating')
    generate_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--data-dir', type=str, required=True, help='Training data directory')
    train_parser.add_argument('--model', type=str, choices=['mda_cnn', 'residual_mlp', 'direct_mlp'], 
                             default='mda_cnn', help='Model type')
    train_parser.add_argument('--hf-budget-analysis', action='store_true', help='Run HF budget analysis')
    train_parser.add_argument('--budgets', nargs='+', type=int, default=[50, 100, 200, 500], 
                             help='HF budgets to test')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--dry-run', action='store_true', help='Show config without training')
    train_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate results')
    evaluate_parser.add_argument('--results-dir', type=str, required=True, help='Results directory')
    evaluate_parser.add_argument('--comprehensive-analysis', action='store_true', 
                                help='Run comprehensive analysis')
    evaluate_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    evaluate_parser.add_argument('--interactive', action='store_true', help='Interactive plots')
    evaluate_parser.add_argument('--dry-run', action='store_true', help='Show config without evaluation')
    evaluate_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run system tests')
    test_parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    test_parser.add_argument('--component', type=str, help='Test specific component')
    test_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser

def run_generate(args):
    """Run data generation."""
    print("Running data generation...")
    
    # Import here to avoid issues
    try:
        from utils.config import ConfigManager
        from utils.logging_utils import setup_logging
        from utils.reproducibility import set_random_seed
        from data_generation.data_orchestrator import DataOrchestrator
        
        setup_logging()
        set_random_seed(42)
        
        # Create configuration
        config_manager = ConfigManager()
        if args.quick_test:
            config = config_manager.create_default_config("quick_test")
            config.data_gen_config.n_parameter_sets = 10
            config.data_gen_config.hf_budget = 20
            config.data_gen_config.mc_paths = 1000
            config.output_dir = "data/quick_test"
        else:
            config = config_manager.create_default_config("data_generation")
            if args.n_surfaces:
                config.data_gen_config.n_parameter_sets = args.n_surfaces
            if args.hf_budget:
                config.data_gen_config.hf_budget = args.hf_budget
            if args.output_dir:
                config.output_dir = args.output_dir
        
        print(f"Configuration:")
        print(f"  Surfaces: {config.data_gen_config.n_parameter_sets}")
        print(f"  HF Budget: {config.data_gen_config.hf_budget}")
        print(f"  Output: {config.output_dir}")
        
        if args.dry_run:
            print("Dry run completed.")
            return
        
        # Generate data
        orchestrator = DataOrchestrator(config)
        orchestrator.generate_training_data()
        
        print(f"✓ Data generation completed successfully!")
        print(f"  Output directory: {config.output_dir}")
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def run_train(args):
    """Run model training."""
    print("Running model training...")
    
    try:
        from utils.config import ConfigManager
        from utils.logging_utils import setup_logging
        from utils.reproducibility import set_random_seed
        
        setup_logging()
        set_random_seed(42)
        
        # Check data directory
        if not Path(args.data_dir).exists():
            print(f"✗ Data directory not found: {args.data_dir}")
            print("  Run data generation first: python run_sabr.py generate --quick-test")
            return
        
        print(f"Configuration:")
        print(f"  Data directory: {args.data_dir}")
        print(f"  Model: {args.model}")
        if args.hf_budget_analysis:
            print(f"  HF Budget Analysis: {args.budgets}")
        
        if args.dry_run:
            print("Dry run completed.")
            return
        
        print("✓ Training configuration validated!")
        print("  Note: Full training implementation requires TensorFlow")
        print("  Use the individual train_model.py script for complete training")
        
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def run_evaluate(args):
    """Run evaluation."""
    print("Running evaluation...")
    
    try:
        from utils.logging_utils import setup_logging
        
        setup_logging()
        
        # Check results directory
        if not Path(args.results_dir).exists():
            print(f"✗ Results directory not found: {args.results_dir}")
            print("  Run training first to generate results")
            return
        
        print(f"Configuration:")
        print(f"  Results directory: {args.results_dir}")
        print(f"  Comprehensive analysis: {args.comprehensive_analysis}")
        print(f"  Visualizations: {args.visualize}")
        
        if args.dry_run:
            print("Dry run completed.")
            return
        
        print("✓ Evaluation configuration validated!")
        print("  Note: Full evaluation requires trained models")
        print("  Use the individual evaluate_model.py script for complete evaluation")
        
    except Exception as e:
        print(f"✗ Evaluation setup failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def run_test(args):
    """Run system tests."""
    print("Running system tests...")
    
    try:
        # Test basic imports
        print("Testing core imports...")
        
        from utils.config import ConfigManager
        print("  ✓ Configuration system")
        
        from utils.logging_utils import setup_logging
        print("  ✓ Logging utilities")
        
        from data_generation.sabr_params import SABRParams, GridConfig
        print("  ✓ SABR parameters")
        
        from data_generation.hagan_surface_generator import HaganSurfaceGenerator
        print("  ✓ Hagan surface generator")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        
        # Test configuration
        config_manager = ConfigManager()
        config = config_manager.create_default_config("test")
        print("  ✓ Configuration creation")
        
        # Test SABR parameters
        sabr_params = SABRParams(F0=100.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3)
        grid_config = GridConfig(strike_range=(80, 120), maturity_range=(0.1, 2.0), n_strikes=10, n_maturities=5)
        print("  ✓ SABR parameter creation")
        
        # Test Hagan surface generation
        hagan_gen = HaganSurfaceGenerator()
        surface = hagan_gen.generate_surface(sabr_params, grid_config)
        print(f"  ✓ Hagan surface generation (shape: {surface.shape})")
        
        print(f"\n✓ All system tests passed!")
        print(f"  System is ready for use")
        
    except Exception as e:
        print(f"✗ System tests failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

def main():
    """Main launcher function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("SABR MDA-CNN System")
    print("=" * 30)
    
    if args.command == 'generate':
        run_generate(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'test':
        run_test(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()