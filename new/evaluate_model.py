#!/usr/bin/env python3
"""
Main evaluation and visualization script for SABR volatility surface models.

This script provides a command-line interface for comprehensive model evaluation,
performance analysis, and visualization generation.

Usage:
    python evaluate_model.py --results-dir results/training_20241201_120000
    python evaluate_model.py --results-dir results/ --comprehensive-analysis
    python evaluate_model.py --results-dir results/ --visualize --interactive
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.config import ConfigManager
from utils.logging_utils import setup_logging, get_logger
from evaluation.comprehensive_pipeline import (
    ComprehensiveEvaluationPipeline, 
    PipelineConfig,
    create_comprehensive_pipeline
)
from evaluation.performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceAnalysisConfig,
    create_performance_analyzer
)
from visualization.smile_plotter import SmilePlotter
from visualization.surface_plotter import SurfacePlotter

logger = get_logger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize SABR volatility surface model results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation of training results
  python evaluate_model.py --results-dir results/training_20241201_120000

  # Comprehensive analysis with all components
  python evaluate_model.py --results-dir results/ --comprehensive-analysis

  # Generate visualizations only
  python evaluate_model.py --results-dir results/ --visualize-only

  # Performance analysis with custom output
  python evaluate_model.py --results-dir results/ --performance-analysis --output-dir analysis/

  # Interactive visualizations
  python evaluate_model.py --results-dir results/ --visualize --interactive

  # Generate report only
  python evaluate_model.py --results-dir results/ --report-only
        """
    )
    
    # Input and output directories
    parser.add_argument(
        '--results-dir', '-r',
        type=str,
        required=True,
        help='Directory containing training results'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for evaluation results (default: results-dir/evaluation)'
    )
    
    # Analysis types
    parser.add_argument(
        '--comprehensive-analysis',
        action='store_true',
        help='Run complete comprehensive evaluation pipeline'
    )
    
    parser.add_argument(
        '--performance-analysis',
        action='store_true',
        help='Run performance analysis only'
    )
    
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Generate visualizations only'
    )
    
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate reports only (no new analysis)'
    )
    
    # Analysis components
    parser.add_argument(
        '--budget-analysis',
        action='store_true',
        help='Include performance vs HF budget analysis'
    )
    
    parser.add_argument(
        '--residual-analysis',
        action='store_true',
        help='Include residual distribution analysis'
    )
    
    parser.add_argument(
        '--convergence-analysis',
        action='store_true',
        help='Include training convergence analysis'
    )
    
    parser.add_argument(
        '--model-comparison',
        action='store_true',
        help='Include model comparison analysis'
    )
    
    # Visualization options
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Generate interactive plots'
    )
    
    parser.add_argument(
        '--plot-formats',
        nargs='+',
        choices=['png', 'pdf', 'svg', 'html'],
        default=['png'],
        help='Output plot formats (default: png)'
    )
    
    parser.add_argument(
        '--smile-plots',
        action='store_true',
        help='Generate volatility smile plots'
    )
    
    parser.add_argument(
        '--surface-plots',
        action='store_true',
        help='Generate 3D surface plots'
    )
    
    # Report options
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive report'
    )
    
    parser.add_argument(
        '--html-report',
        action='store_true',
        help='Generate HTML report'
    )
    
    parser.add_argument(
        '--executive-summary',
        action='store_true',
        help='Generate executive summary'
    )
    
    # Statistical options
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for statistical tests (default: 0.95)'
    )
    
    parser.add_argument(
        '--statistical-tests',
        action='store_true',
        help='Include statistical significance testing'
    )
    
    # Filtering options
    parser.add_argument(
        '--models',
        nargs='+',
        help='Filter analysis to specific models'
    )
    
    parser.add_argument(
        '--budgets',
        nargs='+',
        type=int,
        help='Filter analysis to specific HF budgets'
    )
    
    parser.add_argument(
        '--experiments',
        nargs='+',
        help='Filter analysis to specific experiment IDs'
    )
    
    # Execution options
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing where possible'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit without running analysis'
    )
    
    return parser


def validate_results_directory(results_dir: str) -> bool:
    """Validate that results directory contains expected files."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return False
    
    # Check for detailed results file
    detailed_results = results_path / "detailed_results.csv"
    if not detailed_results.exists():
        logger.warning(f"No detailed_results.csv found in {results_dir}")
        
        # Look for alternative result files
        csv_files = list(results_path.glob("*.csv"))
        if csv_files:
            logger.info(f"Found CSV files: {[f.name for f in csv_files]}")
        else:
            logger.error("No CSV result files found")
            return False
    
    return True


def create_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    """Create pipeline configuration from arguments."""
    
    # Determine which components to run
    if args.comprehensive_analysis:
        # Run all components
        config = PipelineConfig(
            run_performance_analysis=True,
            run_residual_analysis=True,
            run_convergence_analysis=True,
            run_model_comparison=True,
            run_visualization=args.visualize or args.visualize_only,
            save_plots=True,
            generate_report=True,
            create_summary=True,
            plot_formats=args.plot_formats,
            interactive_plots=args.interactive,
            include_detailed_analysis=True,
            include_statistical_tests=args.statistical_tests
        )
    else:
        # Selective components
        config = PipelineConfig(
            run_performance_analysis=args.performance_analysis or args.budget_analysis,
            run_residual_analysis=args.residual_analysis,
            run_convergence_analysis=args.convergence_analysis,
            run_model_comparison=args.model_comparison,
            run_visualization=args.visualize or args.visualize_only,
            save_plots=args.visualize or args.visualize_only,
            generate_report=args.generate_report,
            create_summary=args.executive_summary,
            plot_formats=args.plot_formats,
            interactive_plots=args.interactive,
            include_detailed_analysis=not args.report_only,
            include_statistical_tests=args.statistical_tests
        )
    
    return config


def create_performance_config(args: argparse.Namespace) -> PerformanceAnalysisConfig:
    """Create performance analysis configuration from arguments."""
    return PerformanceAnalysisConfig(
        save_plots=args.visualize or args.visualize_only,
        interactive_plots=args.interactive,
        plot_format=args.plot_formats[0] if args.plot_formats else 'png',
        confidence_level=args.confidence_level
    )


def print_evaluation_summary(args: argparse.Namespace, config: PipelineConfig):
    """Print evaluation configuration summary."""
    print("\n" + "="*60)
    print("SABR MODEL EVALUATION CONFIGURATION")
    print("="*60)
    
    print(f"\nInput Directory: {args.results_dir}")
    print(f"Output Directory: {args.output_dir or args.results_dir + '/evaluation'}")
    
    print(f"\nAnalysis Components:")
    print(f"  Performance Analysis: {config.run_performance_analysis}")
    print(f"  Residual Analysis: {config.run_residual_analysis}")
    print(f"  Convergence Analysis: {config.run_convergence_analysis}")
    print(f"  Model Comparison: {config.run_model_comparison}")
    print(f"  Visualization: {config.run_visualization}")
    
    print(f"\nOutput Options:")
    print(f"  Generate Report: {config.generate_report}")
    print(f"  Executive Summary: {config.create_summary}")
    print(f"  Interactive Plots: {config.interactive_plots}")
    print(f"  Plot Formats: {config.plot_formats}")
    
    if args.models:
        print(f"\nModel Filter: {args.models}")
    
    if args.budgets:
        print(f"Budget Filter: {args.budgets}")
    
    print("="*60)


def run_comprehensive_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run comprehensive evaluation pipeline."""
    logger.info("Running comprehensive evaluation pipeline")
    
    # Create pipeline configuration
    pipeline_config = create_pipeline_config(args)
    
    # Create pipeline
    pipeline = create_comprehensive_pipeline(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        config=pipeline_config
    )
    
    # Run evaluation
    results = pipeline.run_complete_evaluation()
    
    return results


def run_performance_analysis_only(args: argparse.Namespace) -> Dict[str, Any]:
    """Run performance analysis only."""
    logger.info("Running performance analysis")
    
    # Create performance analyzer configuration
    perf_config = create_performance_config(args)
    
    # Create analyzer
    analyzer = create_performance_analyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        config=perf_config
    )
    
    # Run analysis
    results = analyzer.run_comprehensive_analysis()
    
    return results


def generate_visualizations_only(args: argparse.Namespace) -> Dict[str, Any]:
    """Generate visualizations only."""
    logger.info("Generating visualizations")
    
    output_dir = args.output_dir or f"{args.results_dir}/visualizations"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        'visualizations_generated': [],
        'output_directory': output_dir
    }
    
    # Generate smile plots if requested
    if args.smile_plots or not args.surface_plots:
        logger.info("Generating volatility smile plots")
        smile_plotter = SmilePlotter(output_dir=f"{output_dir}/smile_plots")
        results['visualizations_generated'].append('smile_plots')
    
    # Generate surface plots if requested
    if args.surface_plots or not args.smile_plots:
        logger.info("Generating 3D surface plots")
        surface_plotter = SurfacePlotter(output_dir=f"{output_dir}/surface_plots")
        results['visualizations_generated'].append('surface_plots')
    
    return results


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging()
    
    logger.info("Starting SABR model evaluation")
    
    try:
        # Validate results directory
        if not validate_results_directory(args.results_dir):
            print(f"Error: Invalid results directory: {args.results_dir}")
            sys.exit(1)
        
        # Set default output directory
        if args.output_dir is None:
            args.output_dir = f"{args.results_dir}/evaluation"
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create pipeline configuration
        pipeline_config = create_pipeline_config(args)
        
        # Print configuration summary
        print_evaluation_summary(args, pipeline_config)
        
        # Dry run check
        if args.dry_run:
            print("\nDry run completed. No evaluation performed.")
            return
        
        # Run evaluation based on mode
        start_time = time.time()
        
        if args.comprehensive_analysis or (not args.performance_analysis and not args.visualize_only and not args.report_only):
            results = run_comprehensive_evaluation(args)
            evaluation_type = "Comprehensive Evaluation"
        elif args.performance_analysis:
            results = run_performance_analysis_only(args)
            evaluation_type = "Performance Analysis"
        elif args.visualize_only:
            results = generate_visualizations_only(args)
            evaluation_type = "Visualization Generation"
        else:
            # Default to comprehensive
            results = run_comprehensive_evaluation(args)
            evaluation_type = "Comprehensive Evaluation"
        
        total_time = time.time() - start_time
        
        # Success message
        print(f"\n" + "="*60)
        print(f"{evaluation_type.upper()} COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Output Directory: {args.output_dir}")
        
        # Show results summary
        if 'pipeline_summary' in results:
            summary = results['pipeline_summary']
            print(f"Components Run: {', '.join(summary.get('components_run', []))}")
            print(f"Execution Steps: {summary.get('total_steps', 0)}")
        
        # Show generated files
        output_path = Path(args.output_dir)
        if output_path.exists():
            files = list(output_path.rglob("*"))
            reports = [f for f in files if f.suffix in ['.txt', '.html', '.json']]
            plots = [f for f in files if f.suffix in ['.png', '.pdf', '.svg', '.html']]
            
            if reports:
                print(f"\nGenerated Reports ({len(reports)}):")
                for report in sorted(reports)[:5]:  # Show first 5
                    print(f"  {report.relative_to(output_path)}")
                if len(reports) > 5:
                    print(f"  ... and {len(reports) - 5} more")
            
            if plots:
                print(f"\nGenerated Plots ({len(plots)}):")
                for plot in sorted(plots)[:5]:  # Show first 5
                    print(f"  {plot.relative_to(output_path)}")
                if len(plots) > 5:
                    print(f"  ... and {len(plots) - 5} more")
        
        print(f"\nKey Files:")
        key_files = [
            "comprehensive_evaluation_report.txt",
            "executive_summary.txt",
            "pipeline_results.json",
            "performance_analysis_results.json"
        ]
        
        for key_file in key_files:
            file_path = output_path / key_file
            if file_path.exists():
                print(f"  ✓ {key_file}")
            else:
                print(f"  - {key_file}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("\nEvaluation cancelled by user.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\nError: {e}")
        print("Check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()