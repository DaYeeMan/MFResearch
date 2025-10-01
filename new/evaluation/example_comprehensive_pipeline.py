"""
Example usage of the comprehensive evaluation pipeline for SABR volatility surface models.

This script demonstrates the complete evaluation pipeline that generates all
required analysis and reports as specified in task 15.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.comprehensive_pipeline import (
    ComprehensiveEvaluationPipeline,
    PipelineConfig,
    create_comprehensive_pipeline
)
from evaluation.example_performance_analysis import create_synthetic_results_data
from utils.logging_utils import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)


def run_complete_pipeline_example():
    """Run the complete comprehensive evaluation pipeline."""
    logger.info("Starting comprehensive evaluation pipeline example")
    
    # Create synthetic results data
    results_dir = "new/results/comprehensive_pipeline_example"
    create_synthetic_results_data(results_dir)
    
    # Create pipeline configuration
    config = PipelineConfig(
        run_performance_analysis=True,
        run_residual_analysis=True,
        run_convergence_analysis=True,
        run_model_comparison=True,
        run_visualization=True,
        save_plots=True,
        generate_report=True,
        create_summary=True,
        plot_formats=['png', 'pdf'],
        interactive_plots=True,
        include_detailed_analysis=True,
        include_statistical_tests=True
    )
    
    # Create comprehensive pipeline
    pipeline = create_comprehensive_pipeline(
        results_dir=results_dir,
        output_dir=f"{results_dir}/comprehensive_evaluation",
        config=config
    )
    
    # Run complete evaluation
    logger.info("Running complete evaluation pipeline...")
    results = pipeline.run_complete_evaluation()
    
    # Print pipeline summary
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION PIPELINE RESULTS")
    print("="*70)
    
    # Pipeline execution summary
    if 'pipeline_summary' in results:
        summary = results['pipeline_summary']
        print(f"\nPipeline Execution Summary:")
        print(f"• Execution time: {summary['execution_time_seconds']:.2f} seconds")
        print(f"• Total steps: {summary['total_steps']}")
        print(f"• Successful steps: {summary['successful_steps']}")
        print(f"• Failed steps: {summary['failed_steps']}")
        print(f"• Components run: {', '.join(summary['components_run'])}")
    
    # Performance analysis results
    if 'performance_analysis' in results:
        print(f"\n1. Performance Analysis:")
        perf_results = results['performance_analysis']
        
        if 'budget_analysis' in perf_results:
            budget_analysis = perf_results['budget_analysis']
            
            if 'summary_statistics' in budget_analysis:
                stats = budget_analysis['summary_statistics']
                print(f"   • Models analyzed: {len(stats)}")
                
                # Find best model
                best_model = min(stats.keys(), key=lambda k: stats[k]['best_mse'])
                best_mse = stats[best_model]['best_mse']
                print(f"   • Best model: {best_model} (MSE: {best_mse:.6f})")
            
            if 'plots' in budget_analysis:
                plots = budget_analysis['plots']
                print(f"   • Performance plots generated: {len(plots)}")
    
    # Residual analysis results
    if 'residual_analysis' in results:
        print(f"\n2. Residual Analysis:")
        residual_results = results['residual_analysis']
        
        if 'improvement_statistics' in residual_results:
            stats = residual_results['improvement_statistics']
            if 'improvement_percent' in stats:
                improvement = stats['improvement_percent']
                significance = "Yes" if stats.get('is_significant', False) else "No"
                print(f"   • ML improvement over baseline: {improvement:.1f}%")
                print(f"   • Statistically significant: {significance}")
        
        if 'residual_patterns' in residual_results:
            patterns = residual_results['residual_patterns']
            if 'by_model_type' in patterns:
                print(f"   • Residual patterns analyzed for {len(patterns['by_model_type'])} model types")
    
    # Convergence analysis results
    if 'convergence_analysis' in results:
        print(f"\n3. Convergence Analysis:")
        convergence_results = results['convergence_analysis']
        
        if 'convergence_statistics' in convergence_results:
            stats = convergence_results['convergence_statistics']
            if 'total_experiments' in stats:
                print(f"   • Training experiments analyzed: {stats['total_experiments']}")
            
            if 'mean_convergence_epoch' in stats:
                avg_epochs = stats['mean_convergence_epoch']
                print(f"   • Average convergence epoch: {avg_epochs:.1f}")
        
        if 'insights' in convergence_results:
            insights = convergence_results['insights']
            if 'fastest_converging_model' in insights:
                fastest = insights['fastest_converging_model']
                if fastest:
                    print(f"   • Fastest converging model: {fastest}")
    
    # Model comparison results
    if 'model_comparison' in results:
        print(f"\n4. Model Comparison:")
        comparison_results = results['model_comparison']
        
        if 'rankings' in comparison_results:
            rankings = comparison_results['rankings']
            print(f"   • Model rankings computed for {len(rankings)} HF budgets")
            
            # Show best model for each budget
            for budget, ranking in rankings.items():
                if ranking:
                    best_model, best_score = ranking[0]
                    print(f"     - Budget {budget}: {best_model} (MSE: {best_score:.6f})")
        
        if 'pairwise_comparisons' in comparison_results:
            comparisons = comparison_results['pairwise_comparisons']
            print(f"   • Pairwise comparisons: {len(comparisons)}")
    
    # Visualization results
    if 'visualization' in results:
        print(f"\n5. Visualization:")
        viz_results = results['visualization']
        
        if 'plots_generated' in viz_results:
            plots = viz_results['plots_generated']
            print(f"   • Static plots generated: {len(plots)}")
        
        if 'interactive_plots' in viz_results:
            interactive = viz_results['interactive_plots']
            print(f"   • Interactive plots generated: {len(interactive)}")
    
    # Report generation results
    if 'report' in results:
        print(f"\n6. Report Generation:")
        report_results = results['report']
        
        if 'text_report_path' in report_results:
            print(f"   • Text report: {report_results['text_report_path']}")
        
        if 'html_report_path' in report_results:
            print(f"   • HTML report: {report_results['html_report_path']}")
        
        if 'report_sections' in report_results:
            sections = report_results['report_sections']
            print(f"   • Report sections: {', '.join(sections)}")
    
    # Executive summary results
    if 'executive_summary' in results:
        print(f"\n7. Executive Summary:")
        summary_results = results['executive_summary']
        
        if 'summary_path' in summary_results:
            print(f"   • Executive summary: {summary_results['summary_path']}")
        
        if 'metrics_path' in summary_results:
            print(f"   • Summary metrics: {summary_results['metrics_path']}")
        
        if 'key_metrics' in summary_results:
            metrics = summary_results['key_metrics']
            if 'best_model_overall' in metrics:
                best_model = metrics['best_model_overall']
                best_mse = metrics.get('best_performance_mse', 'N/A')
                print(f"   • Best model overall: {best_model} (MSE: {best_mse})")
    
    # Output directory information
    output_dir = pipeline.output_dir
    print(f"\nAll results saved to: {output_dir}")
    print(f"Key files:")
    print(f"• Comprehensive report: {output_dir}/comprehensive_evaluation_report.txt")
    print(f"• Executive summary: {output_dir}/executive_summary.txt")
    print(f"• Pipeline results: {output_dir}/pipeline_results.json")
    print(f"• Execution log: {output_dir}/pipeline_execution_log.json")
    
    print("\n" + "="*70)
    
    return results


def run_custom_pipeline_example():
    """Run pipeline with custom configuration."""
    logger.info("Running custom pipeline configuration example")
    
    results_dir = "new/results/comprehensive_pipeline_example"
    
    # Custom configuration - only run specific components
    custom_config = PipelineConfig(
        run_performance_analysis=True,
        run_residual_analysis=True,
        run_convergence_analysis=False,  # Skip convergence analysis
        run_model_comparison=True,
        run_visualization=False,  # Skip visualization
        save_plots=True,
        generate_report=True,
        create_summary=True,
        plot_formats=['png'],  # Only PNG format
        interactive_plots=False,  # No interactive plots
        include_detailed_analysis=False,  # Simplified analysis
        include_statistical_tests=True
    )
    
    # Create pipeline with custom config
    pipeline = ComprehensiveEvaluationPipeline(
        results_dir=results_dir,
        output_dir=f"{results_dir}/custom_evaluation",
        config=custom_config
    )
    
    # Run evaluation
    results = pipeline.run_complete_evaluation()
    
    print("\n" + "="*50)
    print("CUSTOM PIPELINE CONFIGURATION RESULTS")
    print("="*50)
    
    components_run = results.get('pipeline_summary', {}).get('components_run', [])
    print(f"Components executed: {', '.join(components_run)}")
    
    # Show what was skipped
    all_components = ['performance_analysis', 'residual_analysis', 'convergence_analysis', 
                     'model_comparison', 'visualization']
    skipped = [comp for comp in all_components if comp not in components_run]
    if skipped:
        print(f"Components skipped: {', '.join(skipped)}")
    
    print("="*50)
    
    return results


def demonstrate_pipeline_components():
    """Demonstrate individual pipeline components."""
    logger.info("Demonstrating individual pipeline components")
    
    results_dir = "new/results/comprehensive_pipeline_example"
    
    # Create pipeline
    pipeline = create_comprehensive_pipeline(results_dir)
    
    print("\n" + "="*40)
    print("INDIVIDUAL COMPONENT DEMONSTRATION")
    print("="*40)
    
    # Run individual components
    print("\n1. Performance Analysis Component:")
    pipeline._run_performance_analysis()
    perf_results = pipeline.pipeline_results.get('performance_analysis', {})
    print(f"   • Analysis completed: {'budget_analysis' in perf_results}")
    
    print("\n2. Residual Analysis Component:")
    pipeline._run_residual_analysis()
    residual_results = pipeline.pipeline_results.get('residual_analysis', {})
    print(f"   • Analysis completed: {'improvement_statistics' in residual_results}")
    
    print("\n3. Model Comparison Component:")
    pipeline._run_model_comparison()
    comparison_results = pipeline.pipeline_results.get('model_comparison', {})
    print(f"   • Analysis completed: {'rankings' in comparison_results}")
    
    print("\n4. Report Generation:")
    pipeline._generate_comprehensive_report()
    report_results = pipeline.pipeline_results.get('report', {})
    print(f"   • Report generated: {'text_report_path' in report_results}")
    
    print("="*40)


def main():
    """Run all comprehensive pipeline examples."""
    print("SABR Volatility Surface Comprehensive Evaluation Pipeline")
    print("=" * 70)
    
    try:
        # 1. Run complete pipeline
        logger.info("Running complete pipeline example")
        complete_results = run_complete_pipeline_example()
        
        # 2. Run custom configuration example
        logger.info("Running custom configuration example")
        custom_results = run_custom_pipeline_example()
        
        # 3. Demonstrate individual components
        logger.info("Demonstrating individual components")
        demonstrate_pipeline_components()
        
        print("\n" + "=" * 70)
        print("ALL COMPREHENSIVE PIPELINE EXAMPLES COMPLETED SUCCESSFULLY!")
        print("\nKey Deliverables Generated:")
        print("✓ Performance vs HF budget analysis plots")
        print("✓ Residual distribution analysis before/after ML correction")
        print("✓ Training convergence visualization and analysis")
        print("✓ Automated report generation with key metrics and plots")
        print("✓ Comprehensive evaluation pipeline that generates all analysis")
        print("\nCheck the output directories for all generated files and reports.")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Comprehensive pipeline example failed: {e}")
        raise


if __name__ == "__main__":
    main()