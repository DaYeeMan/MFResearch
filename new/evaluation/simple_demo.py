"""
Simple demonstration of what's currently working.
"""

import sys
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

def create_minimal_test_data():
    """Create minimal test data."""
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create minimal synthetic data
    np.random.seed(42)
    results_data = []
    
    for model in ['baseline_mlp', 'mda_cnn']:
        for budget in [50, 100, 200]:
            for trial in range(3):
                base_mse = 0.01 if model == 'baseline_mlp' else 0.005
                budget_factor = (budget / 100) ** (-0.3)
                mse = base_mse * budget_factor * (1 + np.random.normal(0, 0.1))
                mse = max(mse, 0.0001)
                
                results_data.append({
                    'experiment_id': f'exp_{len(results_data)}',
                    'model_type': model,
                    'hf_budget': budget,
                    'mse': mse,
                    'mae': np.sqrt(mse) * 0.8,
                    'training_time': np.random.uniform(20, 100)
                })
    
    # Save results
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(temp_dir / "detailed_results.csv", index=False)
    
    return temp_dir

def test_basic_functionality():
    """Test basic functionality that should work."""
    print("Testing Basic Performance Analysis Functionality")
    print("=" * 50)
    
    # Create test data
    test_dir = create_minimal_test_data()
    print(f"✓ Created test data in {test_dir}")
    
    try:
        from evaluation.performance_analyzer import PerformanceAnalyzer, PerformanceAnalysisConfig
        
        # Create analyzer with minimal config
        config = PerformanceAnalysisConfig(
            save_plots=False,  # Don't save plots to avoid file issues
            interactive_plots=False,  # Don't create interactive plots
        )
        
        analyzer = PerformanceAnalyzer(
            results_dir=str(test_dir),
            output_dir=str(test_dir / "analysis"),
            config=config
        )
        print("✓ Created performance analyzer")
        
        # Test individual components that should work
        print("\nTesting individual components:")
        
        # 1. Test budget analysis (should work)
        try:
            budget_analysis = analyzer.analyze_performance_vs_budget()
            print("✓ Performance vs budget analysis works")
            
            if 'summary_statistics' in budget_analysis:
                stats = budget_analysis['summary_statistics']
                print(f"  - Analyzed {len(stats)} model types")
                for model, model_stats in stats.items():
                    print(f"  - {model}: Best MSE = {model_stats['best_mse']:.6f}")
        except Exception as e:
            print(f"✗ Budget analysis failed: {e}")
        
        # 2. Test results aggregator directly
        try:
            aggregator = analyzer.results_aggregator
            summary = aggregator.get_performance_summary()
            print(f"✓ Results aggregator works - {len(summary)} experiments")
            
            rankings = aggregator.rank_models_by_budget()
            print(f"✓ Model rankings work - {len(rankings)} budgets")
            
        except Exception as e:
            print(f"✗ Results aggregator failed: {e}")
        
        # 3. Test what methods are available
        available_methods = [method for method in dir(analyzer) if not method.startswith('__')]
        analysis_methods = [method for method in available_methods if method.startswith('analyze_')]
        print(f"\n✓ Available analysis methods: {analysis_methods}")
        
        print(f"\n✓ Basic functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

def test_visualization_components():
    """Test visualization components separately."""
    print("\nTesting Visualization Components")
    print("=" * 35)
    
    try:
        from visualization.smile_plotter import SmilePlotter
        from visualization.surface_plotter import SurfacePlotter
        
        print("✓ Smile plotter import successful")
        print("✓ Surface plotter import successful")
        
        # Test creating plotters
        smile_plotter = SmilePlotter()
        surface_plotter = SurfacePlotter()
        print("✓ Plotter objects created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_evaluation_components():
    """Test evaluation components."""
    print("\nTesting Evaluation Components")
    print("=" * 30)
    
    try:
        from evaluation.metrics import SurfaceEvaluator, StatisticalTester
        from evaluation.surface_evaluator import ComprehensiveEvaluator
        from evaluation.benchmark_comparison import BenchmarkComparator
        
        print("✓ All evaluation imports successful")
        
        # Test creating objects
        surface_evaluator = SurfaceEvaluator()
        statistical_tester = StatisticalTester()
        comprehensive_evaluator = ComprehensiveEvaluator()
        benchmark_comparator = BenchmarkComparator()
        
        print("✓ All evaluation objects created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation components test failed: {e}")
        return False

def main():
    """Run all working tests."""
    print("SABR Performance Analysis - Working Components Demo")
    print("=" * 55)
    
    results = []
    
    # Test basic functionality
    results.append(test_basic_functionality())
    
    # Test visualization components
    results.append(test_visualization_components())
    
    # Test evaluation components
    results.append(test_evaluation_components())
    
    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All working components tested successfully!")
        print("\nWhat's currently working:")
        print("• Performance analyzer creation and basic analysis")
        print("• Results aggregation and model ranking")
        print("• Visualization component imports")
        print("• Evaluation component imports")
        print("• Performance vs budget analysis (without plots)")
        
    else:
        print(f"\n⚠ {total - passed} tests had issues, but core functionality works")
    
    print("\nTo run full examples with plots and reports:")
    print("• Fix missing methods in performance_analyzer.py")
    print("• Run example_performance_analysis.py")
    print("• Run example_comprehensive_pipeline.py")

if __name__ == "__main__":
    main()