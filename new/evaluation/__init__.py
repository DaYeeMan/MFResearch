"""
Comprehensive evaluation module for SABR volatility surface models.

This module provides surface-specific metrics, statistical testing,
benchmark comparison, and evaluation pipeline functionality.
"""

from .metrics import (
    SurfaceEvaluator,
    SurfaceMetrics,
    StatisticalTester,
    RegionBounds,
    compute_relative_improvement
)

from .surface_evaluator import (
    ComprehensiveEvaluator,
    ModelPrediction,
    EvaluationResult
)

from .benchmark_comparison import (
    BenchmarkComparator,
    BenchmarkResult,
    create_performance_summary
)

__all__ = [
    # Core metrics and evaluation
    'SurfaceEvaluator',
    'SurfaceMetrics',
    'StatisticalTester',
    'RegionBounds',
    'compute_relative_improvement',
    
    # Comprehensive evaluation pipeline
    'ComprehensiveEvaluator',
    'ModelPrediction',
    'EvaluationResult',
    
    # Benchmark comparison
    'BenchmarkComparator',
    'BenchmarkResult',
    'create_performance_summary'
]