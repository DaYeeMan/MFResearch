#!/usr/bin/env python3
"""Simple test script for Hagan surface generator."""

from hagan_surface_generator import HaganSurfaceGenerator
from sabr_params import create_test_sabr_params, create_default_grid_config

def main():
    print("Testing Hagan Surface Generator")
    print("=" * 40)
    
    # Create generator
    gen = HaganSurfaceGenerator()
    
    # Test basic surface generation
    params = create_test_sabr_params()
    grid = create_default_grid_config()
    
    print(f"SABR Parameters: {params}")
    print(f"Grid Config: {grid}")
    
    # Generate surface
    surface = gen.generate_surface(params, grid)
    print(f"Generated surface shape: {surface.shape}")
    print(f"Surface stats: min={surface.min():.4f}, max={surface.max():.4f}, mean={surface.mean():.4f}")
    
    # Run benchmark tests
    print("\nRunning benchmark tests...")
    results = gen.benchmark_against_literature()
    
    print(f"Benchmark Results:")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Max Error: {results['max_error']:.6f}")
    
    print("\nDetailed Results:")
    for tc in results['test_cases']:
        status = "PASS" if tc['passed'] else "FAIL"
        print(f"  {tc['name']}: {status} (error: {tc['error']:.6f})")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()