#!/usr/bin/env python3
"""
Test script to verify the project setup is working correctly.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from new.utils.config import ConfigManager, ExperimentConfig
from new.utils.logging_utils import setup_logging, get_logger
from new.utils.reproducibility import set_random_seed, validate_reproducibility
from new.utils.common import setup_experiment, Timer
import numpy as np


def test_config_system():
    """Test configuration management system."""
    print("Testing configuration system...")
    
    # Test default config creation
    config_manager = ConfigManager("new/configs")
    default_config = config_manager.create_default_config("test_experiment")
    
    # Test config validation
    assert config_manager.validate_config(default_config)
    
    # Test config saving and loading
    test_config_path = "new/configs/test_setup.yaml"
    config_manager.save_config(default_config, test_config_path)
    loaded_config = config_manager.load_config(test_config_path)
    
    assert loaded_config.name == default_config.name
    print("✓ Configuration system working correctly")


def test_logging_system():
    """Test logging utilities."""
    print("Testing logging system...")
    
    # Set up logging
    logger = setup_logging(
        log_level="INFO",
        log_dir="new/test_logs",
        experiment_name="test_setup",
        console_output=False  # Disable console for test
    )
    
    # Test logging
    logger.info("Test info message")
    logger.warning("Test warning message")
    
    print("✓ Logging system working correctly")


def test_reproducibility():
    """Test reproducibility utilities."""
    print("Testing reproducibility system...")
    
    # Test seed setting
    set_random_seed(42)
    
    # Test reproducible function
    def random_function():
        return np.random.random(10)
    
    # Validate reproducibility
    is_reproducible = validate_reproducibility(random_function, seed=42, n_runs=3)
    assert is_reproducible
    
    print("✓ Reproducibility system working correctly")


def test_complete_setup():
    """Test complete experiment setup."""
    print("Testing complete experiment setup...")
    
    with Timer("Complete setup test"):
        config, exp_logger = setup_experiment(
            "new/configs/test_config.yaml",
            experiment_name="test_complete_setup",
            log_level="INFO"
        )
        
        # Test that everything is properly configured
        assert config.name == "test_complete_setup"
        assert config.data_gen_config.random_seed == 42
        
        # Log some test metrics
        exp_logger.log_epoch(1, {"loss": 0.5, "accuracy": 0.8})
        exp_logger.log_experiment_end({"final_loss": 0.1, "final_accuracy": 0.95})
    
    print("✓ Complete experiment setup working correctly")


def main():
    """Run all tests."""
    print("Running SABR MDA-CNN project setup tests...\n")
    
    try:
        test_config_system()
        test_logging_system()
        test_reproducibility()
        test_complete_setup()
        
        print("\n🎉 All tests passed! Project setup is working correctly.")
        print("\nNext steps:")
        print("1. Install required dependencies (numpy, pandas, matplotlib, etc.)")
        print("2. Install TensorFlow or PyTorch for model training")
        print("3. Start implementing the data generation components")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()