"""
Common utilities and helper functions for SABR MDA-CNN project.
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json

from .config import ConfigManager, ExperimentConfig
from .logging_utils import setup_logging, get_logger, ExperimentLogger
from .reproducibility import set_random_seed, get_seed_manager

logger = get_logger(__name__)


def setup_experiment(
    config_path: Union[str, Path],
    experiment_name: Optional[str] = None,
    log_level: str = "INFO",
    deterministic_gpu: bool = True
) -> tuple[ExperimentConfig, ExperimentLogger]:
    """
    Set up a complete experiment with configuration, logging, and reproducibility.
    
    Args:
        config_path: Path to configuration file
        experiment_name: Override experiment name from config
        log_level: Logging level
        deterministic_gpu: Enable deterministic GPU operations
        
    Returns:
        Tuple of (config, experiment_logger)
    """
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(config_path)
    
    # Override experiment name if provided
    if experiment_name:
        config.name = experiment_name
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_dir = output_dir / "logs"
    setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        experiment_name=config.name,
        console_output=True,
        json_format=True
    )
    
    # Set up reproducibility
    set_random_seed(config.data_gen_config.random_seed, deterministic_gpu)
    
    # Create experiment logger
    exp_logger = ExperimentLogger(config.name, log_dir)
    
    # Log experiment setup
    exp_logger.log_experiment_start(config_manager._config_to_dict(config))
    
    logger.info(f"Experiment '{config.name}' set up successfully")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {config.data_gen_config.random_seed}")
    
    return config, exp_logger


def save_results(results: Dict[str, Any], output_path: Union[str, Path]):
    """
    Save experiment results to file.
    
    Args:
        results: Dictionary of results to save
        output_path: Path to save results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to results
    results['timestamp'] = datetime.now().isoformat()
    
    # Save as JSON for readability
    if output_path.suffix.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        # Save as pickle for complex objects
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    
    logger.info(f"Results saved to {output_path}")


def load_results(results_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment results from file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Dictionary of loaded results
    """
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    if results_path.suffix.lower() == '.json':
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    
    logger.info(f"Results loaded from {results_path}")
    return results


def create_experiment_directory(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create a timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assume this file is in new/utils/common.py
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "new" / "data"


def get_config_dir() -> Path:
    """Get the configuration directory."""
    return get_project_root() / "new" / "configs"


def get_results_dir() -> Path:
    """Get the results directory."""
    return get_project_root() / "new" / "results"


def format_number(num: float, precision: int = 4) -> str:
    """Format number for display with appropriate precision."""
    if abs(num) < 1e-3:
        return f"{num:.2e}"
    else:
        return f"{num:.{precision}f}"


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def memory_usage() -> str:
    """Get current memory usage information."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    except ImportError:
        return "N/A (psutil not available)"


def gpu_memory_usage() -> str:
    """Get GPU memory usage information."""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # This is a simplified version - actual implementation would need more work
            return f"{len(gpus)} GPU(s) available"
        else:
            return "No GPU available"
    except ImportError:
        return "N/A (TensorFlow not available)"


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"{self.name} started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {format_time(duration)}")
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def validate_environment():
    """Validate that the environment has required dependencies."""
    required_packages = ['numpy', 'pandas', 'matplotlib', 'scipy']
    optional_packages = ['tensorflow', 'torch', 'psutil', 'seaborn']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        raise ImportError(f"Please install required packages: {missing_required}")
    
    if missing_optional:
        logger.warning(f"Missing optional packages: {missing_optional}")
        logger.warning("Some functionality may be limited")
    
    logger.info("Environment validation completed successfully")


# Validate environment on import
try:
    validate_environment()
except ImportError as e:
    logger.error(f"Environment validation failed: {e}")
    # Don't raise here to allow partial functionality