"""
Reproducibility utilities for SABR MDA-CNN project.
Handles random seed management and deterministic behavior setup.
"""

import random
import numpy as np
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import TensorFlow/Keras for GPU determinism
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. GPU determinism will not be set.")

# Try to import PyTorch for additional reproducibility
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_random_seed(seed: int = 42, deterministic_gpu: bool = True):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value
        deterministic_gpu: Whether to enable deterministic GPU operations
                          (may reduce performance)
    """
    logger.info(f"Setting random seed to {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow/Keras
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
        
        if deterministic_gpu:
            # Enable deterministic operations (may reduce performance)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            
            # Configure GPU memory growth to avoid non-deterministic behavior
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Configured {len(gpus)} GPU(s) for deterministic behavior")
                except RuntimeError as e:
                    logger.warning(f"Failed to configure GPU memory growth: {e}")
    
    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic_gpu:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    logger.info("Random seed configuration completed")


def get_random_state() -> dict:
    """
    Get current random state from all libraries.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
    }
    
    if TF_AVAILABLE:
        # TensorFlow state handling is complex with determinism enabled
        # We'll skip state capture for now
        try:
            state['tensorflow_seed'] = tf.random.get_global_generator().state
        except RuntimeError:
            # Skip TF state if determinism is enabled
            state['tensorflow_seed'] = None
    
    if TORCH_AVAILABLE:
        state['torch_random'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['torch_cuda_random'] = torch.cuda.get_rng_state()
    
    return state


def set_random_state(state: dict):
    """
    Restore random state for all libraries.
    
    Args:
        state: Dictionary containing random states from get_random_state()
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])
    
    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])
    
    if TF_AVAILABLE and 'tensorflow_seed' in state and state['tensorflow_seed'] is not None:
        try:
            tf.random.get_global_generator().reset_from_seed_state(state['tensorflow_seed'])
        except RuntimeError:
            # Skip TF state restoration if determinism is enabled
            pass
    
    if TORCH_AVAILABLE:
        if 'torch_random' in state:
            torch.set_rng_state(state['torch_random'])
        if 'torch_cuda_random' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['torch_cuda_random'])


class ReproducibleContext:
    """Context manager for reproducible code execution."""
    
    def __init__(self, seed: int = 42, deterministic_gpu: bool = True):
        self.seed = seed
        self.deterministic_gpu = deterministic_gpu
        self.original_state = None
    
    def __enter__(self):
        # Save current state
        self.original_state = get_random_state()
        
        # Set reproducible seed
        set_random_seed(self.seed, self.deterministic_gpu)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        if self.original_state:
            set_random_state(self.original_state)


def create_reproducible_generator(seed: int = 42):
    """
    Create a NumPy random generator with fixed seed.
    
    Args:
        seed: Random seed
        
    Returns:
        NumPy random generator
    """
    return np.random.Generator(np.random.PCG64(seed))


def validate_reproducibility(func, seed: int = 42, n_runs: int = 3):
    """
    Validate that a function produces reproducible results.
    
    Args:
        func: Function to test (should take no arguments)
        seed: Random seed to use
        n_runs: Number of runs to compare
        
    Returns:
        bool: True if all runs produce identical results
    """
    results = []
    
    for i in range(n_runs):
        with ReproducibleContext(seed):
            result = func()
            results.append(result)
    
    # Check if all results are identical
    first_result = results[0]
    
    for i, result in enumerate(results[1:], 1):
        if isinstance(first_result, np.ndarray):
            if not np.array_equal(first_result, result):
                logger.error(f"Run {i+1} differs from run 1")
                return False
        else:
            if first_result != result:
                logger.error(f"Run {i+1} differs from run 1")
                return False
    
    logger.info(f"Function is reproducible across {n_runs} runs")
    return True


class SeedManager:
    """Manages random seeds for different components of the experiment."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.component_seeds = {}
        self._generate_component_seeds()
    
    def _generate_component_seeds(self):
        """Generate deterministic seeds for different components."""
        # Use a separate generator to create component seeds
        generator = create_reproducible_generator(self.base_seed)
        
        self.component_seeds = {
            'data_generation': int(generator.integers(0, 2**31)),
            'data_splitting': int(generator.integers(0, 2**31)),
            'model_initialization': int(generator.integers(0, 2**31)),
            'training': int(generator.integers(0, 2**31)),
            'evaluation': int(generator.integers(0, 2**31)),
            'visualization': int(generator.integers(0, 2**31))
        }
        
        logger.info(f"Generated component seeds: {self.component_seeds}")
    
    def get_seed(self, component: str) -> int:
        """Get seed for a specific component."""
        if component not in self.component_seeds:
            raise ValueError(f"Unknown component: {component}")
        return self.component_seeds[component]
    
    def set_component_seed(self, component: str):
        """Set random seed for a specific component."""
        seed = self.get_seed(component)
        set_random_seed(seed)
        logger.debug(f"Set seed {seed} for component '{component}'")
    
    def get_all_seeds(self) -> dict:
        """Get all component seeds."""
        return self.component_seeds.copy()


# Global seed manager instance
_global_seed_manager = None


def get_seed_manager(base_seed: Optional[int] = None) -> SeedManager:
    """Get global seed manager instance."""
    global _global_seed_manager
    
    if _global_seed_manager is None or base_seed is not None:
        seed = base_seed if base_seed is not None else 42
        _global_seed_manager = SeedManager(seed)
    
    return _global_seed_manager


def reproducible_function(component: str):
    """Decorator to make a function reproducible using component-specific seed."""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            seed_manager = get_seed_manager()
            with ReproducibleContext(seed_manager.get_seed(component)):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator