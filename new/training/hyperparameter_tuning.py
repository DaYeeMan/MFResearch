"""
Hyperparameter tuning utilities for SABR MDA-CNN models.

This module provides various hyperparameter optimization strategies:
- Random search
- Grid search
- Bayesian optimization (using Optuna)
- Evolutionary algorithms
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Bayesian optimization will not work.")

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ExperimentConfig, ModelConfig, TrainingConfig
from utils.logging_utils import get_logger
from utils.reproducibility import set_random_seed

logger = get_logger(__name__)


@dataclass
class HyperparameterBounds:
    """Define bounds for hyperparameter optimization."""
    # Training hyperparameters
    learning_rate_min: float = 1e-5
    learning_rate_max: float = 1e-2
    batch_size_choices: List[int] = None
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-3
    
    # Model architecture hyperparameters
    dropout_rate_min: float = 0.0
    dropout_rate_max: float = 0.5
    cnn_filters_choices: List[List[int]] = None
    mlp_hidden_dims_choices: List[List[int]] = None
    fusion_dims_choices: List[List[int]] = None
    
    # CNN specific
    cnn_kernel_size_choices: List[int] = None
    
    def __post_init__(self):
        if self.batch_size_choices is None:
            self.batch_size_choices = [16, 32, 64, 128, 256]
        
        if self.cnn_filters_choices is None:
            self.cnn_filters_choices = [
                [16, 32],
                [32, 64],
                [32, 64, 128],
                [64, 128, 256],
                [16, 32, 64, 128]
            ]
        
        if self.mlp_hidden_dims_choices is None:
            self.mlp_hidden_dims_choices = [
                [32],
                [64],
                [128],
                [32, 32],
                [64, 64],
                [128, 64],
                [128, 128],
                [256, 128],
                [128, 64, 32],
                [256, 128, 64]
            ]
        
        if self.fusion_dims_choices is None:
            self.fusion_dims_choices = [
                [64],
                [128],
                [64, 32],
                [128, 64],
                [256, 128],
                [128, 64, 32]
            ]
        
        if self.cnn_kernel_size_choices is None:
            self.cnn_kernel_size_choices = [3, 5]


class HyperparameterOptimizer:
    """Base class for hyperparameter optimization."""
    
    def __init__(
        self,
        bounds: HyperparameterBounds,
        objective_function: Callable,
        random_seed: int = 42
    ):
        """
        Initialize optimizer.
        
        Args:
            bounds: Hyperparameter bounds
            objective_function: Function to optimize (should return metric to minimize)
            random_seed: Random seed for reproducibility
        """
        self.bounds = bounds
        self.objective_function = objective_function
        self.random_seed = random_seed
        self.trials_history = []
    
    def optimize(self, n_trials: int) -> Dict[str, Any]:
        """
        Run optimization for specified number of trials.
        
        Args:
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters found
        """
        raise NotImplementedError("Subclasses must implement optimize method")
    
    def _evaluate_hyperparameters(self, hyperparameters: Dict[str, Any]) -> float:
        """Evaluate hyperparameters using objective function."""
        try:
            score = self.objective_function(hyperparameters)
            self.trials_history.append({
                'hyperparameters': hyperparameters.copy(),
                'score': score
            })
            return score
        except Exception as e:
            logger.error(f"Error evaluating hyperparameters {hyperparameters}: {e}")
            return float('inf')  # Return worst possible score on error


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search hyperparameter optimizer."""
    
    def optimize(self, n_trials: int) -> Dict[str, Any]:
        """Run random search optimization."""
        logger.info(f"Starting random search with {n_trials} trials")
        
        np.random.seed(self.random_seed)
        best_score = float('inf')
        best_params = None
        
        for trial in range(n_trials):
            # Sample random hyperparameters
            hyperparameters = self._sample_random_hyperparameters()
            
            # Evaluate
            score = self._evaluate_hyperparameters(hyperparameters)
            
            # Update best
            if score < best_score:
                best_score = score
                best_params = hyperparameters.copy()
            
            logger.info(f"Trial {trial + 1}/{n_trials}: score={score:.6f}, best={best_score:.6f}")
        
        logger.info(f"Random search completed. Best score: {best_score:.6f}")
        return best_params
    
    def _sample_random_hyperparameters(self) -> Dict[str, Any]:
        """Sample random hyperparameters within bounds."""
        return {
            'learning_rate': np.random.uniform(
                self.bounds.learning_rate_min, 
                self.bounds.learning_rate_max
            ),
            'batch_size': np.random.choice(self.bounds.batch_size_choices),
            'weight_decay': np.random.uniform(
                self.bounds.weight_decay_min,
                self.bounds.weight_decay_max
            ),
            'dropout_rate': np.random.uniform(
                self.bounds.dropout_rate_min,
                self.bounds.dropout_rate_max
            ),
            'cnn_filters': np.random.choice(self.bounds.cnn_filters_choices, axis=0).tolist(),
            'mlp_hidden_dims': np.random.choice(self.bounds.mlp_hidden_dims_choices, axis=0).tolist(),
            'fusion_dims': np.random.choice(self.bounds.fusion_dims_choices, axis=0).tolist(),
            'cnn_kernel_size': np.random.choice(self.bounds.cnn_kernel_size_choices)
        }


class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search hyperparameter optimizer."""
    
    def __init__(
        self,
        bounds: HyperparameterBounds,
        objective_function: Callable,
        grid_params: Dict[str, List[Any]],
        random_seed: int = 42
    ):
        """
        Initialize grid search optimizer.
        
        Args:
            bounds: Hyperparameter bounds (not used for grid search)
            objective_function: Objective function
            grid_params: Dictionary defining grid search space
            random_seed: Random seed
        """
        super().__init__(bounds, objective_function, random_seed)
        self.grid_params = grid_params
    
    def optimize(self, n_trials: int = None) -> Dict[str, Any]:
        """Run grid search optimization."""
        # Generate all combinations
        param_names = list(self.grid_params.keys())
        param_values = list(self.grid_params.values())
        
        from itertools import product
        all_combinations = list(product(*param_values))
        
        logger.info(f"Starting grid search with {len(all_combinations)} combinations")
        
        best_score = float('inf')
        best_params = None
        
        for i, combination in enumerate(all_combinations):
            # Create hyperparameter dictionary
            hyperparameters = dict(zip(param_names, combination))
            
            # Evaluate
            score = self._evaluate_hyperparameters(hyperparameters)
            
            # Update best
            if score < best_score:
                best_score = score
                best_params = hyperparameters.copy()
            
            logger.info(f"Combination {i + 1}/{len(all_combinations)}: score={score:.6f}, best={best_score:.6f}")
        
        logger.info(f"Grid search completed. Best score: {best_score:.6f}")
        return best_params


class BayesianOptimizer(HyperparameterOptimizer):
    """Bayesian optimization using Optuna."""
    
    def __init__(
        self,
        bounds: HyperparameterBounds,
        objective_function: Callable,
        random_seed: int = 42,
        sampler_type: str = "tpe"
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            bounds: Hyperparameter bounds
            objective_function: Objective function
            random_seed: Random seed
            sampler_type: Type of sampler ('tpe' or 'random')
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        super().__init__(bounds, objective_function, random_seed)
        
        # Create sampler
        if sampler_type == "tpe":
            self.sampler = TPESampler(seed=random_seed)
        elif sampler_type == "random":
            self.sampler = RandomSampler(seed=random_seed)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        # Create study
        self.study = optuna.create_study(
            direction="minimize",
            sampler=self.sampler
        )
    
    def optimize(self, n_trials: int) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        
        # Define objective function for Optuna
        def optuna_objective(trial):
            # Sample hyperparameters
            hyperparameters = {
                'learning_rate': trial.suggest_float(
                    'learning_rate',
                    self.bounds.learning_rate_min,
                    self.bounds.learning_rate_max,
                    log=True
                ),
                'batch_size': trial.suggest_categorical(
                    'batch_size',
                    self.bounds.batch_size_choices
                ),
                'weight_decay': trial.suggest_float(
                    'weight_decay',
                    self.bounds.weight_decay_min,
                    self.bounds.weight_decay_max,
                    log=True
                ),
                'dropout_rate': trial.suggest_float(
                    'dropout_rate',
                    self.bounds.dropout_rate_min,
                    self.bounds.dropout_rate_max
                ),
                'cnn_filters': trial.suggest_categorical(
                    'cnn_filters',
                    [str(f) for f in self.bounds.cnn_filters_choices]
                ),
                'mlp_hidden_dims': trial.suggest_categorical(
                    'mlp_hidden_dims',
                    [str(d) for d in self.bounds.mlp_hidden_dims_choices]
                ),
                'fusion_dims': trial.suggest_categorical(
                    'fusion_dims',
                    [str(d) for d in self.bounds.fusion_dims_choices]
                ),
                'cnn_kernel_size': trial.suggest_categorical(
                    'cnn_kernel_size',
                    self.bounds.cnn_kernel_size_choices
                )
            }
            
            # Convert string representations back to lists
            hyperparameters['cnn_filters'] = eval(hyperparameters['cnn_filters'])
            hyperparameters['mlp_hidden_dims'] = eval(hyperparameters['mlp_hidden_dims'])
            hyperparameters['fusion_dims'] = eval(hyperparameters['fusion_dims'])
            
            return self._evaluate_hyperparameters(hyperparameters)
        
        # Run optimization
        self.study.optimize(optuna_objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = self.study.best_params.copy()
        
        # Convert string representations back to lists
        best_params['cnn_filters'] = eval(best_params['cnn_filters'])
        best_params['mlp_hidden_dims'] = eval(best_params['mlp_hidden_dims'])
        best_params['fusion_dims'] = eval(best_params['fusion_dims'])
        
        logger.info(f"Bayesian optimization completed. Best score: {self.study.best_value:.6f}")
        return best_params


class HyperparameterTuner:
    """
    High-level interface for hyperparameter tuning.
    
    Provides easy-to-use methods for different optimization strategies
    and integrates with the experiment framework.
    """
    
    def __init__(
        self,
        base_config: ExperimentConfig,
        data_loaders: Dict[str, Any],
        bounds: Optional[HyperparameterBounds] = None,
        random_seed: int = 42
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            base_config: Base experiment configuration
            data_loaders: Data loaders for training/validation
            bounds: Hyperparameter bounds
            random_seed: Random seed
        """
        self.base_config = base_config
        self.data_loaders = data_loaders
        self.bounds = bounds or HyperparameterBounds()
        self.random_seed = random_seed
        
        # Import here to avoid circular imports
        from training.trainer import ModelTrainer
        from models.mda_cnn import create_mda_cnn_model
        
        self.ModelTrainer = ModelTrainer
        self.create_mda_cnn_model = create_mda_cnn_model
    
    def objective_function(self, hyperparameters: Dict[str, Any]) -> float:
        """
        Objective function for hyperparameter optimization.
        
        Args:
            hyperparameters: Hyperparameters to evaluate
            
        Returns:
            Validation loss (metric to minimize)
        """
        try:
            # Set random seeds
            set_random_seed(self.random_seed)
            
            # Create modified config
            config = self._create_config_with_hyperparameters(hyperparameters)
            
            # Create model
            model = self.create_mda_cnn_model(
                patch_size=tuple(config.model_config.patch_size),
                cnn_filters=tuple(config.model_config.cnn_filters),
                mlp_hidden_dims=tuple(config.model_config.mlp_hidden_dims),
                fusion_hidden_dims=tuple(config.model_config.fusion_dims),
                dropout_rate=config.model_config.dropout_rate,
                cnn_kernel_size=config.model_config.cnn_kernel_size
            )
            
            # Create trainer
            trainer = self.ModelTrainer(config)
            
            # Train model (with reduced epochs for faster tuning)
            history = trainer.train(
                model=model,
                train_dataset=self.data_loaders['train'],
                validation_dataset=self.data_loaders['val']
            )
            
            # Return best validation loss
            return min(history.history['val_loss'])
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return float('inf')
    
    def _create_config_with_hyperparameters(
        self,
        hyperparameters: Dict[str, Any]
    ) -> ExperimentConfig:
        """Create experiment config with specified hyperparameters."""
        # Copy base config
        config = ExperimentConfig(
            name=self.base_config.name,
            sabr_params=self.base_config.sabr_params,
            grid_config=self.base_config.grid_config,
            data_gen_config=self.base_config.data_gen_config,
            model_config=ModelConfig(
                patch_size=self.base_config.model_config.patch_size,
                cnn_filters=hyperparameters.get('cnn_filters', self.base_config.model_config.cnn_filters),
                cnn_kernel_size=hyperparameters.get('cnn_kernel_size', self.base_config.model_config.cnn_kernel_size),
                mlp_hidden_dims=hyperparameters.get('mlp_hidden_dims', self.base_config.model_config.mlp_hidden_dims),
                fusion_dims=hyperparameters.get('fusion_dims', self.base_config.model_config.fusion_dims),
                dropout_rate=hyperparameters.get('dropout_rate', self.base_config.model_config.dropout_rate),
                activation=self.base_config.model_config.activation
            ),
            training_config=TrainingConfig(
                batch_size=hyperparameters.get('batch_size', self.base_config.training_config.batch_size),
                epochs=min(50, self.base_config.training_config.epochs),  # Reduce epochs for tuning
                learning_rate=hyperparameters.get('learning_rate', self.base_config.training_config.learning_rate),
                weight_decay=hyperparameters.get('weight_decay', self.base_config.training_config.weight_decay),
                early_stopping_patience=10,  # Reduce patience for faster tuning
                lr_scheduler_patience=5,
                lr_scheduler_factor=self.base_config.training_config.lr_scheduler_factor,
                gradient_clip_norm=self.base_config.training_config.gradient_clip_norm,
                save_best_only=self.base_config.training_config.save_best_only
            ),
            output_dir=self.base_config.output_dir
        )
        
        return config
    
    def tune_hyperparameters(
        self,
        method: str = "random",
        n_trials: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using specified method.
        
        Args:
            method: Optimization method ('random', 'grid', 'bayesian')
            n_trials: Number of trials
            **kwargs: Additional arguments for optimizer
            
        Returns:
            Best hyperparameters found
        """
        logger.info(f"Starting hyperparameter tuning with {method} search")
        
        if method == "random":
            optimizer = RandomSearchOptimizer(
                bounds=self.bounds,
                objective_function=self.objective_function,
                random_seed=self.random_seed
            )
        elif method == "grid":
            grid_params = kwargs.get('grid_params')
            if grid_params is None:
                raise ValueError("Grid search requires 'grid_params' argument")
            
            optimizer = GridSearchOptimizer(
                bounds=self.bounds,
                objective_function=self.objective_function,
                grid_params=grid_params,
                random_seed=self.random_seed
            )
        elif method == "bayesian":
            optimizer = BayesianOptimizer(
                bounds=self.bounds,
                objective_function=self.objective_function,
                random_seed=self.random_seed,
                sampler_type=kwargs.get('sampler_type', 'tpe')
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Run optimization
        best_params = optimizer.optimize(n_trials)
        
        # Save tuning history
        self._save_tuning_results(optimizer, best_params, method)
        
        return best_params
    
    def _save_tuning_results(
        self,
        optimizer: HyperparameterOptimizer,
        best_params: Dict[str, Any],
        method: str
    ):
        """Save hyperparameter tuning results."""
        results_dir = Path(self.base_config.output_dir) / "hyperparameter_tuning"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        best_params_path = results_dir / f"best_params_{method}.json"
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2, default=str)
        
        # Save trials history
        trials_df = pd.DataFrame([
            {**trial['hyperparameters'], 'score': trial['score']}
            for trial in optimizer.trials_history
        ])
        
        trials_path = results_dir / f"trials_history_{method}.csv"
        trials_df.to_csv(trials_path, index=False)
        
        logger.info(f"Hyperparameter tuning results saved to {results_dir}")


def create_hyperparameter_tuner(
    base_config: ExperimentConfig,
    data_loaders: Dict[str, Any],
    **kwargs
) -> HyperparameterTuner:
    """
    Factory function to create hyperparameter tuner.
    
    Args:
        base_config: Base experiment configuration
        data_loaders: Data loaders
        **kwargs: Additional arguments
        
    Returns:
        HyperparameterTuner instance
    """
    return HyperparameterTuner(
        base_config=base_config,
        data_loaders=data_loaders,
        **kwargs
    )