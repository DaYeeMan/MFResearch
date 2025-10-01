"""
Experiment orchestrator for HF budget analysis.

This module implements a comprehensive experiment runner that:
- Tests multiple HF budget sizes
- Compares different model architectures
- Performs automated hyperparameter tuning
- Aggregates results and provides statistical analysis
- Ensures experiment reproducibility
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ExperimentConfig, DataGenConfig, ModelConfig, TrainingConfig
from utils.logging_utils import get_logger, ExperimentLogger
from utils.reproducibility import set_random_seed
from training.trainer import ModelTrainer
from models.mda_cnn import create_mda_cnn_model
from models.baseline_models import create_baseline_model
from evaluation.metrics import SurfaceEvaluator, StatisticalTester
# from preprocessing.data_loader import create_data_loaders  # Import when needed
# from data_generation.data_orchestrator import DataOrchestrator  # Import when needed

logger = get_logger(__name__)


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space for tuning."""
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    dropout_rates: List[float] = None
    cnn_filters: List[List[int]] = None
    mlp_hidden_dims: List[List[int]] = None
    
    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [1e-4, 3e-4, 1e-3]
        if self.batch_sizes is None:
            self.batch_sizes = [32, 64, 128]
        if self.dropout_rates is None:
            self.dropout_rates = [0.1, 0.2, 0.3]
        if self.cnn_filters is None:
            self.cnn_filters = [[32, 64], [32, 64, 128], [64, 128, 256]]
        if self.mlp_hidden_dims is None:
            self.mlp_hidden_dims = [[64, 64], [128, 64], [128, 128, 64]]


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    experiment_id: str
    model_type: str
    hf_budget: int
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    model_path: str
    config_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentOrchestrator:
    """
    Main orchestrator for HF budget analysis experiments.
    
    Manages the complete experimental pipeline including:
    - Data generation for different HF budgets
    - Model training with hyperparameter tuning
    - Cross-architecture comparison
    - Statistical analysis and result aggregation
    """
    
    def __init__(
        self,
        base_config: ExperimentConfig,
        output_dir: str = "new/results/hf_budget_analysis",
        n_parallel_jobs: int = 1,
        random_seed: int = 42
    ):
        """
        Initialize experiment orchestrator.
        
        Args:
            base_config: Base experiment configuration
            output_dir: Directory to save all experiment outputs
            n_parallel_jobs: Number of parallel jobs for experiments
            random_seed: Random seed for reproducibility
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_parallel_jobs = n_parallel_jobs
        self.random_seed = random_seed
        
        # Set up experiment logger
        self.experiment_logger = ExperimentLogger(
            experiment_name="hf_budget_analysis",
            log_dir=self.output_dir / "logs"
        )
        
        # Initialize components
        self.surface_evaluator = SurfaceEvaluator()
        self.statistical_tester = StatisticalTester()
        
        # Experiment tracking
        self.results: List[ExperimentResult] = []
        self.experiment_counter = 0
        
        logger.info(f"Experiment orchestrator initialized with output dir: {self.output_dir}")
    
    def run_hf_budget_analysis(
        self,
        hf_budgets: List[int],
        model_types: List[str],
        hyperparameter_space: Optional[HyperparameterSpace] = None,
        n_hyperparameter_trials: int = 5,
        n_random_seeds: int = 3
    ) -> pd.DataFrame:
        """
        Run comprehensive HF budget analysis across multiple architectures.
        
        Args:
            hf_budgets: List of HF budget sizes to test
            model_types: List of model types to compare
            hyperparameter_space: Hyperparameter search space
            n_hyperparameter_trials: Number of hyperparameter combinations to try
            n_random_seeds: Number of random seeds for statistical robustness
            
        Returns:
            DataFrame with aggregated results
        """
        logger.info("Starting HF budget analysis")
        logger.info(f"HF budgets: {hf_budgets}")
        logger.info(f"Model types: {model_types}")
        logger.info(f"Hyperparameter trials: {n_hyperparameter_trials}")
        logger.info(f"Random seeds: {n_random_seeds}")
        
        # Initialize hyperparameter space
        if hyperparameter_space is None:
            hyperparameter_space = HyperparameterSpace()
        
        # Generate all experiment combinations
        experiment_configs = self._generate_experiment_configs(
            hf_budgets, model_types, hyperparameter_space, 
            n_hyperparameter_trials, n_random_seeds
        )
        
        logger.info(f"Generated {len(experiment_configs)} experiment configurations")
        
        # Run experiments
        if self.n_parallel_jobs > 1:
            results = self._run_experiments_parallel(experiment_configs)
        else:
            results = self._run_experiments_sequential(experiment_configs)
        
        # Aggregate and analyze results
        results_df = self._aggregate_results(results)
        
        # Perform statistical analysis
        statistical_results = self._perform_statistical_analysis(results_df)
        
        # Save results
        self._save_results(results_df, statistical_results)
        
        logger.info("HF budget analysis completed")
        return results_df
    
    def _generate_experiment_configs(
        self,
        hf_budgets: List[int],
        model_types: List[str],
        hyperparameter_space: HyperparameterSpace,
        n_hyperparameter_trials: int,
        n_random_seeds: int
    ) -> List[Dict[str, Any]]:
        """Generate all experiment configurations."""
        configs = []
        
        # Sample hyperparameter combinations
        hyperparameter_combinations = self._sample_hyperparameters(
            hyperparameter_space, n_hyperparameter_trials
        )
        
        for hf_budget in hf_budgets:
            for model_type in model_types:
                for hp_combo in hyperparameter_combinations:
                    for seed in range(n_random_seeds):
                        config = {
                            'experiment_id': f"exp_{self.experiment_counter:04d}",
                            'hf_budget': hf_budget,
                            'model_type': model_type,
                            'hyperparameters': hp_combo,
                            'random_seed': self.random_seed + seed,
                            'trial_id': seed
                        }
                        configs.append(config)
                        self.experiment_counter += 1
        
        return configs
    
    def _sample_hyperparameters(
        self,
        hyperparameter_space: HyperparameterSpace,
        n_trials: int
    ) -> List[Dict[str, Any]]:
        """Sample hyperparameter combinations."""
        # For simplicity, use random sampling
        # In practice, could use more sophisticated methods like Bayesian optimization
        
        combinations = []
        np.random.seed(self.random_seed)
        
        for _ in range(n_trials):
            combo = {
                'learning_rate': np.random.choice(hyperparameter_space.learning_rates),
                'batch_size': np.random.choice(hyperparameter_space.batch_sizes),
                'dropout_rate': np.random.choice(hyperparameter_space.dropout_rates),
                'cnn_filters': np.random.choice(hyperparameter_space.cnn_filters, axis=0).tolist(),
                'mlp_hidden_dims': np.random.choice(hyperparameter_space.mlp_hidden_dims, axis=0).tolist()
            }
            combinations.append(combo)
        
        return combinations
    
    def _run_experiments_sequential(
        self,
        experiment_configs: List[Dict[str, Any]]
    ) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        
        for i, config in enumerate(experiment_configs):
            logger.info(f"Running experiment {i+1}/{len(experiment_configs)}: {config['experiment_id']}")
            
            try:
                result = self._run_single_experiment(config)
                results.append(result)
                logger.info(f"Experiment {config['experiment_id']} completed successfully")
            except Exception as e:
                logger.error(f"Experiment {config['experiment_id']} failed: {e}")
                continue
        
        return results
    
    def _run_experiments_parallel(
        self,
        experiment_configs: List[Dict[str, Any]]
    ) -> List[ExperimentResult]:
        """Run experiments in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_parallel_jobs) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self._run_single_experiment, config): config
                for config in experiment_configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Experiment {config['experiment_id']} completed successfully")
                except Exception as e:
                    logger.error(f"Experiment {config['experiment_id']} failed: {e}")
                    continue
        
        return results
    
    def _run_single_experiment(self, config: Dict[str, Any]) -> ExperimentResult:
        """Run a single experiment with given configuration."""
        experiment_id = config['experiment_id']
        hf_budget = config['hf_budget']
        model_type = config['model_type']
        hyperparameters = config['hyperparameters']
        random_seed = config['random_seed']
        
        # Set random seeds for reproducibility
        set_random_seed(random_seed)
        
        # Create experiment-specific output directory
        exp_output_dir = self.output_dir / experiment_id
        exp_output_dir.mkdir(exist_ok=True)
        
        # Create experiment configuration
        exp_config = self._create_experiment_config(
            hf_budget, model_type, hyperparameters, exp_output_dir
        )
        
        # Save experiment configuration
        config_path = exp_output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(exp_config.__dict__, f, indent=2, default=str)
        
        start_time = time.time()
        
        try:
            # Generate or load data
            data_loaders = self._prepare_data(exp_config, hf_budget)
            
            # Create and train model
            model = self._create_model(model_type, exp_config.model_config)
            trainer = ModelTrainer(exp_config, str(exp_output_dir))
            
            # Train model
            history = trainer.train(
                model=model,
                train_dataset=data_loaders['train'],
                validation_dataset=data_loaders['val']
            )
            
            # Evaluate model
            test_metrics = trainer.evaluate(
                model=model,
                test_dataset=data_loaders['test'],
                use_best_model=True
            )
            
            training_time = time.time() - start_time
            
            # Create result object
            result = ExperimentResult(
                experiment_id=experiment_id,
                model_type=model_type,
                hf_budget=hf_budget,
                hyperparameters=hyperparameters,
                metrics=test_metrics,
                training_time=training_time,
                model_path=str(trainer.best_model_path or ""),
                config_path=str(config_path)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in experiment {experiment_id}: {e}")
            raise
    
    def _create_experiment_config(
        self,
        hf_budget: int,
        model_type: str,
        hyperparameters: Dict[str, Any],
        output_dir: Path
    ) -> ExperimentConfig:
        """Create experiment configuration for a single run."""
        # Copy base config
        config = ExperimentConfig(
            name=f"hf_budget_{hf_budget}_{model_type}",
            sabr_params=self.base_config.sabr_params,
            grid_config=self.base_config.grid_config,
            data_gen_config=self.base_config.data_gen_config,
            model_config=self.base_config.model_config,
            training_config=self.base_config.training_config,
            output_dir=str(output_dir)
        )
        
        # Update HF budget
        config.data_gen_config.hf_budget = hf_budget
        
        # Update hyperparameters
        config.training_config.learning_rate = hyperparameters['learning_rate']
        config.training_config.batch_size = hyperparameters['batch_size']
        config.model_config.dropout_rate = hyperparameters['dropout_rate']
        config.model_config.cnn_filters = hyperparameters['cnn_filters']
        config.model_config.mlp_hidden_dims = hyperparameters['mlp_hidden_dims']
        
        return config
    
    def _prepare_data(
        self,
        config: ExperimentConfig,
        hf_budget: int
    ) -> Dict[str, Any]:
        """Prepare data loaders for the experiment."""
        # Check if data already exists for this HF budget
        data_dir = Path("new/data/processed") / f"hf_budget_{hf_budget}"
        
        if not data_dir.exists():
            logger.info(f"Generating data for HF budget {hf_budget}")
            
            # Generate data using data orchestrator (import here to avoid circular imports)
            from data_generation.data_orchestrator import DataOrchestrator
            data_orchestrator = DataOrchestrator(config)
            data_orchestrator.generate_training_data()
        
        # Create data loaders (import here to avoid circular imports)
        from preprocessing.data_loader import create_data_loaders
        data_loaders = create_data_loaders(
            data_dir=str(data_dir),
            batch_size=config.training_config.batch_size,
            validation_split=config.data_gen_config.validation_split,
            test_split=config.data_gen_config.test_split,
            random_seed=config.data_gen_config.random_seed
        )
        
        return data_loaders
    
    def _create_model(self, model_type: str, model_config: ModelConfig):
        """Create model based on type and configuration."""
        if model_type == 'mda_cnn':
            return create_mda_cnn_model(
                patch_size=tuple(model_config.patch_size),
                cnn_filters=tuple(model_config.cnn_filters),
                mlp_hidden_dims=tuple(model_config.mlp_hidden_dims),
                fusion_hidden_dims=tuple(model_config.fusion_dims),
                dropout_rate=model_config.dropout_rate,
                activation=model_config.activation
            )
        else:
            return create_baseline_model(
                model_type=model_type,
                dropout_rate=model_config.dropout_rate,
                activation=model_config.activation
            )
    
    def _aggregate_results(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Aggregate experiment results into a DataFrame."""
        data = []
        
        for result in results:
            row = {
                'experiment_id': result.experiment_id,
                'model_type': result.model_type,
                'hf_budget': result.hf_budget,
                'training_time': result.training_time,
                **result.hyperparameters,
                **result.metrics
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add summary statistics
        summary_stats = df.groupby(['model_type', 'hf_budget']).agg({
            'loss': ['mean', 'std', 'min', 'max'],
            'mse': ['mean', 'std', 'min', 'max'],
            'mae': ['mean', 'std', 'min', 'max'],
            'training_time': ['mean', 'std']
        }).round(6)
        
        return df
    
    def _perform_statistical_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on results."""
        statistical_results = {}
        
        # Compare models at each HF budget
        for hf_budget in results_df['hf_budget'].unique():
            budget_data = results_df[results_df['hf_budget'] == hf_budget]
            
            # Compare MDA-CNN vs baselines
            if 'mda_cnn' in budget_data['model_type'].values:
                mda_cnn_errors = budget_data[budget_data['model_type'] == 'mda_cnn']['mse'].values
                
                for model_type in budget_data['model_type'].unique():
                    if model_type != 'mda_cnn':
                        baseline_errors = budget_data[budget_data['model_type'] == model_type]['mse'].values
                        
                        if len(mda_cnn_errors) > 1 and len(baseline_errors) > 1:
                            # Perform paired t-test
                            t_test_result = self.statistical_tester.paired_t_test(
                                baseline_errors, mda_cnn_errors
                            )
                            
                            statistical_results[f'hf_{hf_budget}_{model_type}_vs_mda_cnn'] = t_test_result
        
        return statistical_results
    
    def _save_results(self, results_df: pd.DataFrame, statistical_results: Dict[str, Any]):
        """Save experiment results and analysis."""
        # Save detailed results
        results_path = self.output_dir / "detailed_results.csv"
        results_df.to_csv(results_path, index=False)
        
        # Save summary statistics
        summary_stats = results_df.groupby(['model_type', 'hf_budget']).agg({
            'loss': ['mean', 'std', 'count'],
            'mse': ['mean', 'std', 'count'],
            'mae': ['mean', 'std', 'count'],
            'training_time': ['mean', 'std']
        }).round(6)
        
        summary_path = self.output_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_path)
        
        # Save statistical analysis
        stats_path = self.output_dir / "statistical_analysis.json"
        with open(stats_path, 'w') as f:
            json.dump(statistical_results, f, indent=2, default=str)
        
        # Create performance vs budget plot data
        plot_data = results_df.groupby(['model_type', 'hf_budget'])['mse'].agg(['mean', 'std']).reset_index()
        plot_data_path = self.output_dir / "performance_vs_budget.csv"
        plot_data.to_csv(plot_data_path, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")


def create_experiment_orchestrator(
    base_config: ExperimentConfig,
    output_dir: str = "new/results/hf_budget_analysis",
    **kwargs
) -> ExperimentOrchestrator:
    """
    Factory function to create experiment orchestrator.
    
    Args:
        base_config: Base experiment configuration
        output_dir: Output directory for results
        **kwargs: Additional arguments for ExperimentOrchestrator
        
    Returns:
        ExperimentOrchestrator instance
    """
    return ExperimentOrchestrator(
        base_config=base_config,
        output_dir=output_dir,
        **kwargs
    )