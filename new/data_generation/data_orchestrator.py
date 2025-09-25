"""
Data generation orchestrator for SABR volatility surface modeling.

This module coordinates the generation of both high-fidelity Monte Carlo and
low-fidelity Hagan analytical surfaces, with data quality validation,
progress tracking, and organized file storage.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import json
import time
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

from .sabr_params import SABRParams, GridConfig, ParameterSampler
from .sabr_mc_generator import SABRMCGenerator, MCConfig, ParallelSABRMCGenerator
from .hagan_surface_generator import HaganSurfaceGenerator, HaganConfig
from ..utils.logging_utils import get_logger
from ..utils.common import ensure_directory, Timer, format_time

logger = get_logger(__name__)


@dataclass
class DataGenerationConfig:
    """
    Configuration for data generation orchestrator.
    
    Attributes:
        n_parameter_sets: Number of SABR parameter combinations to generate
        sampling_strategy: Parameter sampling strategy ("uniform", "latin_hypercube", "adaptive")
        hf_budget: Number of high-fidelity points per surface
        validation_split: Fraction of data for validation
        test_split: Fraction of data for test
        output_dir: Base directory for saving generated data
        save_intermediate: Whether to save intermediate results
        parallel_generation: Enable parallel surface generation
        max_workers: Maximum number of parallel workers
        random_seed: Random seed for reproducibility
        quality_checks: Enable data quality validation
        outlier_detection: Enable outlier detection and removal
    """
    n_parameter_sets: int = 1000
    sampling_strategy: str = "latin_hypercube"
    hf_budget: int = 200
    validation_split: float = 0.15
    test_split: float = 0.15
    output_dir: str = "new/data"
    save_intermediate: bool = True
    parallel_generation: bool = True
    max_workers: Optional[int] = None
    random_seed: int = 42
    quality_checks: bool = True
    outlier_detection: bool = True


@dataclass
class SurfaceData:
    """
    Container for a single volatility surface dataset.
    
    Attributes:
        parameters: SABR parameters used to generate the surface
        grid_config: Grid configuration for the surface
        hf_surface: High-fidelity Monte Carlo surface
        lf_surface: Low-fidelity Hagan analytical surface
        residuals: Residuals (HF - LF)
        strikes: Strike prices array
        maturities: Maturity times array
        generation_time: Time taken to generate this surface
        quality_metrics: Data quality metrics
    """
    parameters: SABRParams
    grid_config: GridConfig
    hf_surface: np.ndarray
    lf_surface: np.ndarray
    residuals: np.ndarray
    strikes: np.ndarray
    maturities: np.ndarray
    generation_time: float
    quality_metrics: Dict[str, float]


@dataclass
class GenerationProgress:
    """
    Progress tracking for data generation.
    
    Attributes:
        total_surfaces: Total number of surfaces to generate
        completed_surfaces: Number of completed surfaces
        failed_surfaces: Number of failed surfaces
        start_time: Generation start time
        estimated_completion: Estimated completion time
        current_stage: Current generation stage
        stage_progress: Progress within current stage
    """
    total_surfaces: int
    completed_surfaces: int = 0
    failed_surfaces: int = 0
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    current_stage: str = "initializing"
    stage_progress: float = 0.0
    
    def update_progress(self, completed: int, failed: int = 0):
        """Update progress counters and estimate completion time."""
        self.completed_surfaces = completed
        self.failed_surfaces = failed
        
        if self.start_time and completed > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = completed / elapsed  # surfaces per second
            remaining = self.total_surfaces - completed - failed
            
            if rate > 0:
                eta_seconds = remaining / rate
                self.estimated_completion = datetime.now() + timedelta(seconds=eta_seconds)
    
    def get_progress_percentage(self) -> float:
        """Get overall progress as percentage."""
        total_processed = self.completed_surfaces + self.failed_surfaces
        return (total_processed / self.total_surfaces) * 100 if self.total_surfaces > 0 else 0.0
    
    def get_eta_string(self) -> str:
        """Get estimated time to completion as string."""
        if self.estimated_completion:
            eta = self.estimated_completion - datetime.now()
            return format_time(eta.total_seconds())
        return "Unknown"


class DataQualityValidator:
    """
    Data quality validation and outlier detection for volatility surfaces.
    """
    
    def __init__(self, tolerance_config: Optional[Dict[str, float]] = None):
        """
        Initialize data quality validator.
        
        Args:
            tolerance_config: Dictionary of tolerance parameters
        """
        self.tolerances = tolerance_config or {
            'min_volatility': 0.001,      # 0.1% minimum volatility
            'max_volatility': 5.0,        # 500% maximum volatility
            'max_nan_fraction': 0.1,      # 10% maximum NaN values
            'min_smile_monotonicity': 0.8, # 80% of smile should be monotonic in wings
            'max_residual_std': 2.0,      # Maximum residual standard deviation
            'max_surface_gradient': 10.0,  # Maximum surface gradient
        }
    
    def validate_surface(self, surface_data: SurfaceData) -> Dict[str, Any]:
        """
        Validate a single surface and compute quality metrics.
        
        Args:
            surface_data: Surface data to validate
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'quality_metrics': {}
        }
        
        hf_surface = surface_data.hf_surface
        lf_surface = surface_data.lf_surface
        residuals = surface_data.residuals
        
        # Check for basic data integrity
        self._check_data_integrity(hf_surface, lf_surface, residuals, validation_results)
        
        # Check volatility ranges
        self._check_volatility_ranges(hf_surface, lf_surface, validation_results)
        
        # Check for excessive NaN values
        self._check_nan_values(hf_surface, lf_surface, validation_results)
        
        # Check smile properties
        self._check_smile_properties(hf_surface, lf_surface, surface_data.strikes, validation_results)
        
        # Check residual properties
        self._check_residual_properties(residuals, validation_results)
        
        # Check surface smoothness
        self._check_surface_smoothness(hf_surface, lf_surface, validation_results)
        
        # Compute overall quality score
        validation_results['quality_score'] = self._compute_quality_score(validation_results['quality_metrics'])
        
        return validation_results
    
    def _check_data_integrity(self, hf_surface: np.ndarray, lf_surface: np.ndarray, 
                            residuals: np.ndarray, results: Dict[str, Any]):
        """Check basic data integrity."""
        if hf_surface.shape != lf_surface.shape:
            results['errors'].append("HF and LF surfaces have different shapes")
            results['is_valid'] = False
        
        if residuals.shape != hf_surface.shape:
            results['errors'].append("Residuals have different shape than surfaces")
            results['is_valid'] = False
        
        if hf_surface.size == 0:
            results['errors'].append("Empty surface data")
            results['is_valid'] = False
    
    def _check_volatility_ranges(self, hf_surface: np.ndarray, lf_surface: np.ndarray, 
                               results: Dict[str, Any]):
        """Check volatility value ranges."""
        for name, surface in [("HF", hf_surface), ("LF", lf_surface)]:
            finite_vals = surface[np.isfinite(surface)]
            
            if len(finite_vals) > 0:
                min_vol = np.min(finite_vals)
                max_vol = np.max(finite_vals)
                
                results['quality_metrics'][f'{name.lower()}_min_vol'] = min_vol
                results['quality_metrics'][f'{name.lower()}_max_vol'] = max_vol
                
                if min_vol < self.tolerances['min_volatility']:
                    results['warnings'].append(f"{name} surface has very low volatilities (min: {min_vol:.6f})")
                
                if max_vol > self.tolerances['max_volatility']:
                    results['warnings'].append(f"{name} surface has very high volatilities (max: {max_vol:.4f})")
                
                if min_vol < 0:
                    results['errors'].append(f"{name} surface has negative volatilities")
                    results['is_valid'] = False
    
    def _check_nan_values(self, hf_surface: np.ndarray, lf_surface: np.ndarray, 
                         results: Dict[str, Any]):
        """Check for excessive NaN values."""
        for name, surface in [("HF", hf_surface), ("LF", lf_surface)]:
            nan_fraction = np.sum(np.isnan(surface)) / surface.size
            results['quality_metrics'][f'{name.lower()}_nan_fraction'] = nan_fraction
            
            if nan_fraction > self.tolerances['max_nan_fraction']:
                results['warnings'].append(f"{name} surface has {nan_fraction:.2%} NaN values")
                
                if nan_fraction > 0.5:  # More than 50% NaN is an error
                    results['errors'].append(f"{name} surface has excessive NaN values")
                    results['is_valid'] = False
    
    def _check_smile_properties(self, hf_surface: np.ndarray, lf_surface: np.ndarray, 
                              strikes: np.ndarray, results: Dict[str, Any]):
        """Check volatility smile properties."""
        # Check smile shape for each maturity
        n_maturities, n_strikes = hf_surface.shape
        
        for name, surface in [("HF", hf_surface), ("LF", lf_surface)]:
            smile_scores = []
            
            for t_idx in range(n_maturities):
                smile = surface[t_idx, :]
                finite_mask = np.isfinite(smile)
                
                if np.sum(finite_mask) > 3:  # Need at least 4 points
                    finite_smile = smile[finite_mask]
                    finite_strikes = strikes[finite_mask]
                    
                    # Check for basic smile shape (U-shaped)
                    mid_idx = len(finite_smile) // 2
                    left_wing = finite_smile[:mid_idx]
                    right_wing = finite_smile[mid_idx:]
                    
                    # Left wing should generally decrease towards ATM
                    left_monotonic = np.sum(np.diff(left_wing) <= 0) / max(1, len(left_wing) - 1)
                    # Right wing should generally increase away from ATM
                    right_monotonic = np.sum(np.diff(right_wing) >= 0) / max(1, len(right_wing) - 1)
                    
                    smile_score = (left_monotonic + right_monotonic) / 2
                    smile_scores.append(smile_score)
            
            if smile_scores:
                avg_smile_score = np.mean(smile_scores)
                results['quality_metrics'][f'{name.lower()}_smile_score'] = avg_smile_score
                
                if avg_smile_score < self.tolerances['min_smile_monotonicity']:
                    results['warnings'].append(f"{name} surface has poor smile shape (score: {avg_smile_score:.3f})")
    
    def _check_residual_properties(self, residuals: np.ndarray, results: Dict[str, Any]):
        """Check residual properties."""
        finite_residuals = residuals[np.isfinite(residuals)]
        
        if len(finite_residuals) > 0:
            residual_mean = np.mean(finite_residuals)
            residual_std = np.std(finite_residuals)
            residual_max_abs = np.max(np.abs(finite_residuals))
            
            results['quality_metrics']['residual_mean'] = residual_mean
            results['quality_metrics']['residual_std'] = residual_std
            results['quality_metrics']['residual_max_abs'] = residual_max_abs
            
            if residual_std > self.tolerances['max_residual_std']:
                results['warnings'].append(f"High residual standard deviation: {residual_std:.4f}")
            
            # Check for systematic bias
            if abs(residual_mean) > 0.1:  # 10% systematic bias
                results['warnings'].append(f"Systematic residual bias detected: {residual_mean:.4f}")
    
    def _check_surface_smoothness(self, hf_surface: np.ndarray, lf_surface: np.ndarray, 
                                results: Dict[str, Any]):
        """Check surface smoothness via gradients."""
        for name, surface in [("HF", hf_surface), ("LF", lf_surface)]:
            if surface.shape[0] > 1 and surface.shape[1] > 1:
                # Compute gradients
                grad_t = np.gradient(surface, axis=0)  # Time gradient
                grad_k = np.gradient(surface, axis=1)  # Strike gradient
                
                # Compute gradient magnitudes
                grad_magnitude = np.sqrt(grad_t**2 + grad_k**2)
                finite_grad = grad_magnitude[np.isfinite(grad_magnitude)]
                
                if len(finite_grad) > 0:
                    max_gradient = np.max(finite_grad)
                    mean_gradient = np.mean(finite_grad)
                    
                    results['quality_metrics'][f'{name.lower()}_max_gradient'] = max_gradient
                    results['quality_metrics'][f'{name.lower()}_mean_gradient'] = mean_gradient
                    
                    if max_gradient > self.tolerances['max_surface_gradient']:
                        results['warnings'].append(f"{name} surface has high gradients (max: {max_gradient:.4f})")
    
    def _compute_quality_score(self, metrics: Dict[str, float]) -> float:
        """Compute overall quality score from metrics."""
        score = 1.0
        
        # Penalize for high NaN fractions
        for key in ['hf_nan_fraction', 'lf_nan_fraction']:
            if key in metrics:
                score *= (1.0 - metrics[key])
        
        # Reward good smile shapes
        for key in ['hf_smile_score', 'lf_smile_score']:
            if key in metrics:
                score *= metrics[key]
        
        # Penalize high residual standard deviation
        if 'residual_std' in metrics:
            score *= max(0.1, 1.0 - metrics['residual_std'] / self.tolerances['max_residual_std'])
        
        return max(0.0, min(1.0, score))


class DataSaver:
    """
    Utilities for saving and loading generated surface data with proper organization.
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize data saver.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.splits_dir = self.base_dir / "splits"
        
        # Ensure directories exist
        for dir_path in [self.raw_dir, self.processed_dir, self.splits_dir]:
            ensure_directory(dir_path)
    
    def save_surface_data(self, surface_data: SurfaceData, surface_id: int) -> Path:
        """
        Save a single surface dataset.
        
        Args:
            surface_data: Surface data to save
            surface_id: Unique identifier for the surface
            
        Returns:
            Path to saved file
        """
        filename = f"surface_{surface_id:06d}.pkl"
        filepath = self.raw_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(surface_data, f)
        
        return filepath
    
    def load_surface_data(self, surface_id: int) -> SurfaceData:
        """
        Load a single surface dataset.
        
        Args:
            surface_id: Surface identifier
            
        Returns:
            Loaded surface data
        """
        filename = f"surface_{surface_id:06d}.pkl"
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Surface data not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_parameter_sets(self, parameter_sets: List[SABRParams]) -> Path:
        """
        Save parameter sets to CSV file.
        
        Args:
            parameter_sets: List of SABR parameter sets
            
        Returns:
            Path to saved file
        """
        # Convert to DataFrame
        param_dicts = [asdict(params) for params in parameter_sets]
        df = pd.DataFrame(param_dicts)
        
        filepath = self.raw_dir / "parameter_sets.csv"
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def load_parameter_sets(self) -> List[SABRParams]:
        """
        Load parameter sets from CSV file.
        
        Returns:
            List of SABR parameter sets
        """
        filepath = self.raw_dir / "parameter_sets.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Parameter sets not found: {filepath}")
        
        df = pd.read_csv(filepath)
        return [SABRParams(**row) for _, row in df.iterrows()]
    
    def save_generation_metadata(self, metadata: Dict[str, Any]) -> Path:
        """
        Save generation metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Path to saved file
        """
        filepath = self.raw_dir / "generation_metadata.json"
        
        # Convert datetime objects to strings
        metadata_copy = metadata.copy()
        for key, value in metadata_copy.items():
            if isinstance(value, datetime):
                metadata_copy[key] = value.isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(metadata_copy, f, indent=2, default=str)
        
        return filepath
    
    def load_generation_metadata(self) -> Dict[str, Any]:
        """
        Load generation metadata.
        
        Returns:
            Metadata dictionary
        """
        filepath = self.raw_dir / "generation_metadata.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Generation metadata not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_data_splits(self, train_indices: List[int], val_indices: List[int], 
                        test_indices: List[int]) -> Dict[str, Path]:
        """
        Save data split indices.
        
        Args:
            train_indices: Training set indices
            val_indices: Validation set indices
            test_indices: Test set indices
            
        Returns:
            Dictionary of saved file paths
        """
        paths = {}
        
        for name, indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            filepath = self.splits_dir / f"{name}_indices.npy"
            np.save(filepath, np.array(indices))
            paths[name] = filepath
        
        return paths
    
    def load_data_splits(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Load data split indices.
        
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        train_path = self.splits_dir / "train_indices.npy"
        val_path = self.splits_dir / "val_indices.npy"
        test_path = self.splits_dir / "test_indices.npy"
        
        for path in [train_path, val_path, test_path]:
            if not path.exists():
                raise FileNotFoundError(f"Split indices not found: {path}")
        
        train_indices = np.load(train_path).tolist()
        val_indices = np.load(val_path).tolist()
        test_indices = np.load(test_path).tolist()
        
        return train_indices, val_indices, test_indices


class DataGenerationOrchestrator:
    """
    Main orchestrator for SABR volatility surface data generation.
    
    Coordinates the generation of both high-fidelity Monte Carlo and low-fidelity
    Hagan analytical surfaces with quality validation, progress tracking, and
    organized data storage.
    """
    
    def __init__(self, 
                 config: DataGenerationConfig,
                 grid_config: GridConfig,
                 mc_config: MCConfig,
                 hagan_config: Optional[HaganConfig] = None):
        """
        Initialize data generation orchestrator.
        
        Args:
            config: Data generation configuration
            grid_config: Grid configuration for surfaces
            mc_config: Monte Carlo configuration
            hagan_config: Hagan formula configuration
        """
        self.config = config
        self.grid_config = grid_config
        self.mc_config = mc_config
        self.hagan_config = hagan_config or HaganConfig()
        
        # Initialize components
        self.parameter_sampler = ParameterSampler(random_seed=config.random_seed)
        self.mc_generator = SABRMCGenerator(mc_config)
        self.hagan_generator = HaganSurfaceGenerator(self.hagan_config)
        self.parallel_mc_generator = ParallelSABRMCGenerator(mc_config)
        self.quality_validator = DataQualityValidator()
        self.data_saver = DataSaver(config.output_dir)
        
        # Progress tracking
        self.progress = GenerationProgress(total_surfaces=config.n_parameter_sets)
        self.progress_callbacks: List[Callable[[GenerationProgress], None]] = []
        
        logger.info(f"Initialized data generation orchestrator")
        logger.info(f"Configuration: {config.n_parameter_sets} surfaces, "
                   f"{config.sampling_strategy} sampling, "
                   f"HF budget: {config.hf_budget}")
    
    def add_progress_callback(self, callback: Callable[[GenerationProgress], None]):
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self):
        """Notify all progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def generate_parameter_sets(self) -> List[SABRParams]:
        """
        Generate SABR parameter sets using configured sampling strategy.
        
        Returns:
            List of SABR parameter sets
        """
        logger.info(f"Generating {self.config.n_parameter_sets} parameter sets "
                   f"using {self.config.sampling_strategy} sampling")
        
        self.progress.current_stage = "parameter_sampling"
        self._notify_progress()
        
        try:
            parameter_sets = self.parameter_sampler.sample_parameters(
                n_samples=self.config.n_parameter_sets,
                strategy=self.config.sampling_strategy
            )
            
            logger.info(f"Generated {len(parameter_sets)} valid parameter sets")
            
            # Save parameter sets
            if self.config.save_intermediate:
                param_path = self.data_saver.save_parameter_sets(parameter_sets)
                logger.info(f"Saved parameter sets to {param_path}")
            
            return parameter_sets
            
        except Exception as e:
            logger.error(f"Failed to generate parameter sets: {e}")
            raise
    
    def generate_single_surface(self, sabr_params: SABRParams, surface_id: int) -> Optional[SurfaceData]:
        """
        Generate a single volatility surface with both HF and LF data.
        
        Args:
            sabr_params: SABR parameters
            surface_id: Unique surface identifier
            
        Returns:
            SurfaceData object or None if generation failed
        """
        start_time = time.time()
        
        try:
            # Generate grid points
            strikes = self.grid_config.get_strikes(sabr_params.F0)
            maturities = self.grid_config.get_maturities()
            
            # Generate high-fidelity Monte Carlo surface
            logger.debug(f"Generating HF surface {surface_id}")
            hf_result = self.mc_generator.generate_surface(sabr_params, self.grid_config)
            
            # Extract surface from result (handle convergence info if present)
            if isinstance(hf_result, tuple):
                hf_surface, mc_info = hf_result
            else:
                hf_surface = hf_result
                mc_info = {}
            
            # Generate low-fidelity Hagan surface
            logger.debug(f"Generating LF surface {surface_id}")
            lf_surface = self.hagan_generator.generate_surface(sabr_params, self.grid_config)
            
            # Calculate residuals
            residuals = hf_surface - lf_surface
            
            # Create surface data object
            generation_time = time.time() - start_time
            surface_data = SurfaceData(
                parameters=sabr_params,
                grid_config=self.grid_config,
                hf_surface=hf_surface,
                lf_surface=lf_surface,
                residuals=residuals,
                strikes=strikes,
                maturities=maturities,
                generation_time=generation_time,
                quality_metrics={}
            )
            
            # Validate surface quality
            if self.config.quality_checks:
                validation_results = self.quality_validator.validate_surface(surface_data)
                surface_data.quality_metrics = validation_results['quality_metrics']
                
                if not validation_results['is_valid']:
                    logger.warning(f"Surface {surface_id} failed validation: {validation_results['errors']}")
                    if not self.config.outlier_detection:
                        return None
                
                if validation_results['warnings']:
                    logger.debug(f"Surface {surface_id} validation warnings: {validation_results['warnings']}")
            
            logger.debug(f"Generated surface {surface_id} in {generation_time:.2f}s")
            return surface_data
            
        except Exception as e:
            logger.error(f"Failed to generate surface {surface_id}: {e}")
            return None
    
    def generate_surfaces_sequential(self, parameter_sets: List[SABRParams]) -> List[SurfaceData]:
        """
        Generate surfaces sequentially with progress tracking.
        
        Args:
            parameter_sets: List of SABR parameter sets
            
        Returns:
            List of generated surface data
        """
        logger.info("Generating surfaces sequentially")
        
        self.progress.current_stage = "surface_generation"
        self.progress.start_time = datetime.now()
        self._notify_progress()
        
        surfaces = []
        failed_count = 0
        
        for i, sabr_params in enumerate(parameter_sets):
            surface_data = self.generate_single_surface(sabr_params, i)
            
            if surface_data is not None:
                surfaces.append(surface_data)
                
                # Save intermediate results
                if self.config.save_intermediate:
                    self.data_saver.save_surface_data(surface_data, i)
            else:
                failed_count += 1
            
            # Update progress
            self.progress.update_progress(len(surfaces), failed_count)
            self._notify_progress()
            
            # Log progress periodically
            if (i + 1) % 50 == 0 or i == len(parameter_sets) - 1:
                progress_pct = self.progress.get_progress_percentage()
                eta = self.progress.get_eta_string()
                logger.info(f"Progress: {len(surfaces)}/{len(parameter_sets)} surfaces "
                           f"({progress_pct:.1f}%), ETA: {eta}")
        
        logger.info(f"Sequential generation completed: {len(surfaces)} surfaces, "
                   f"{failed_count} failures")
        
        return surfaces
    
    def generate_surfaces_parallel(self, parameter_sets: List[SABRParams]) -> List[SurfaceData]:
        """
        Generate surfaces in parallel with progress tracking.
        
        Args:
            parameter_sets: List of SABR parameter sets
            
        Returns:
            List of generated surface data
        """
        logger.info(f"Generating surfaces in parallel with {self.config.max_workers or 'auto'} workers")
        
        self.progress.current_stage = "surface_generation"
        self.progress.start_time = datetime.now()
        self._notify_progress()
        
        surfaces = [None] * len(parameter_sets)
        completed_count = 0
        failed_count = 0
        
        # Use thread-safe progress updates
        progress_lock = threading.Lock()
        
        def update_progress_thread_safe(completed: int, failed: int):
            with progress_lock:
                self.progress.update_progress(completed, failed)
                self._notify_progress()
        
        # Prepare worker arguments
        worker_args = [
            (params, i, self.grid_config, self.mc_config, self.hagan_config, 
             self.config.quality_checks, self.config.outlier_detection)
            for i, params in enumerate(parameter_sets)
        ]
        
        max_workers = self.config.max_workers or min(len(parameter_sets), 4)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_idx = {
                executor.submit(_generate_surface_worker, args): args[1] 
                for args in worker_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                
                try:
                    surface_data = future.result()
                    
                    if surface_data is not None:
                        surfaces[idx] = surface_data
                        completed_count += 1
                        
                        # Save intermediate results
                        if self.config.save_intermediate:
                            self.data_saver.save_surface_data(surface_data, idx)
                    else:
                        failed_count += 1
                    
                    # Update progress
                    update_progress_thread_safe(completed_count, failed_count)
                    
                    # Log progress periodically
                    total_processed = completed_count + failed_count
                    if total_processed % 50 == 0 or total_processed == len(parameter_sets):
                        progress_pct = self.progress.get_progress_percentage()
                        eta = self.progress.get_eta_string()
                        logger.info(f"Progress: {completed_count}/{len(parameter_sets)} surfaces "
                                   f"({progress_pct:.1f}%), ETA: {eta}")
                    
                except Exception as e:
                    logger.error(f"Worker {idx} failed: {e}")
                    failed_count += 1
                    update_progress_thread_safe(completed_count, failed_count)
        
        # Filter out None values
        valid_surfaces = [s for s in surfaces if s is not None]
        
        logger.info(f"Parallel generation completed: {len(valid_surfaces)} surfaces, "
                   f"{failed_count} failures")
        
        return valid_surfaces
    
    def create_data_splits(self, n_surfaces: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Create train/validation/test splits.
        
        Args:
            n_surfaces: Total number of surfaces
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        logger.info("Creating data splits")
        
        # Set random seed for reproducible splits
        np.random.seed(self.config.random_seed)
        
        # Create shuffled indices
        indices = np.arange(n_surfaces)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        n_test = int(n_surfaces * self.config.test_split)
        n_val = int(n_surfaces * self.config.validation_split)
        n_train = n_surfaces - n_test - n_val
        
        # Create splits
        test_indices = indices[:n_test].tolist()
        val_indices = indices[n_test:n_test + n_val].tolist()
        train_indices = indices[n_test + n_val:].tolist()
        
        logger.info(f"Data splits: {n_train} train, {n_val} val, {n_test} test")
        
        # Save splits
        split_paths = self.data_saver.save_data_splits(train_indices, val_indices, test_indices)
        logger.info(f"Saved data splits to {self.data_saver.splits_dir}")
        
        return train_indices, val_indices, test_indices
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """
        Generate complete dataset with all surfaces and metadata.
        
        Returns:
            Dictionary with generation results and metadata
        """
        logger.info("Starting complete dataset generation")
        
        with Timer("Complete dataset generation"):
            # Generate parameter sets
            parameter_sets = self.generate_parameter_sets()
            
            # Generate surfaces
            if self.config.parallel_generation and len(parameter_sets) > 1:
                surfaces = self.generate_surfaces_parallel(parameter_sets)
            else:
                surfaces = self.generate_surfaces_sequential(parameter_sets)
            
            # Create data splits
            if len(surfaces) > 0:
                train_indices, val_indices, test_indices = self.create_data_splits(len(surfaces))
            else:
                train_indices, val_indices, test_indices = [], [], []
            
            # Compute dataset statistics
            dataset_stats = self._compute_dataset_statistics(surfaces)
            
            # Create metadata
            metadata = {
                'generation_config': asdict(self.config),
                'grid_config': asdict(self.grid_config),
                'mc_config': asdict(self.mc_config),
                'hagan_config': asdict(self.hagan_config),
                'generation_time': datetime.now(),
                'n_parameter_sets_requested': self.config.n_parameter_sets,
                'n_surfaces_generated': len(surfaces),
                'n_surfaces_failed': self.progress.failed_surfaces,
                'success_rate': len(surfaces) / self.config.n_parameter_sets if self.config.n_parameter_sets > 0 else 0,
                'data_splits': {
                    'train': len(train_indices),
                    'val': len(val_indices),
                    'test': len(test_indices)
                },
                'dataset_statistics': dataset_stats
            }
            
            # Save metadata
            metadata_path = self.data_saver.save_generation_metadata(metadata)
            logger.info(f"Saved generation metadata to {metadata_path}")
            
            # Final progress update
            self.progress.current_stage = "completed"
            self._notify_progress()
            
            logger.info(f"Dataset generation completed successfully")
            logger.info(f"Generated {len(surfaces)} surfaces out of {self.config.n_parameter_sets} requested")
            logger.info(f"Success rate: {metadata['success_rate']:.2%}")
            
            return {
                'surfaces': surfaces,
                'metadata': metadata,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices
            }
    
    def _compute_dataset_statistics(self, surfaces: List[SurfaceData]) -> Dict[str, Any]:
        """
        Compute statistics for the generated dataset.
        
        Args:
            surfaces: List of generated surfaces
            
        Returns:
            Dictionary with dataset statistics
        """
        if not surfaces:
            return {}
        
        logger.info("Computing dataset statistics")
        
        # Extract quality metrics
        quality_scores = [s.quality_metrics.get('quality_score', 0) for s in surfaces if s.quality_metrics]
        generation_times = [s.generation_time for s in surfaces]
        
        # Extract parameter ranges
        param_stats = {}
        for param_name in ['F0', 'alpha', 'beta', 'nu', 'rho']:
            values = [getattr(s.parameters, param_name) for s in surfaces]
            param_stats[param_name] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        
        # Extract surface statistics
        hf_vols = []
        lf_vols = []
        residuals = []
        
        for surface in surfaces:
            hf_finite = surface.hf_surface[np.isfinite(surface.hf_surface)]
            lf_finite = surface.lf_surface[np.isfinite(surface.lf_surface)]
            res_finite = surface.residuals[np.isfinite(surface.residuals)]
            
            hf_vols.extend(hf_finite.tolist())
            lf_vols.extend(lf_finite.tolist())
            residuals.extend(res_finite.tolist())
        
        surface_stats = {}
        for name, values in [('hf_volatilities', hf_vols), ('lf_volatilities', lf_vols), ('residuals', residuals)]:
            if values:
                surface_stats[name] = {
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values)
                }
        
        stats = {
            'n_surfaces': len(surfaces),
            'parameter_statistics': param_stats,
            'surface_statistics': surface_stats,
            'generation_time_stats': {
                'min': float(np.min(generation_times)),
                'max': float(np.max(generation_times)),
                'mean': float(np.mean(generation_times)),
                'total': float(np.sum(generation_times))
            }
        }
        
        if quality_scores:
            stats['quality_statistics'] = {
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores)),
                'mean': float(np.mean(quality_scores)),
                'std': float(np.std(quality_scores))
            }
        
        return stats


def _generate_surface_worker(args: Tuple) -> Optional[SurfaceData]:
    """
    Worker function for parallel surface generation.
    
    Args:
        args: Tuple of (sabr_params, surface_id, grid_config, mc_config, hagan_config, quality_checks, outlier_detection)
        
    Returns:
        SurfaceData object or None if generation failed
    """
    sabr_params, surface_id, grid_config, mc_config, hagan_config, quality_checks, outlier_detection = args
    
    try:
        # Create generators for this worker
        mc_generator = SABRMCGenerator(mc_config)
        hagan_generator = HaganSurfaceGenerator(hagan_config)
        quality_validator = DataQualityValidator()
        
        start_time = time.time()
        
        # Generate grid points
        strikes = grid_config.get_strikes(sabr_params.F0)
        maturities = grid_config.get_maturities()
        
        # Generate surfaces
        hf_result = mc_generator.generate_surface(sabr_params, grid_config)
        if isinstance(hf_result, tuple):
            hf_surface, _ = hf_result
        else:
            hf_surface = hf_result
        
        lf_surface = hagan_generator.generate_surface(sabr_params, grid_config)
        residuals = hf_surface - lf_surface
        
        # Create surface data
        generation_time = time.time() - start_time
        surface_data = SurfaceData(
            parameters=sabr_params,
            grid_config=grid_config,
            hf_surface=hf_surface,
            lf_surface=lf_surface,
            residuals=residuals,
            strikes=strikes,
            maturities=maturities,
            generation_time=generation_time,
            quality_metrics={}
        )
        
        # Validate if requested
        if quality_checks:
            validation_results = quality_validator.validate_surface(surface_data)
            surface_data.quality_metrics = validation_results['quality_metrics']
            
            if not validation_results['is_valid'] and not outlier_detection:
                return None
        
        return surface_data
        
    except Exception as e:
        logger.error(f"Worker failed for surface {surface_id}: {e}")
        return None


def create_default_generation_config() -> DataGenerationConfig:
    """
    Create default data generation configuration.
    
    Returns:
        DataGenerationConfig with reasonable defaults
    """
    return DataGenerationConfig(
        n_parameter_sets=100,
        sampling_strategy="latin_hypercube",
        hf_budget=200,
        validation_split=0.15,
        test_split=0.15,
        output_dir="new/data",
        save_intermediate=True,
        parallel_generation=True,
        max_workers=None,
        random_seed=42,
        quality_checks=True,
        outlier_detection=True
    )