"""
Integration tests for data generation orchestrator.

Tests the complete data generation workflow including parameter sampling,
surface generation, quality validation, and data organization.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import time

from .data_orchestrator import (
    DataGenerationOrchestrator, DataGenerationConfig, SurfaceData,
    DataQualityValidator, DataSaver, GenerationProgress,
    create_default_generation_config
)
from .sabr_params import SABRParams, GridConfig, create_default_grid_config, create_test_sabr_params
from .sabr_mc_generator import MCConfig, create_default_mc_config
from .hagan_surface_generator import HaganConfig, create_default_hagan_config


class TestDataQualityValidator:
    """Test data quality validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataQualityValidator()
        self.test_params = create_test_sabr_params()
        self.test_grid = create_default_grid_config()
        
        # Create test surface data
        n_maturities, n_strikes = self.test_grid.get_grid_shape()
        self.test_hf_surface = np.random.uniform(0.1, 0.5, (n_maturities, n_strikes))
        self.test_lf_surface = np.random.uniform(0.1, 0.5, (n_maturities, n_strikes))
        self.test_residuals = self.test_hf_surface - self.test_lf_surface
        
        self.test_surface_data = SurfaceData(
            parameters=self.test_params,
            grid_config=self.test_grid,
            hf_surface=self.test_hf_surface,
            lf_surface=self.test_lf_surface,
            residuals=self.test_residuals,
            strikes=self.test_grid.get_strikes(self.test_params.F0),
            maturities=self.test_grid.get_maturities(),
            generation_time=1.0,
            quality_metrics={}
        )
    
    def test_validate_good_surface(self):
        """Test validation of a good quality surface."""
        results = self.validator.validate_surface(self.test_surface_data)
        
        assert results['is_valid'] is True
        assert 'quality_metrics' in results
        assert 'quality_score' in results
        assert 0 <= results['quality_score'] <= 1
    
    def test_validate_surface_with_nans(self):
        """Test validation of surface with NaN values."""
        # Add some NaN values
        surface_with_nans = self.test_hf_surface.copy()
        surface_with_nans[0, 0] = np.nan
        surface_with_nans[1, 1] = np.nan
        
        surface_data = SurfaceData(
            parameters=self.test_params,
            grid_config=self.test_grid,
            hf_surface=surface_with_nans,
            lf_surface=self.test_lf_surface,
            residuals=surface_with_nans - self.test_lf_surface,
            strikes=self.test_grid.get_strikes(self.test_params.F0),
            maturities=self.test_grid.get_maturities(),
            generation_time=1.0,
            quality_metrics={}
        )
        
        results = self.validator.validate_surface(surface_data)
        
        assert 'hf_nan_fraction' in results['quality_metrics']
        assert results['quality_metrics']['hf_nan_fraction'] > 0
        assert len(results['warnings']) > 0
    
    def test_validate_surface_with_negative_vols(self):
        """Test validation of surface with negative volatilities."""
        # Add negative volatilities
        surface_with_negatives = self.test_hf_surface.copy()
        surface_with_negatives[0, 0] = -0.1
        
        surface_data = SurfaceData(
            parameters=self.test_params,
            grid_config=self.test_grid,
            hf_surface=surface_with_negatives,
            lf_surface=self.test_lf_surface,
            residuals=surface_with_negatives - self.test_lf_surface,
            strikes=self.test_grid.get_strikes(self.test_params.F0),
            maturities=self.test_grid.get_maturities(),
            generation_time=1.0,
            quality_metrics={}
        )
        
        results = self.validator.validate_surface(surface_data)
        
        assert results['is_valid'] is False
        assert any('negative volatilities' in error for error in results['errors'])
    
    def test_validate_empty_surface(self):
        """Test validation of empty surface."""
        empty_surface_data = SurfaceData(
            parameters=self.test_params,
            grid_config=self.test_grid,
            hf_surface=np.array([]),
            lf_surface=np.array([]),
            residuals=np.array([]),
            strikes=np.array([]),
            maturities=np.array([]),
            generation_time=1.0,
            quality_metrics={}
        )
        
        results = self.validator.validate_surface(empty_surface_data)
        
        assert results['is_valid'] is False
        assert any('Empty surface data' in error for error in results['errors'])


class TestDataSaver:
    """Test data saving and loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_saver = DataSaver(self.temp_dir)
        
        # Create test data
        self.test_params = create_test_sabr_params()
        self.test_grid = create_default_grid_config()
        
        n_maturities, n_strikes = self.test_grid.get_grid_shape()
        self.test_surface_data = SurfaceData(
            parameters=self.test_params,
            grid_config=self.test_grid,
            hf_surface=np.random.uniform(0.1, 0.5, (n_maturities, n_strikes)),
            lf_surface=np.random.uniform(0.1, 0.5, (n_maturities, n_strikes)),
            residuals=np.random.uniform(-0.1, 0.1, (n_maturities, n_strikes)),
            strikes=self.test_grid.get_strikes(self.test_params.F0),
            maturities=self.test_grid.get_maturities(),
            generation_time=1.0,
            quality_metrics={'quality_score': 0.8}
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_surface_data(self):
        """Test saving and loading surface data."""
        # Save surface data
        surface_id = 42
        saved_path = self.data_saver.save_surface_data(self.test_surface_data, surface_id)
        
        assert saved_path.exists()
        assert saved_path.name == f"surface_{surface_id:06d}.pkl"
        
        # Load surface data
        loaded_data = self.data_saver.load_surface_data(surface_id)
        
        assert loaded_data.parameters.F0 == self.test_surface_data.parameters.F0
        assert loaded_data.parameters.alpha == self.test_surface_data.parameters.alpha
        np.testing.assert_array_equal(loaded_data.hf_surface, self.test_surface_data.hf_surface)
        np.testing.assert_array_equal(loaded_data.lf_surface, self.test_surface_data.lf_surface)
    
    def test_save_and_load_parameter_sets(self):
        """Test saving and loading parameter sets."""
        # Create test parameter sets
        param_sets = [
            SABRParams(F0=100.0, alpha=0.2, beta=0.5, nu=0.3, rho=-0.2),
            SABRParams(F0=110.0, alpha=0.3, beta=0.7, nu=0.4, rho=-0.3),
            SABRParams(F0=90.0, alpha=0.25, beta=0.6, nu=0.35, rho=-0.1)
        ]
        
        # Save parameter sets
        saved_path = self.data_saver.save_parameter_sets(param_sets)
        assert saved_path.exists()
        
        # Load parameter sets
        loaded_params = self.data_saver.load_parameter_sets()
        
        assert len(loaded_params) == len(param_sets)
        for original, loaded in zip(param_sets, loaded_params):
            assert original.F0 == loaded.F0
            assert original.alpha == loaded.alpha
            assert original.beta == loaded.beta
            assert original.nu == loaded.nu
            assert original.rho == loaded.rho
    
    def test_save_and_load_data_splits(self):
        """Test saving and loading data splits."""
        train_indices = [0, 1, 2, 5, 6, 7]
        val_indices = [3, 8]
        test_indices = [4, 9]
        
        # Save splits
        saved_paths = self.data_saver.save_data_splits(train_indices, val_indices, test_indices)
        
        assert all(path.exists() for path in saved_paths.values())
        
        # Load splits
        loaded_train, loaded_val, loaded_test = self.data_saver.load_data_splits()
        
        assert loaded_train == train_indices
        assert loaded_val == val_indices
        assert loaded_test == test_indices
    
    def test_save_and_load_metadata(self):
        """Test saving and loading generation metadata."""
        metadata = {
            'n_surfaces': 100,
            'generation_time': '2024-01-01T12:00:00',
            'config': {'param1': 'value1', 'param2': 42},
            'statistics': {'mean_quality': 0.85, 'success_rate': 0.95}
        }
        
        # Save metadata
        saved_path = self.data_saver.save_generation_metadata(metadata)
        assert saved_path.exists()
        
        # Load metadata
        loaded_metadata = self.data_saver.load_generation_metadata()
        
        assert loaded_metadata['n_surfaces'] == metadata['n_surfaces']
        assert loaded_metadata['config'] == metadata['config']
        assert loaded_metadata['statistics'] == metadata['statistics']


class TestGenerationProgress:
    """Test progress tracking functionality."""
    
    def test_progress_initialization(self):
        """Test progress initialization."""
        progress = GenerationProgress(total_surfaces=100)
        
        assert progress.total_surfaces == 100
        assert progress.completed_surfaces == 0
        assert progress.failed_surfaces == 0
        assert progress.start_time is None
        assert progress.estimated_completion is None
    
    def test_progress_updates(self):
        """Test progress updates and ETA calculation."""
        progress = GenerationProgress(total_surfaces=100)
        progress.start_time = time.time()
        
        # Simulate some progress
        time.sleep(0.1)  # Small delay to ensure time difference
        progress.update_progress(completed=25, failed=5)
        
        assert progress.completed_surfaces == 25
        assert progress.failed_surfaces == 5
        assert progress.get_progress_percentage() == 30.0  # (25 + 5) / 100 * 100
        assert progress.estimated_completion is not None
    
    def test_eta_calculation(self):
        """Test ETA calculation."""
        progress = GenerationProgress(total_surfaces=100)
        progress.start_time = time.time()
        
        # Simulate progress
        time.sleep(0.01)
        progress.update_progress(completed=10)
        
        eta_string = progress.get_eta_string()
        assert isinstance(eta_string, str)
        assert eta_string != "Unknown"


class TestDataGenerationOrchestrator:
    """Test the main data generation orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configurations
        self.config = DataGenerationConfig(
            n_parameter_sets=5,  # Small number for testing
            sampling_strategy="uniform",
            hf_budget=50,
            validation_split=0.2,
            test_split=0.2,
            output_dir=self.temp_dir,
            save_intermediate=True,
            parallel_generation=False,  # Disable for testing
            random_seed=42,
            quality_checks=True,
            outlier_detection=True
        )
        
        self.grid_config = GridConfig(
            strike_range=(0.8, 1.2),
            maturity_range=(0.5, 2.0),
            n_strikes=5,  # Small grid for testing
            n_maturities=3,
            log_strikes=False,
            log_maturities=False
        )
        
        self.mc_config = MCConfig(
            n_paths=1000,  # Small number for testing
            n_steps=50,
            convergence_check=False,  # Disable for speed
            random_seed=42
        )
        
        self.hagan_config = create_default_hagan_config()
        
        self.orchestrator = DataGenerationOrchestrator(
            config=self.config,
            grid_config=self.grid_config,
            mc_config=self.mc_config,
            hagan_config=self.hagan_config
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.config == self.config
        assert self.orchestrator.grid_config == self.grid_config
        assert self.orchestrator.mc_config == self.mc_config
        assert self.orchestrator.progress.total_surfaces == self.config.n_parameter_sets
    
    def test_generate_parameter_sets(self):
        """Test parameter set generation."""
        parameter_sets = self.orchestrator.generate_parameter_sets()
        
        assert len(parameter_sets) <= self.config.n_parameter_sets
        assert all(isinstance(params, SABRParams) for params in parameter_sets)
        
        # Check that parameter file was saved
        param_file = Path(self.temp_dir) / "raw" / "parameter_sets.csv"
        assert param_file.exists()
    
    def test_generate_single_surface(self):
        """Test single surface generation."""
        test_params = create_test_sabr_params()
        surface_data = self.orchestrator.generate_single_surface(test_params, 0)
        
        assert surface_data is not None
        assert isinstance(surface_data, SurfaceData)
        assert surface_data.parameters == test_params
        assert surface_data.hf_surface.shape == self.grid_config.get_grid_shape()
        assert surface_data.lf_surface.shape == self.grid_config.get_grid_shape()
        assert surface_data.residuals.shape == self.grid_config.get_grid_shape()
        assert surface_data.generation_time > 0
    
    def test_generate_surfaces_sequential(self):
        """Test sequential surface generation."""
        # Generate parameter sets
        parameter_sets = self.orchestrator.generate_parameter_sets()
        
        # Generate surfaces
        surfaces = self.orchestrator.generate_surfaces_sequential(parameter_sets)
        
        assert len(surfaces) <= len(parameter_sets)
        assert all(isinstance(surface, SurfaceData) for surface in surfaces)
        
        # Check progress was updated
        assert self.orchestrator.progress.completed_surfaces == len(surfaces)
    
    def test_create_data_splits(self):
        """Test data split creation."""
        n_surfaces = 10
        train_indices, val_indices, test_indices = self.orchestrator.create_data_splits(n_surfaces)
        
        # Check split sizes
        expected_test = int(n_surfaces * self.config.test_split)
        expected_val = int(n_surfaces * self.config.validation_split)
        expected_train = n_surfaces - expected_test - expected_val
        
        assert len(test_indices) == expected_test
        assert len(val_indices) == expected_val
        assert len(train_indices) == expected_train
        
        # Check no overlap
        all_indices = set(train_indices + val_indices + test_indices)
        assert len(all_indices) == n_surfaces
        assert all_indices == set(range(n_surfaces))
        
        # Check split files were saved
        splits_dir = Path(self.temp_dir) / "splits"
        assert (splits_dir / "train_indices.npy").exists()
        assert (splits_dir / "val_indices.npy").exists()
        assert (splits_dir / "test_indices.npy").exists()
    
    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_calls = []
        
        def test_callback(progress: GenerationProgress):
            callback_calls.append(progress.completed_surfaces)
        
        self.orchestrator.add_progress_callback(test_callback)
        
        # Generate some data to trigger callbacks
        parameter_sets = self.orchestrator.generate_parameter_sets()
        surfaces = self.orchestrator.generate_surfaces_sequential(parameter_sets[:2])  # Just first 2
        
        # Check that callback was called
        assert len(callback_calls) > 0
    
    @pytest.mark.slow
    def test_generate_complete_dataset(self):
        """Test complete dataset generation workflow."""
        # This is a comprehensive integration test
        result = self.orchestrator.generate_complete_dataset()
        
        assert 'surfaces' in result
        assert 'metadata' in result
        assert 'train_indices' in result
        assert 'val_indices' in result
        assert 'test_indices' in result
        
        surfaces = result['surfaces']
        metadata = result['metadata']
        
        # Check surfaces
        assert len(surfaces) <= self.config.n_parameter_sets
        assert all(isinstance(surface, SurfaceData) for surface in surfaces)
        
        # Check metadata
        assert metadata['n_surfaces_generated'] == len(surfaces)
        assert metadata['success_rate'] <= 1.0
        assert 'dataset_statistics' in metadata
        assert 'generation_config' in metadata
        
        # Check files were created
        base_dir = Path(self.temp_dir)
        assert (base_dir / "raw" / "parameter_sets.csv").exists()
        assert (base_dir / "raw" / "generation_metadata.json").exists()
        assert (base_dir / "splits" / "train_indices.npy").exists()
        
        # Check that some surface files were created
        raw_files = list((base_dir / "raw").glob("surface_*.pkl"))
        assert len(raw_files) == len(surfaces)


class TestIntegrationWorkflow:
    """Integration tests for the complete workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end data generation workflow."""
        # Create configurations
        config = create_default_generation_config()
        config.n_parameter_sets = 3  # Small for testing
        config.output_dir = self.temp_dir
        config.parallel_generation = False
        
        grid_config = create_default_grid_config()
        grid_config.n_strikes = 5
        grid_config.n_maturities = 3
        
        mc_config = create_default_mc_config()
        mc_config.n_paths = 500
        mc_config.convergence_check = False
        
        hagan_config = create_default_hagan_config()
        
        # Create orchestrator
        orchestrator = DataGenerationOrchestrator(
            config=config,
            grid_config=grid_config,
            mc_config=mc_config,
            hagan_config=hagan_config
        )
        
        # Run complete workflow
        result = orchestrator.generate_complete_dataset()
        
        # Verify results
        assert len(result['surfaces']) > 0
        assert result['metadata']['success_rate'] > 0
        
        # Verify file structure
        base_dir = Path(self.temp_dir)
        assert (base_dir / "raw").exists()
        assert (base_dir / "processed").exists()
        assert (base_dir / "splits").exists()
        
        # Verify data can be loaded back
        data_saver = DataSaver(self.temp_dir)
        loaded_params = data_saver.load_parameter_sets()
        loaded_metadata = data_saver.load_generation_metadata()
        train_idx, val_idx, test_idx = data_saver.load_data_splits()
        
        assert len(loaded_params) > 0
        assert loaded_metadata['n_surfaces_generated'] > 0
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(result['surfaces'])
    
    def test_workflow_with_quality_filtering(self):
        """Test workflow with quality filtering enabled."""
        config = create_default_generation_config()
        config.n_parameter_sets = 5
        config.output_dir = self.temp_dir
        config.quality_checks = True
        config.outlier_detection = True
        config.parallel_generation = False
        
        grid_config = create_default_grid_config()
        grid_config.n_strikes = 5
        grid_config.n_maturities = 3
        
        mc_config = create_default_mc_config()
        mc_config.n_paths = 500
        mc_config.convergence_check = False
        
        orchestrator = DataGenerationOrchestrator(
            config=config,
            grid_config=grid_config,
            mc_config=mc_config
        )
        
        result = orchestrator.generate_complete_dataset()
        
        # Check that quality metrics were computed
        surfaces = result['surfaces']
        if surfaces:
            assert all('quality_score' in surface.quality_metrics for surface in surfaces)
            
        # Check that statistics include quality information
        stats = result['metadata']['dataset_statistics']
        if 'quality_statistics' in stats:
            assert 'mean' in stats['quality_statistics']
            assert 'std' in stats['quality_statistics']


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])