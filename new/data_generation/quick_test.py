"""Quick test of the data orchestrator."""

from data_orchestrator import create_default_generation_config, DataGenerationOrchestrator
from sabr_params import create_default_grid_config
from sabr_mc_generator import create_default_mc_config
from hagan_surface_generator import create_default_hagan_config

# Create small test configuration
config = create_default_generation_config()
config.n_parameter_sets = 3
config.output_dir = 'new/data/test_demo'
config.parallel_generation = False

grid_config = create_default_grid_config()
grid_config.n_strikes = 5
grid_config.n_maturities = 3

mc_config = create_default_mc_config()
mc_config.n_paths = 1000
mc_config.convergence_check = False

hagan_config = create_default_hagan_config()

# Create orchestrator and generate parameter sets
orchestrator = DataGenerationOrchestrator(config, grid_config, mc_config, hagan_config)
param_sets = orchestrator.generate_parameter_sets()
print(f'Generated {len(param_sets)} parameter sets')

# Generate one surface
if param_sets:
    surface = orchestrator.generate_single_surface(param_sets[0], 0)
    if surface:
        print(f'Generated surface with shape {surface.hf_surface.shape}')
        quality_score = surface.quality_metrics.get('quality_score', 'N/A')
        print(f'Quality score: {quality_score}')
        print('Demo completed successfully!')
    else:
        print('Failed to generate surface')