# MDA-CNN SABR Implementation

Multi-fidelity Data Aggregation CNN for SABR derivatives pricing with limited high-fidelity data.

## Quick Start

1. **Generate data once** (saves MC simulations):
   ```bash
   python generate_sabr_data.py
   ```

2. **Train and evaluate models**:
   ```bash
   jupyter notebook mda_cnn_training.ipynb
   ```

## Key Files

- `generate_sabr_data.py` - Creates and saves MC simulation datasets
- `mda_cnn_training.ipynb` - Main training notebook
- `requirements.txt` - Python dependencies
- `mda_cnn_sabr_project.md` - Project specification

## Datasets Created

- `data/sabr_small.pkl` - 100 samples (quick testing)
- `data/sabr_medium.pkl` - 200 samples (main training)
- `data/sabr_large.pkl` - 500 samples (comparison)

## Results Structure

- `models/` - Saved trained models
- `results/` - Performance metrics and comparisons

## Key Insight

MDA-CNN leverages low-fidelity surface patches to achieve good performance with minimal expensive high-fidelity Monte Carlo data (50-200 samples vs traditional 1000+ samples).

## Architecture

- **MDA-CNN**: LF patches (9x9) + point features → residual prediction
- **MLP Baseline**: Point features only → residual prediction
- **Residual mapping**: Predicts D(ξ) = σ_MC(ξ) - σ_Hagan(ξ)

## Expected Performance

With 200 HF samples:
- MDA-CNN: ~10-20% improvement over MLP baseline
- Demonstrates multi-fidelity advantage with limited data