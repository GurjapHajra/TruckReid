# TruckReid Project

## Setup Environment

This project requires a specific Python environment to avoid binary incompatibilities with NumPy and Scikit-Learn.

### Using Conda (Recommended)

1. Create the environment from the provided file:

   ```bash
   conda env create -f environment.yml
   ```

   Or create it manually:

   ```bash
   conda create -n truckreid python=3.10 numpy scikit-learn pandas -c conda-forge -y
   ```

2. Activate the environment:

   ```bash
   conda activate truckreid
   ```

3. Run the training script:
   ```bash
   python train.py
   ```

## Changes Made

- **Fixed Binary Incompatibility**: Switched to a clean conda environment to resolve `ValueError: numpy.dtype size changed`.
- **Fixed Topology Logic**: Updated `train.py` to correctly handle "Southbound" traffic (Drone 3 -> 2 -> 1). Previously, it only supported Northbound (1 -> 2 -> 3), causing Southbound trucks to have no candidates.
- **Tuned Threshold**: Lowered the cosine similarity threshold from 0.75 to 0.6 based on observed scores (typically ~0.61-0.64 for matches).
