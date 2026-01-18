#!/bin/bash
#SBATCH --job-name=admet_benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=3700
#SBATCH --time=02:00:00

# Load Python module
module purge
module load Python/3.8.6-GCCcore-10.2.0  # Adjust to your cluster's Python module

# Create virtual environment
python -m venv admet_venv

# Activate the virtual environment
source admet_venv/bin/activate

# Install dependencies with pip
pip install -r requirements.txt

# Run the script
python benchmark.py