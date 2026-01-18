#!/bin/bash
#SBATCH --job-name=admet_benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=3700
#SBATCH --time=2:00:00

# Load miniconda module
module purge
module load miniconda

# Create conda environment from yml file
conda env create -f environment.yml -n admet_benchmark_hpc

# Activate the environment
conda activate admet_benchmark_hpc

# Run the script
python benchmark.py