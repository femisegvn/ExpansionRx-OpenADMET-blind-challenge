#!/bin/bash
#SBATCH --job-name=admet_benchmark
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=gpu  # or cpu, depending on your cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

# Load miniconda module
module load miniconda

# Create conda environment from yml file
conda env create -f environment.yml -n admet_benchmark_hpc

# Activate the environment
conda activate admet_benchmark_hpc

# Run the script
python benchmark.py