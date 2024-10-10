#!/bin/bash

#SBATCH --job-name=MCMC --output=OutMCMC_A3158.out
#SBATCH --partition=xifu -N1 --cpus-per-task=4

source /home/sila/miniconda3/etc/profile.d/conda.sh
conda activate myenv

srun python mean_model_MCMC_A3158.py