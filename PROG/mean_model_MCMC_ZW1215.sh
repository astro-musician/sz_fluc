#!/bin/bash

#SBATCH --job-name=MCMC --output=OutMCMC_ZW1215.out
#SBATCH --partition=xifu -N1 --cpus-per-task=4

source /home/sila/miniconda3/etc/profile.d/conda.sh
conda activate myenv

srun python mean_model_MCMC_ZW1215.py