#!/bin/bash
#SBATCH --job-name="trainy"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=16
#SBATCH --partition=epyc2



# Your code below this line
module load Anaconda3
module load CUDA/12.2.0
module load Workspace_Home
eval "$(conda shell.bash hook)"
NOW=$(date +"%Y-%m-%d/%H-%M-%S")

/storage/homefs/tf24s166/.conda/envs/performance_metrics/bin/python /storage/homefs/tf24s166/code/Volumetric_ART/scripts/training.py

