#!/bin/bash
# Merge benchmark parts after array completes. Submit with:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> run_benchmark_merge.sbatch
#SBATCH --job-name=brats_merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/linika/checkpoints/logs/benchmark_merge_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/logs/benchmark_merge_%j.err

set -euo pipefail
cd /scratch/users/linika
/home/users/linika/miniconda3/envs/swin/bin/python /scratch/users/linika/merge_benchmark_results.py "$@"
echo "Merge finished $(date)"
