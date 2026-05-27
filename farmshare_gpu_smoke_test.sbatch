#!/bin/bash
# One-minute check: Slurm + GPU + torch. Submit: sbatch farmshare_gpu_smoke_test.sbatch
# Read: checkpoints/logs/smoke_%j.out

#SBATCH --job-name=gpu_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/linika/checkpoints/logs/smoke_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/logs/smoke_%j.err

set -x
hostname
date
nvidia-smi
command -v python3
python3 -V
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
