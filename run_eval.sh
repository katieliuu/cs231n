#!/bin/bash
#SBATCH --job-name=swin_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/linika/logs/swin_eval_%j.out
#SBATCH --error=/scratch/users/linika/logs/swin_eval_%j.err

source /home/users/linika/miniconda3/bin/activate swin

python /scratch/users/linika/eval_sensitivity_hd95.py --epochs 50
