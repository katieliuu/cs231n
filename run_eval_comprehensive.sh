#!/bin/bash
#SBATCH --job-name=swin_eval_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/linika/logs/swin_eval_comprehensive_%j.out
#SBATCH --error=/scratch/users/linika/logs/swin_eval_comprehensive_%j.err

set -euo pipefail
source /home/users/linika/miniconda3/bin/activate swin
cd /scratch/users/linika
python eval_comprehensive.py --epochs 50
