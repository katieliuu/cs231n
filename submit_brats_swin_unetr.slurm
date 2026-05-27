#!/bin/bash
#SBATCH --job-name=brats_swin
#SBATCH --output=/scratch/users/linika/checkpoints/slurm_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/slurm_%j.err
#
# Increase wall time and host RAM for long 3D training + validation.
# Edit --partition / --account to match your cluster (examples below).
#
#SBATCH --time=96:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

set -euo pipefail
cd /scratch/users/linika

export BRATS_MPL_BACKEND=Agg
export BRATS_FULL_TRAIN=1

# Resume weights after epoch 9; checkpoint file is model_epoch_10.pth (next loop epoch = 10).
export BRATS_RESUME=1
export BRATS_RESUME_FILE_EPOCH=10

# After training, write FLAIR / GT / pred / overlay PNG using this checkpoint.
export BRATS_GEN_OVERLAYS=1
export BRATS_OVERLAY_CHECKPOINT=/scratch/users/linika/checkpoints/model_epoch_10.pth
export BRATS_OVERLAY_OUTDIR=/scratch/users/linika/checkpoints/overlays

# Optional: change subject id (digits only, as in original script).
# export BRATS_OVERLAY_CASE=01619

exec python3 /scratch/users/linika/Copy_of_swin_unetr_brats21_segmentation_3d.py
