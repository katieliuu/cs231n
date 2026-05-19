#!/bin/bash
#SBATCH --job-name=brats_swin
#SBATCH --output=/scratch/users/linika/checkpoints/slurm_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/slurm_%j.err
# FarmShare (Stanford): flexible single-GPU job; scheduler may assign
# L40S / A100 / A40 / L4 or another GPU type on partition gpu.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#
# Use --cpus-per-task=16 if you need more CPU parallelism for dataloaders.
#
set -euo pipefail

log() { echo "[$(date -Iseconds)] $*"; }

SWIN_PY=/home/users/linika/miniconda3/envs/swin/bin/python
if [[ ! -x "${SWIN_PY}" ]]; then
  log "Missing conda env python: ${SWIN_PY}"
  exit 1
fi

if ! "${SWIN_PY}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log "PyTorch CUDA not available; not starting training."
  exit 1
fi

cd /scratch/users/linika
export BRATS_MPL_BACKEND=Agg
export BRATS_FULL_TRAIN=1
export BRATS_MAX_EPOCHS=50
export BRATS_RESUME=1
export BRATS_RESUME_FILE_EPOCH=10

_DEV=$("${SWIN_PY}" -c "import torch; print(torch.cuda.get_device_name(0))")
log "Training on CUDA device: ${_DEV}"
exec "${SWIN_PY}" /scratch/users/linika/Copy_of_swin_unetr_brats21_segmentation_3d.py
