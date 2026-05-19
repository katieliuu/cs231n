#!/bin/bash
# BraTS Swin UNETR — FarmShare GPU batch job (oat nodes: NVIDIA GPUs per SRCC docs).
#
# If submission fails, try lowering --time and/or --mem first; check exact message from sbatch.
# If squeue fails with "Unable to contact slurm controller", try another login node (rice-02…05)
# or email srcc-support with "FarmShare" in the subject.
#
# Submit:  cd /scratch/users/linika && sbatch run_brats_swin_l4.sbatch

#SBATCH --job-name=brats_swin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/linika/checkpoints/logs/slurm_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/logs/slurm_%j.err

# Was 120h + 128G — many clusters reject overly large requests. Raise after jobs run:
#   #SBATCH --mem=128G
#   #SBATCH --time=120:00:00

set -euo pipefail
cd /scratch/users/linika
mkdir -p checkpoints/logs

echo "=== $(date) brats_swin host=$(hostname) JOB_ID=${SLURM_JOB_ID:-} ==="

# --- Activate YOUR environment where MONAI + PyTorch are installed (compute nodes often lack these in system python3) ---
# Uncomment ONE line after checking paths: ls ~/miniconda3 ~/anaconda3 ~/.conda/envs

# source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate base
# source "$HOME/anaconda3/etc/profile.d/conda.sh" && conda activate base

nvidia-smi 2>&1 || {
  echo "ERROR: nvidia-smi failed (no GPU driver on this node?)"
  exit 1
}
echo "=== python ==="
command -v python3
python3 -V
echo "=== import torch, monai (will fail if env not set up) ==="
python3 - <<'PY' || { echo "ERROR: Python imports failed. Uncomment conda activate above."; exit 1; }
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
import monai
print("monai", monai.__version__)
PY

export BRATS_FULL_TRAIN=1
export BRATS_MAX_EPOCHS=50
export BRATS_VAL_EVERY=5
export BRATS_RESUME=1
export BRATS_RESUME_FILE_EPOCH=10

export BRATS_LOG_FILE="/scratch/users/linika/checkpoints/logs/brats_train_job${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

# Fail fast if CUDA missing after imports proved torch works (training needs GPU)
export REQUIRE_CUDA=1

python3 /scratch/users/linika/Copy_of_swin_unetr_brats21_segmentation_3d.py
echo "=== finished $(date) log=${BRATS_LOG_FILE} ==="
