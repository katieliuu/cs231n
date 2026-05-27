#!/bin/bash
# Full benchmark: all eval conditions on a given checkpoint (default: pct50 epoch 50).
# Usage: sbatch run_full_benchmark.sbatch [checkpoint_path]
#SBATCH --job-name=brats_benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# Inference-only: 32G with 8 CPUs = 4G/CPU (scheduling-friendly on FarmShare).
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/users/linika/checkpoints/logs/benchmark_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/logs/benchmark_%j.err

set -euo pipefail
cd /scratch/users/linika
mkdir -p checkpoints/logs checkpoints/benchmark

SWIN_PY=/home/users/linika/miniconda3/envs/swin/bin/python
CKPT="${1:-checkpoints/curriculum/pct50/model_epoch_50.pth}"
if [[ "${CKPT}" != /* ]]; then
  CKPT="/scratch/users/linika/${CKPT}"
fi

"${SWIN_PY}" -c "import torch, scipy, monai; assert torch.cuda.is_available()" || exit 1

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: Checkpoint not found: ${CKPT}"
  exit 1
fi

echo "Using checkpoint: ${CKPT}"

"${SWIN_PY}" /scratch/users/linika/eval_full_benchmark.py \
  --checkpoint "${CKPT}" \
  --out-dir /scratch/users/linika/checkpoints/benchmark

echo "Benchmark finished $(date)"
