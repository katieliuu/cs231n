#!/bin/bash
# One benchmark condition per array task (minimal CPU/RAM for faster scheduling).
# Usage: sbatch run_benchmark_array.sbatch [checkpoint_path]
#SBATCH --job-name=brats_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/users/linika/checkpoints/logs/bench_part_%A_%a.out
#SBATCH --error=/scratch/users/linika/checkpoints/logs/bench_part_%A_%a.err

set -euo pipefail
cd /scratch/users/linika

CONDITIONS=(baseline gaussian_noise motion_blur bias_field without_FLAIR without_T1ce without_T1 without_T2)
COND="${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"

SWIN_PY=/home/users/linika/miniconda3/envs/swin/bin/python
CKPT="${1:-checkpoints/model_epoch_50.pth}"
if [[ "${CKPT}" != /* ]]; then
  CKPT="/scratch/users/linika/${CKPT}"
fi

PARTS_DIR=/scratch/users/linika/checkpoints/benchmark/parts
mkdir -p checkpoints/logs "${PARTS_DIR}"

if [[ -f "${PARTS_DIR}/${COND}.json" ]]; then
  echo "Part already exists: ${PARTS_DIR}/${COND}.json — skipping"
  exit 0
fi

"${SWIN_PY}" -c "import torch, scipy, monai; assert torch.cuda.is_available()" || exit 1
if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: Checkpoint not found: ${CKPT}"
  exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: condition=${COND} checkpoint=${CKPT}"

"${SWIN_PY}" /scratch/users/linika/eval_full_benchmark.py \
  --checkpoint "${CKPT}" \
  --conditions "${COND}" \
  --parts-dir "${PARTS_DIR}"

echo "Done ${COND} $(date)"
