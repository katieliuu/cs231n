#!/bin/bash
# Single benchmark condition (set COND before submit).
#   sbatch --export=ALL,COND=without_FLAIR run_benchmark_one.sbatch [checkpoint_path]
#SBATCH --job-name=brats_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/linika/checkpoints/logs/bench_%x_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/logs/bench_%x_%j.err

set -euo pipefail
cd /scratch/users/linika

: "${COND:?Set COND (e.g. without_FLAIR) via sbatch --export=ALL,COND=...}"

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

echo "condition=${COND} checkpoint=${CKPT}"

"${SWIN_PY}" /scratch/users/linika/eval_full_benchmark.py \
  --checkpoint "${CKPT}" \
  --conditions "${COND}" \
  --parts-dir "${PARTS_DIR}"

echo "Done ${COND} $(date)"
