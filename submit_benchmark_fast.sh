#!/bin/bash
# Cancel stuck jobs, clear partials if needed, submit split benchmark + merge.
set -euo pipefail
cd /scratch/users/linika

CKPT="${1:-checkpoints/model_epoch_50.pth}"

echo "Cancelling redundant pending jobs (curriculum pct100 + monolithic benchmark)..."
scancel 1576826 1576827 2>/dev/null || true

PARTS_DIR=/scratch/users/linika/checkpoints/benchmark/parts
# Keep finished parts; only remove if REPART=1
if [[ "${REPART:-0}" == "1" ]]; then
  rm -f "${PARTS_DIR}"/*.json
  echo "Cleared ${PARTS_DIR}"
fi
mkdir -p "${PARTS_DIR}" checkpoints/logs checkpoints/benchmark

ARRAY_ID=$(sbatch --parsable run_benchmark_array.sbatch "${CKPT}")
echo "Submitted benchmark array: ${ARRAY_ID} (8 conditions, 2 CPU / 8G / 1 GPU each)"

MERGE_ID=$(sbatch --parsable --dependency="afterok:${ARRAY_ID}" run_benchmark_merge.sbatch)
echo "Submitted merge job: ${MERGE_ID} (after array ${ARRAY_ID})"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Parts:    ls -la ${PARTS_DIR}/"
echo "Final:    checkpoints/benchmark/final_results.json"
