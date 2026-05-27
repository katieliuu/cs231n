#!/bin/bash
# Cancel pending array tasks (incl. bias_field), run without_* conditions, merge without bias_field.
set -euo pipefail
cd /scratch/users/linika

CKPT="${1:-checkpoints/model_epoch_50.pth}"

echo "Cancelling pending array tasks and old merge..."
scancel 1576864_3 1576864_4 1576864_5 1576864_6 1576864_7 1576865 2>/dev/null || true

MODALITIES=(without_FLAIR without_T1ce without_T1 without_T2)
JOB_IDS=()

for COND in "${MODALITIES[@]}"; do
  if [[ -f "checkpoints/benchmark/parts/${COND}.json" ]]; then
    echo "Skip ${COND} — part already exists"
    continue
  fi
  JID=$(sbatch --parsable --export=ALL,COND="${COND}" run_benchmark_one.sbatch "${CKPT}")
  JOB_IDS+=("${JID}")
  echo "Submitted ${COND}: ${JID}"
done

if [[ ${#JOB_IDS[@]} -eq 0 ]]; then
  echo "All modality parts already present."
  MERGE_ID=$(sbatch --parsable run_benchmark_merge.sbatch --skip bias_field)
  echo "Submitted merge only: ${MERGE_ID}"
else
  DEP="afterok:$(IFS=:; echo "${JOB_IDS[*]}")"
  MERGE_ID=$(sbatch --parsable --dependency="${DEP}" run_benchmark_merge.sbatch --skip bias_field)
  echo "Submitted merge: ${MERGE_ID} (${DEP})"
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Parts:    ls -la checkpoints/benchmark/parts/"
