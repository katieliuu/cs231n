#!/bin/bash
#SBATCH --job-name=brats_curriculum
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/users/linika/checkpoints/logs/curriculum_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/logs/curriculum_%j.err

set -euo pipefail
cd /scratch/users/linika
mkdir -p checkpoints/logs checkpoints/curriculum

SWIN_PY=/home/users/linika/miniconda3/envs/swin/bin/python
if [[ ! -x "${SWIN_PY}" ]]; then
  echo "Missing ${SWIN_PY}"
  exit 1
fi

"${SWIN_PY}" -c "import torch; assert torch.cuda.is_available()" || exit 1
"${SWIN_PY}" make_curriculum_splits.py

export BRATS_FULL_TRAIN=1
export BRATS_MAX_EPOCHS=50
export BRATS_VAL_EVERY=5
export BRATS_MPL_BACKEND=Agg
export BRATS_SKIP_PLOTS=1
export REQUIRE_CUDA=1
export MONAI_DATA_DIRECTORY=/scratch/users/linika
TRAIN_PY=/scratch/users/linika/Copy_of_swin_unetr_brats21_segmentation_3d.py

run_stage() {
  local stage="$1"
  local pct="$2"
  local split_json="$3"
  local ckpt_dir="$4"
  local resume_ckpt="${5:-}"

  echo ""
  echo "========== Curriculum stage ${stage} (${pct}% train) $(date) =========="
  export BRATS21_SPLIT_JSON="${split_json}"
  export BRATS_CHECKPOINT_DIR="${ckpt_dir}"
  export BRATS_CURRICULUM_STAGE="${stage}"
  export BRATS_LOG_FILE="/scratch/users/linika/checkpoints/logs/curriculum_${stage}_job${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
  unset BRATS_RESUME BRATS_RESUME_FILE_EPOCH BRATS_RESUME_CHECKPOINT BRATS_FINETUNE_ONLY

  if [[ -n "${resume_ckpt}" ]]; then
    export BRATS_RESUME_CHECKPOINT="${resume_ckpt}"
    export BRATS_FINETUNE_ONLY=1
    echo "Finetune from ${resume_ckpt}"
  fi

  mkdir -p "${ckpt_dir}"
  "${SWIN_PY}" "${TRAIN_PY}"
}

BASE=/scratch/users/linika/curriculum_splits
CKPT_BASE=/scratch/users/linika/checkpoints/curriculum

run_stage "pct10" "10" "${BASE}/brats21_train_10pct.json" "${CKPT_BASE}/pct10"
run_stage "pct25" "25" "${BASE}/brats21_train_25pct.json" "${CKPT_BASE}/pct25" "${CKPT_BASE}/pct10/model_epoch_50.pth"
run_stage "pct50" "50" "${BASE}/brats21_train_50pct.json" "${CKPT_BASE}/pct50" "${CKPT_BASE}/pct25/model_epoch_50.pth"

echo ""
echo "========== Curriculum Dice summary =========="
"${SWIN_PY}" <<'PY'
import json
from pathlib import Path

root = Path("/scratch/users/linika/checkpoints/curriculum")
rows = []
for stage, pct in [("pct10", 10), ("pct25", 25), ("pct50", 50)]:
    p = root / stage / "curriculum_dice.json"
    if not p.is_file():
        print(f"MISSING {p}")
        continue
    d = json.loads(p.read_text())
    rows.append(d)
    print(
        f"{pct}% train (n={d['n_train']}): "
        f"Dice TC={d['final_dice_tc']:.4f} WT={d['final_dice_wt']:.4f} "
        f"ET={d['final_dice_et']:.4f} Mean={d['final_dice_avg']:.4f} "
        f"(best mean={d['best_dice_avg']:.4f})"
    )

out = Path("/scratch/users/linika/checkpoints/curriculum/all_stages_dice.json")
out.write_text(json.dumps(rows, indent=2))
print(f"Wrote {out}")
PY

echo "Done $(date)"
