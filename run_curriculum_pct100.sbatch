#!/bin/bash
# Curriculum: 100% only (10%, 25%, 50% already complete).
#SBATCH --job-name=brats_cur100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# FarmShare: MaxMemPerCPU=4000MB → 32G with 8 CPUs = 4G/CPU (scheduling-friendly).
# Still 1 GPU (--gres=gpu:1 below); only CPU/RAM request is reduced.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/users/linika/checkpoints/logs/curriculum_100_%j.out
#SBATCH --error=/scratch/users/linika/checkpoints/logs/curriculum_100_%j.err

set -euo pipefail
cd /scratch/users/linika
mkdir -p checkpoints/logs checkpoints/curriculum

SWIN_PY=/home/users/linika/miniconda3/envs/swin/bin/python
CKPT_BASE=/scratch/users/linika/checkpoints/curriculum
PCT50_CKPT="${CKPT_BASE}/pct50/model_epoch_50.pth"

if [[ ! -f "${PCT50_CKPT}" ]]; then
  echo "Missing 50% checkpoint: ${PCT50_CKPT}"
  exit 1
fi

BASE=/scratch/users/linika/curriculum_splits
PCT100_JSON="${BASE}/brats21_train_100pct.json"
if [[ ! -f "${PCT100_JSON}" ]]; then
  echo "Missing 100% split JSON: ${PCT100_JSON}"
  echo "Run make_curriculum_splits.py once, or restore curriculum_splits/ from backup."
  exit 1
fi
echo "Using existing split: ${PCT100_JSON}"

"${SWIN_PY}" -c "import torch; assert torch.cuda.is_available()" || exit 1

export BRATS_FULL_TRAIN=1
export BRATS_MAX_EPOCHS=50
export BRATS_VAL_EVERY=5
export BRATS_MPL_BACKEND=Agg
export BRATS_SKIP_PLOTS=1
export REQUIRE_CUDA=1
export MONAI_DATA_DIRECTORY=/scratch/users/linika
TRAIN_PY=/scratch/users/linika/Copy_of_swin_unetr_brats21_segmentation_3d.py
if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "Missing training script: ${TRAIN_PY}"
  exit 1
fi

echo "========== Curriculum stage pct100 (100% train) $(date) =========="
export BRATS21_SPLIT_JSON="${BASE}/brats21_train_100pct.json"
export BRATS_CHECKPOINT_DIR="${CKPT_BASE}/pct100"
export BRATS_CURRICULUM_STAGE="pct100"
export BRATS_LOG_FILE="/scratch/users/linika/checkpoints/logs/curriculum_pct100_job${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"
export BRATS_RESUME_CHECKPOINT="${PCT50_CKPT}"
export BRATS_FINETUNE_ONLY=1
echo "Finetune from ${PCT50_CKPT}"

mkdir -p "${CKPT_BASE}/pct100"
"${SWIN_PY}" "${TRAIN_PY}"

echo ""
echo "========== Curriculum Dice summary =========="
"${SWIN_PY}" <<'PY'
import json
from pathlib import Path

root = Path("/scratch/users/linika/checkpoints/curriculum")
rows = []
for stage, pct in [("pct10", 10), ("pct25", 25), ("pct50", 50), ("pct100", 100)]:
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
