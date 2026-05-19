# BraTS 2021 Swin UNETR (FarmShare)

Swin UNETR training and evaluation for BraTS 2021 brain tumor segmentation on Stanford FarmShare GPU nodes.

## Layout

| Path | Description |
|------|-------------|
| `Copy_of_swin_unetr_brats21_segmentation_3d.py` | Main training script |
| `eval_sensitivity_hd95.py` | Validation sensitivity + HD95 |
| `eval_comprehensive.py` | FP, ECE, Brier, surface Dice, ASSD, timing |
| `plot_training_figures.py` | Loss/Dice curves and qualitative figures |
| `make_curriculum_splits.py` | Build 10% / 25% / 50% curriculum JSON splits |
| `brats21_80_20.json` | Train/val split (paths relative to data root) |
| `curriculum_splits/` | Generated subset JSONs |
| `run_*.sbatch`, `run_*.sh`, `*.slurm` | Slurm / shell launchers |

## Data

Place BraTS 2021 training NIfTIs under `TrainingData/` (not in git). Set `MONAI_DATA_DIRECTORY` or use default `/scratch/users/linika`.

## Environment

```bash
conda activate swin   # PyTorch + MONAI; pip install scipy for HD95
```

## Training

```bash
export BRATS_FULL_TRAIN=1 BRATS_MAX_EPOCHS=50 BRATS_VAL_EVERY=5
python Copy_of_swin_unetr_brats21_segmentation_3d.py
```

Curriculum (10% → 25% → 50%): `sbatch run_curriculum_train.sbatch`

## Evaluation

```bash
sbatch run_eval.sh
sbatch run_eval_comprehensive.sh
```

## Git / GitHub

Branch: `swin_linika` on shared repo `cs231n` (code only; data and checkpoints are gitignored).

```bash
cd /scratch/users/linika
git push -u origin swin_linika
```

Remote: `git@github.com:cs231n/cs231n.git`
