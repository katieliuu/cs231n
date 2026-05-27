# BraTS 2021 Swin UNETR (FarmShare)

Swin UNETR training and evaluation for BraTS 2021 brain tumor segmentation on Stanford FarmShare GPU nodes.

## Files

| Path | Description |
|------|-------------|
| `Copy_of_swin_unetr_brats21_segmentation_3d.py` | Main 3D training script (Swin UNETR, sliding-window validation) |
| `eval_sensitivity_hd95.py` | Per-region sensitivity and HD95 on the validation set |
| `eval_comprehensive.py` | FP/case, ECE, Brier, surface Dice, ASSD, inference timing |
| `eval_full_benchmark.py` | Full benchmark: curriculum Dice, noise/blur/bias, missing modalities |
| `merge_benchmark_results.py` | Merge Slurm benchmark shard JSONs into `final_results` |
| `plot_training_figures.py` | Loss/Dice curves and example segmentations |
| `make_curriculum_splits.py` | Build 10% / 25% / 50% / 100% curriculum JSON splits |
| `brats21_80_20.json`, `brats21_actual_80_20.json` | Train/val case lists |
| `curriculum_splits/` | Subset JSONs (`brats21_train_*pct.json`) |
| `run_*.sbatch`, `run_*.sh`, `*.slurm` | Slurm/shell launchers (train, eval, curriculum, benchmark) |

**Launchers (brief):** `run_brats_swin_l4.sbatch` / `train_brats_l4.slurm` — full training; `run_curriculum_*.sbatch` — curriculum stages; `run_eval*.sh` — sensitivity/HD95 and comprehensive metrics; `run_full_benchmark.sbatch` + `run_benchmark_*.sbatch` — robustness/modality ablations; `submit_benchmark_*.sh` — queue benchmark jobs.

## Environment

```bash
conda activate swin   # PyTorch + MONAI; scipy for HD95
```

## Training & evaluation

```bash
export BRATS_FULL_TRAIN=1 BRATS_MAX_EPOCHS=50 BRATS_VAL_EVERY=5
python Copy_of_swin_unetr_brats21_segmentation_3d.py

sbatch run_curriculum_train.sbatch      # 10% → 25% → 50%
sbatch run_curriculum_pct100.sbatch     # 100% curriculum stage
sbatch run_eval.sh
sbatch run_eval_comprehensive.sh
sbatch run_full_benchmark.sbatch
```
