#!/usr/bin/env python3
"""
Robustness evaluation for nnU-Net on BraTS21.

For each condition, modifies the input NIfTI images on the fly before
running nnUNetv2_predict, then computes Dice against GT.

The key difference for nnU-Net is that we don't have sigmoid probabilities 
(it outputs hard labels), so ECE/Brier will be approximate, and robustness 
requires re-running inference with modified inputs rather than just changing the transform pipeline.

Conditions:
  - baseline (already in preds/, just scores)
  - without_FLAIR / without_T1ce / without_T1 / without_T2
  - gaussian_noise
  - motion_blur
  - bias_field

BraTS channel order in nnU-Net imagesTr:
  _0000 = FLAIR, _0001 = T1ce, _0002 = T2, _0003 = T1
  (as set up by prepare_brats21_nnunet.py)
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.metrics.meandice import DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import gaussian_filter

from nnunet_inference_utils import load_prediction_and_gt

# ── config ────────────────────────────────────────────────────────────────────

IMAGE_DIR = Path(os.environ.get(
    "NNUNET_IMAGE_DIR",
    "/root/shared/nnUNet_raw/Dataset137_BraTS2021/imagesTr",
))
GT_DIR = Path(os.environ.get(
    "NNUNET_GT_DIR",
    "/root/shared/nnUNet_raw/Dataset137_BraTS2021/labelsTr",
))
RESULTS_DIR = Path(os.environ.get(
    "nnUNet_results",
    "/root/shared/nnUNet_results",
))
OUT_BASE = Path(os.environ.get("NNUNET_ROBUSTNESS_DIR", "/root/shared/robustness_preds"))

# BraTS suffix → channel index
MODALITY_SUFFIX = {
    "without_FLAIR":  "_0000",
    "without_T1ce":   "_0001",
    "without_T2":     "_0002",
    "without_T1":     "_0003",
}

PREDICT_CMD = [
    "nnUNetv2_predict",
    "-d", "137",
    "-c", "3d_fullres",
    "-f", "0", "1", "2", "3", "4",
    "-chk", "checkpoint_best.pth",
    "--disable_progress_bar",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _mean_finite(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _copy_and_perturb(src_dir: Path, dst_dir: Path, perturb_fn) -> None:
    """Copy all NIfTI files from src_dir to dst_dir, applying perturb_fn to each."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for nii_path in sorted(src_dir.glob("*.nii.gz")):
        img = nib.load(str(nii_path))
        data = img.get_fdata(dtype=np.float32)
        data = perturb_fn(data, nii_path.name)
        nib.save(nib.Nifti1Image(data, img.affine, img.header), str(dst_dir / nii_path.name))


def _zero_channel(suffix: str):
    """Return a perturb_fn that zeros the channel with the given suffix."""
    def fn(data: np.ndarray, name: str) -> np.ndarray:
        if name.endswith(suffix + ".nii.gz"):
            return np.zeros_like(data)
        return data
    return fn


def _gaussian_noise(data: np.ndarray, name: str, std: float = 0.1) -> np.ndarray:
    return data + np.random.normal(0, std, data.shape).astype(np.float32)


def _motion_blur(data: np.ndarray, name: str, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter(data, sigma=sigma).astype(np.float32)


def _bias_field(data: np.ndarray, name: str, degree: int = 3, coeff_range: float = 0.3) -> np.ndarray:
    shape = data.shape
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    bias = np.ones(shape, dtype=np.float32)
    rng = np.random.default_rng(42)
    for d in range(1, degree + 1):
        for g in grid:
            coeff = rng.uniform(-coeff_range, coeff_range)
            bias += coeff * (g ** d).astype(np.float32)
    return (data * bias).astype(np.float32)


def _run_predict(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = PREDICT_CMD + ["-i", str(input_dir), "-o", str(output_dir)]
    subprocess.run(cmd, check=True)


def _score_predictions(pred_dir: Path) -> dict[str, float]:
    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    tc_list, wt_list, et_list = [], [], []
    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        gt_path = GT_DIR / pred_path.name
        if not gt_path.exists():
            continue
        pred, gt = load_prediction_and_gt(pred_path, gt_path)
        pred_t = torch.tensor(pred)[None]
        gt_t = torch.tensor(gt)[None]
        dice_metric.reset()
        dice_metric(y_pred=pred_t, y=gt_t)
        d, _ = dice_metric.aggregate()
        d = d.flatten().cpu().numpy()
        tc_list.append(float(d[0]))
        wt_list.append(float(d[1]))
        et_list.append(float(d[2]))
    return {
        "dice_tc": _mean_finite(tc_list),
        "dice_wt": _mean_finite(wt_list),
        "dice_et": _mean_finite(et_list),
        "dice_avg": _mean_finite(tc_list + wt_list + et_list),
        "n": len(tc_list),
    }


# ── conditions ────────────────────────────────────────────────────────────────

CONDITIONS: dict[str, object] = {
    **{name: _zero_channel(suffix) for name, suffix in MODALITY_SUFFIX.items()},
    "gaussian_noise": _gaussian_noise,
    "motion_blur": _motion_blur,
    "bias_field": _bias_field,
}

# ── main ──────────────────────────────────────────────────────────────────────

# ── main ──────────────────────────────────────────────────────────────────────

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--condition",
    type=str,
    default=None,
    help="Run a single condition (e.g. without_T1). Default: run all.",
)
_args = _parser.parse_args()

if _args.condition is not None:
    if _args.condition not in CONDITIONS:
        raise SystemExit(
            f"Unknown condition '{_args.condition}'. "
            f"Valid: {list(CONDITIONS)}"
        )
    run_conditions = {
        _args.condition: CONDITIONS[_args.condition]
    }
else:
    run_conditions = CONDITIONS

results: dict[str, dict] = {}

for condition_name, perturb_fn in run_conditions.items():
    print(f"\n=== {condition_name} ===", flush=True)
    perturbed_dir = OUT_BASE / condition_name / "images"
    pred_dir      = OUT_BASE / condition_name / "preds"

    print(f"  Perturbing images → {perturbed_dir}", flush=True)
    _copy_and_perturb(IMAGE_DIR, perturbed_dir, perturb_fn)

    print(f"  Running nnUNetv2_predict → {pred_dir}", flush=True)
    _run_predict(perturbed_dir, pred_dir)

    print(f"  Scoring...", flush=True)
    scores = _score_predictions(pred_dir)
    results[condition_name] = scores
    print(f"  {scores}", flush=True)

print("\n===== Robustness Summary =====")
for cond, scores in results.items():
    print(f"  {cond:20s}  dice_tc={scores['dice_tc']:.4f}  dice_wt={scores['dice_wt']:.4f}  "
          f"dice_et={scores['dice_et']:.4f}  dice_avg={scores['dice_avg']:.4f}")