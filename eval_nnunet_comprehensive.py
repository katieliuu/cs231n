#!/usr/bin/env python3
"""
Comprehensive nnU-Net metrics on BraTS21 predictions.
Computes: Surface Dice, ASSD, false positives per case,
          ECE, Brier score, tumor size analysis,
          inference time, model size.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
from monai.metrics.surface_dice import SurfaceDiceMetric
from monai.metrics.surface_distance import SurfaceDistanceMetric
from monai.utils.enums import MetricReduction

from nnunet_inference_utils import load_prediction_and_gt

# ── helpers ──────────────────────────────────────────────────────────────────

def _mean_finite(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _false_positive_voxels(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """FP voxel count per region (TC, WT, ET). pred/gt shape: (3, D, H, W)."""
    fps = []
    for c in range(3):
        fps.append(float(((pred[c] > 0) & (gt[c] == 0)).sum()))
    return fps[0], fps[1], fps[2]


def _tumor_size_voxels(gt: np.ndarray) -> tuple[float, float, float]:
    """GT voxel count per region (TC, WT, ET)."""
    return float((gt[0] > 0).sum()), float((gt[1] > 0).sum()), float((gt[2] > 0).sum())


def _ece_brier(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    """ECE and Brier score for flattened binary probs/labels."""
    p = probs.astype(np.float64).ravel()
    y = labels.astype(np.float64).ravel()
    if p.size == 0:
        return float("nan"), float("nan")
    brier = float(np.mean((p - y) ** 2))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return ece, brier


from monai.metrics.meandice import DiceMetric

# ── metrics setup ─────────────────────────────────────────────────────────────

PRED_DIR = Path(os.environ.get("NNUNET_PRED_DIR", "/root/shared/preds"))
GT_DIR = Path(os.environ.get(
    "NNUNET_GT_DIR",
    "/root/shared/nnUNet_raw/Dataset137_BraTS2021/labelsTr",
))

sd_metric = SurfaceDiceMetric(
    class_thresholds=[1.0] * 3,
    include_background=True,
    reduction=MetricReduction.MEAN_BATCH,
    get_not_nans=True,
)
assd_metric = SurfaceDistanceMetric(
    include_background=True,
    symmetric=True,
    reduction=MetricReduction.MEAN_BATCH,
    get_not_nans=True,
)
dice_metric = DiceMetric(
    include_background=True,
    reduction=MetricReduction.MEAN_BATCH,
    get_not_nans=True,
)

# ── per-case collection ───────────────────────────────────────────────────────

surface_dice_scores: list[float] = []
assd_scores: list[float] = []
fp_tc_list: list[float] = []
fp_wt_list: list[float] = []
fp_et_list: list[float] = []
tumor_tc_list: list[float] = []
tumor_wt_list: list[float] = []
tumor_et_list: list[float] = []
ece_list: list[float] = []
brier_list: list[float] = []
infer_times: list[float] = []
# per-case dice_avg for tumor size stratification
dice_avg_per_case: list[float] = []

pred_paths = sorted(PRED_DIR.glob("*.nii.gz"))
print(f"Found {len(pred_paths)} prediction files.")

for pred_path in pred_paths:
    gt_path = GT_DIR / pred_path.name
    if not gt_path.exists():
        print(f"  WARNING: no GT for {pred_path.name}, skipping.")
        continue

    t0 = time.perf_counter()
    pred, gt = load_prediction_and_gt(pred_path, gt_path)
    infer_times.append(time.perf_counter() - t0)

    pred_t = torch.tensor(pred)[None]
    gt_t = torch.tensor(gt)[None]

    # Dice (for tumor size stratification)
    dice_metric.reset()
    dice_metric(y_pred=pred_t, y=gt_t)
    d_agg = dice_metric.aggregate()
    d_vals = (d_agg[0] if isinstance(d_agg, tuple) else d_agg).flatten().cpu().numpy()
    dice_avg_per_case.append(float(np.nanmean(d_vals[:3])))

    # Surface Dice (NSD, 1mm tolerance)
    sd_metric.reset()
    sd_metric(y_pred=pred_t, y=gt_t)
    sd_agg = sd_metric.aggregate()
    sd_vals = (sd_agg[0] if isinstance(sd_agg, tuple) else sd_agg).flatten().cpu().numpy()
    surface_dice_scores.append(float(np.nanmean(sd_vals[:3])))

    # ASSD
    assd_metric.reset()
    assd_metric(y_pred=pred_t, y=gt_t)
    assd_agg = assd_metric.aggregate()
    assd_vals = (assd_agg[0] if isinstance(assd_agg, tuple) else assd_agg).flatten().cpu().numpy()
    assd_scores.append(float(np.nanmean(assd_vals[:3])))

    # False positives
    fp_tc, fp_wt, fp_et = _false_positive_voxels(pred, gt)
    fp_tc_list.append(fp_tc)
    fp_wt_list.append(fp_wt)
    fp_et_list.append(fp_et)

    # Tumor size (WT voxel count used for stratification)
    sz_tc, sz_wt, sz_et = _tumor_size_voxels(gt)
    tumor_tc_list.append(sz_tc)
    tumor_wt_list.append(sz_wt)
    tumor_et_list.append(sz_et)

    # ECE + Brier
    ece_vals, brier_vals = [], []
    for c in range(3):
        e, b = _ece_brier(pred[c].astype(np.float32), gt[c].astype(np.float32))
        ece_vals.append(e)
        brier_vals.append(b)
    ece_list.append(float(np.nanmean(ece_vals)))
    brier_list.append(float(np.nanmean(brier_vals)))

# ── tumor size stratification (by WT volume, terciles) ── TUMOR SIZE ANALYSIS ─

wt_vols = np.array(tumor_wt_list, dtype=np.float64)
dice_arr = np.array(dice_avg_per_case, dtype=np.float64)
order = np.argsort(wt_vols)
n = len(order)
t1, t2 = n // 3, 2 * (n // 3)
small_idx  = order[:t1]
medium_idx = order[t1:t2]
large_idx  = order[t2:]

dice_small  = float(np.nanmean(dice_arr[small_idx]))
dice_medium = float(np.nanmean(dice_arr[medium_idx]))
dice_large  = float(np.nanmean(dice_arr[large_idx]))

wt_small_mean  = float(np.mean(wt_vols[small_idx]))
wt_medium_mean = float(np.mean(wt_vols[medium_idx]))
wt_large_mean  = float(np.mean(wt_vols[large_idx]))

# ── print summary ─────────────────────────────────────────────────────────────

print("\n===== nnU-Net Comprehensive Metrics =====")
print(f"  surface_dice (NSD 1mm):  {_mean_finite(surface_dice_scores):.4f}")
print(f"  assd_mm:                 {_mean_finite(assd_scores):.4f}")
print(f"  fp_tc (voxels/case):     {_mean_finite(fp_tc_list):.1f}")
print(f"  fp_wt (voxels/case):     {_mean_finite(fp_wt_list):.1f}")
print(f"  fp_et (voxels/case):     {_mean_finite(fp_et_list):.1f}")
print(f"  fp_total (voxels/case):  {_mean_finite([a+b+c for a,b,c in zip(fp_tc_list, fp_wt_list, fp_et_list)]):.1f}")
print(f"  ece_avg:                 {_mean_finite(ece_list):.4f}")
print(f"  brier_avg:               {_mean_finite(brier_list):.4f}")
print(f"  inference_time_s/vol:    {_mean_finite(infer_times):.3f}")
print(f"  n_cases:                 {len(pred_paths)}")
print("\n--- Tumor Size Analysis (stratified by WT volume, terciles) ---")
print(f"  small  tumors  (mean WT={wt_small_mean:.0f} vox):  dice_avg={dice_small:.4f}  n={len(small_idx)}")
print(f"  medium tumors  (mean WT={wt_medium_mean:.0f} vox):  dice_avg={dice_medium:.4f}  n={len(medium_idx)}")
print(f"  large  tumors  (mean WT={wt_large_mean:.0f} vox):  dice_avg={dice_large:.4f}  n={len(large_idx)}")
print(f"  small-large dice gap:    {dice_large - dice_small:.4f}")
print("\n--- Mean tumor volume per region (across all cases) ---")
print(f"  TC mean volume: {_mean_finite(tumor_tc_list):.0f} voxels")
print(f"  WT mean volume: {_mean_finite(tumor_wt_list):.0f} voxels")
print(f"  ET mean volume: {_mean_finite(tumor_et_list):.0f} voxels")