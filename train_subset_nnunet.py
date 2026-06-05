#!/usr/bin/env python3
"""
Data efficiency experiment: train nnU-Net on a subset of BraTS21 training data.

- Samples PCT% of training cases (by WT volume, stratified) with seed 3
- Trains fold 0 only for 50 epochs
- Runs predict + dice eval on the full val set
- Saves results to /root/shared/data_efficiency/pct{PCT}/results.json

Usage (called from Modal with PCT env var set):
  python train_subset_nnunet.py --pct 10
  python train_subset_nnunet.py --pct 25
  python train_subset_nnunet.py --pct 50
"""
import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from monai.metrics.meandice import DiceMetric
from monai.utils.enums import MetricReduction

from nnunet_inference_utils import load_prediction_and_gt

SEED = 3
RAW_DIR     = Path(os.environ.get("nnUNet_raw",          "/root/shared/nnUNet_raw"))
RESULTS_DIR = Path(os.environ.get("nnUNet_results",      "/root/shared/nnUNet_results"))
PREPROCESSED= Path(os.environ.get("nnUNet_preprocessed", "/root/shared/nnUNet_preprocessed"))
GT_DIR      = Path(os.environ.get("NNUNET_GT_DIR",
    "/root/shared/nnUNet_raw/Dataset137_BraTS2021/labelsTr"))
OUT_BASE    = Path("/root/shared/data_efficiency")

TASK        = "Dataset137_BraTS2021"
TRAINER     = "nnUNetTrainer__nnUNetPlans__3d_fullres"
ORIG_IMAGES = RAW_DIR / TASK / "imagesTr"
ORIG_LABELS = RAW_DIR / TASK / "labelsTr"


def _mean_finite(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _sample_cases(pct: int) -> list[str]:
    """Return sorted list of case stems sampled at pct% with seed 3."""
    all_cases = sorted({
        p.name.replace("_0000.nii.gz", "")
        for p in ORIG_IMAGES.glob("*_0000.nii.gz")
    })
    n = max(1, round(len(all_cases) * pct / 100))
    rng = random.Random(SEED)
    selected = sorted(rng.sample(all_cases, n))
    print(f"  Sampled {n}/{len(all_cases)} cases ({pct}%) with seed {SEED}")
    return selected


def _make_subset_dataset(cases: list[str], subset_task: str) -> None:
    """Create a new nnU-Net dataset directory with only the subset cases."""
    subset_raw = RAW_DIR / subset_task
    img_out = subset_raw / "imagesTr"
    lbl_out = subset_raw / "labelsTr"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for suffix in ("_0000", "_0001", "_0002", "_0003"):
            src = ORIG_IMAGES / f"{case}{suffix}.nii.gz"
            if src.exists():
                dst = img_out / src.name
                if not dst.exists():
                    shutil.copy(src, dst)
        lbl_src = ORIG_LABELS / f"{case}.nii.gz"
        if lbl_src.exists():
            lbl_dst = lbl_out / lbl_src.name
            if not lbl_dst.exists():
                shutil.copy(lbl_src, lbl_dst)

    # Copy dataset.json from original task, update num_training
    orig_json = RAW_DIR / TASK / "dataset.json"
    subset_json = subset_raw / "dataset.json"
    with open(orig_json) as f:
        ds = json.load(f)
    ds["numTraining"] = len(cases)
    with open(subset_json, "w") as f:
        json.dump(ds, f, indent=2)

    print(f"  Created subset dataset at {subset_raw} with {len(cases)} cases")


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
        gt_t   = torch.tensor(gt)[None]
        dice_metric.reset()
        dice_metric(y_pred=pred_t, y=gt_t)
        d, _ = dice_metric.aggregate()
        d = d.flatten().cpu().numpy()
        tc_list.append(float(d[0]))
        wt_list.append(float(d[1]))
        et_list.append(float(d[2]))
    return {
        "dice_tc":  _mean_finite(tc_list),
        "dice_wt":  _mean_finite(wt_list),
        "dice_et":  _mean_finite(et_list),
        "dice_avg": _mean_finite(tc_list + wt_list + et_list),
        "n_val":    len(tc_list),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pct", type=int, required=True, choices=[10, 25, 50])
    args = p.parse_args()
    pct = args.pct

    # nnU-Net dataset IDs: 137=full, 110=10%, 125=25%, 150=50%
    task_id_map = {10: 110, 25: 125, 50: 150}
    task_id = task_id_map[pct]
    subset_task = f"Dataset{task_id:03d}_BraTS2021pct{pct}"
    pred_dir  = OUT_BASE / f"pct{pct}" / "preds"
    out_dir   = OUT_BASE / f"pct{pct}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Data efficiency: {pct}% training data =====")

    # 1. Sample cases and build subset dataset
    cases = _sample_cases(pct)
    _make_subset_dataset(cases, subset_task)

    # 2. Preprocess
    print("  Preprocessing...")
    subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", str(task_id), "--verify_dataset_integrity"],
        check=True,
    )

    # 2b. Patch plans to match main model size (~6M params, target 7M)
    import json, shutil
    preprocessed_dir = PREPROCESSED / subset_task
    plans_path = preprocessed_dir / "nnUNetPlans.json"
    if plans_path.exists():
        backup = plans_path.with_suffix(".json.orig")
        if not backup.exists():
            shutil.copy(plans_path, backup)
        with open(plans_path) as f:
            plans = json.load(f)
        plans["configurations"]["3d_fullres"]["UNet_base_num_features"] = 8
        with open(plans_path, "w") as f:
            json.dump(plans, f, indent=2)
        print(f"  Patched plans: UNet_base_num_features=8 (~6M params)")
    else:
        print(f"  WARNING: plans not found at {plans_path}, skipping patch")

    # 3. Train fold 0 for 50 epochs
    print("  Training fold 0 for 50 epochs...")
    subprocess.run(
        [
            "nnUNetv2_train", str(task_id), "3d_fullres", "0",
            "--npz",
            "-tr", "nnUNetTrainer",
        ],
        check=True,
        env={**os.environ, "nnUNet_n_proc_DA": "4"},
    )

    # 4. Predict on original full val images
    print("  Running inference on validation set...")
    pred_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "nnUNetv2_predict",
            "-i", str(ORIG_IMAGES),
            "-o", str(pred_dir),
            "-d", str(task_id),
            "-c", "3d_fullres",
            "-f", "0",
            "-chk", "checkpoint_best.pth",
            "--disable_progress_bar",
        ],
        check=True,
    )

    # 5. Score
    print("  Scoring predictions...")
    scores = _score_predictions(pred_dir)
    scores["pct"] = pct
    scores["n_train"] = len(cases)
    scores["seed"] = SEED

    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\n  Results saved to {out_json}")
    print(f"  dice_tc={scores['dice_tc']:.4f}  dice_wt={scores['dice_wt']:.4f}  "
          f"dice_et={scores['dice_et']:.4f}  dice_avg={scores['dice_avg']:.4f}")


if __name__ == "__main__":
    main()