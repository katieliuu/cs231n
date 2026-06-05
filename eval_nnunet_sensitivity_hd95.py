#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import torch
from monai.metrics.confusion_matrix import ConfusionMatrixMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.utils.enums import MetricReduction

from nnunet_inference_utils import load_prediction_and_gt


def _mean(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


PRED_DIR = Path(os.environ.get("NNUNET_PRED_DIR", "/root/shared/preds"))
GT_DIR = Path(
    os.environ.get(
        "NNUNET_GT_DIR",
        "/root/shared/nnUNet_raw/Dataset137_BraTS2021/labelsTr",
    )
)

sens_metric = ConfusionMatrixMetric(
    include_background=True,
    metric_name="sensitivity",
    reduction=MetricReduction.MEAN_BATCH,
    get_not_nans=True,
)
dice_metric = DiceMetric(
    include_background=True,
    reduction=MetricReduction.MEAN_BATCH,
    get_not_nans=True,
)
hd_metric = HausdorffDistanceMetric(
    include_background=True,
    percentile=95,
    reduction=MetricReduction.MEAN_BATCH,
    get_not_nans=True,
)

sens_tc: list[float] = []
sens_wt: list[float] = []
sens_et: list[float] = []
dice_tc: list[float] = []
dice_wt: list[float] = []
dice_et: list[float] = []
hd_tc: list[float] = []
hd_wt: list[float] = []
hd_et: list[float] = []

for pred_path in sorted(PRED_DIR.glob("*.nii.gz")):
    gt_path = GT_DIR / pred_path.name
    pred, gt = load_prediction_and_gt(pred_path, gt_path)
    pred_t = torch.tensor(pred)[None]
    gt_t = torch.tensor(gt)[None]

    # ConfusionMatrixMetric returns list[tuple[Tensor, Tensor]] → need [0]
    sens_metric.reset()
    sens_metric(y_pred=pred_t, y=gt_t)
    s, _ = sens_metric.aggregate()[0]
    s = s.flatten().cpu().numpy()
    sens_tc.append(float(s[0]))
    sens_wt.append(float(s[1]))
    sens_et.append(float(s[2]))

    # DiceMetric returns tuple[Tensor, Tensor] directly → no [0]
    dice_metric.reset()
    dice_metric(y_pred=pred_t, y=gt_t)
    d, _ = dice_metric.aggregate()
    d = d.flatten().cpu().numpy()
    dice_tc.append(float(d[0]))
    dice_wt.append(float(d[1]))
    dice_et.append(float(d[2]))

    # HausdorffDistanceMetric returns tuple[Tensor, Tensor] directly → no [0]
    hd_metric.reset()
    hd_metric(y_pred=pred_t, y=gt_t)
    h, _ = hd_metric.aggregate()
    h = h.flatten().cpu().numpy()
    hd_tc.append(float(h[0]) if np.isfinite(h[0]) else float("nan"))
    hd_wt.append(float(h[1]) if np.isfinite(h[1]) else float("nan"))
    hd_et.append(float(h[2]) if np.isfinite(h[2]) else float("nan"))

results = {
    "dice_tc": _mean(dice_tc),
    "dice_wt": _mean(dice_wt),
    "dice_et": _mean(dice_et),
    "dice_avg": _mean(dice_tc + dice_wt + dice_et),
    "sens_tc": _mean(sens_tc),
    "sens_wt": _mean(sens_wt),
    "sens_et": _mean(sens_et),
    "sens_avg": _mean(sens_tc + sens_wt + sens_et),
    "hd95_tc": _mean(hd_tc),
    "hd95_wt": _mean(hd_wt),
    "hd95_et": _mean(hd_et),
    "hd95_avg": _mean(hd_tc + hd_wt + hd_et),
}

for k, v in results.items():
    print(f"  {k}: {v:.4f}")