#!/usr/bin/env python3

from pathlib import Path
from typing import cast
import nibabel as nib
from nibabel.spatialimages import SpatialImage
import numpy as np


def load_seg(path: str | Path) -> np.ndarray:
    # nib.load() is typed as -> FileBasedImage, but every real image file
    # (NIfTI, MGH, …) returns a SpatialImage, which has get_fdata().
    img = cast(SpatialImage, nib.load(str(path)))
    return img.get_fdata().astype(np.uint8)


def nnunet_to_brats_regions(seg: np.ndarray) -> np.ndarray:
    tc = np.logical_or(seg == 2, seg == 3)
    wt = np.logical_or.reduce([seg == 1, seg == 2, seg == 3])
    et = seg == 3

    return np.stack([tc, wt, et], axis=0).astype(np.float32)


def load_prediction_and_gt(
    pred_path: str | Path,
    gt_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    pred = load_seg(pred_path)
    gt = load_seg(gt_path)

    return (
        nnunet_to_brats_regions(pred),
        nnunet_to_brats_regions(gt),
    )