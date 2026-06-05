#!/usr/bin/env python3
import os
import shutil

import SimpleITK as sitk
import numpy as np

# Replace wildcard import with explicit names actually used in this file.
# The wildcard also re-exported `os`, which is now imported directly above.
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    maybe_mkdir_p,
    subdirs,
)
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

BRATS_DATA_DIR = os.environ.get("BRATS_DATA_DIR", "/root/shared/TrainingData")
TASK_ID = 137
TASK_NAME = "BraTS2021"


def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2

    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


if __name__ == "__main__":
    # nnUNet_raw is typed str | None; assert early so the rest of the file
    # can treat it as str without per-call casts.
    assert nnUNet_raw is not None, (
        "nnUNet_raw is not set. Export the nnUNet_raw environment variable."
    )

    foldername = f"Dataset{TASK_ID:03d}_{TASK_NAME}"

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(BRATS_DATA_DIR, prefix="BraTS", join=False)
    for c in case_ids:
        shutil.copy(join(BRATS_DATA_DIR, c, c + "_t1.nii.gz"),    join(imagestr, c + "_0000.nii.gz"))
        shutil.copy(join(BRATS_DATA_DIR, c, c + "_t1ce.nii.gz"),  join(imagestr, c + "_0001.nii.gz"))
        shutil.copy(join(BRATS_DATA_DIR, c, c + "_t2.nii.gz"),    join(imagestr, c + "_0002.nii.gz"))
        shutil.copy(join(BRATS_DATA_DIR, c, c + "_flair.nii.gz"), join(imagestr, c + "_0003.nii.gz"))
        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
            join(BRATS_DATA_DIR, c, c + "_seg.nii.gz"),
            join(labelstr, c + ".nii.gz"),
        )

    generate_dataset_json(
        out_base,
        channel_names={
            0: "T1",
            1: "T1ce",
            2: "T2",
            3: "Flair",
        },
        labels={
            "background": 0,
            "whole tumor": (1, 2, 3),
            "tumor core": (2, 3),
            "enhancing tumor": (3,),
        },
        num_training_cases=len(case_ids),
        file_ending=".nii.gz",
        regions_class_order=(1, 2, 3),
    )

    print(f"Prepared nnU-Net dataset at {out_base}")