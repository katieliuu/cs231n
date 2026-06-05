#!/bin/bash

export nnUNet_raw="${nnUNet_raw:-/root/shared/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/root/shared/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/root/shared/nnUNet_results}"

DATASET_ID=137

nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity

for FOLD in 0 1 2 3 4
 do
    nnUNetv2_train ${DATASET_ID} 3d_fullres ${FOLD}
 done