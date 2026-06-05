#!/bin/bash

export nnUNet_raw="${nnUNet_raw:-/root/shared/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/root/shared/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/root/shared/nnUNet_results}"

nnUNetv2_predict \
    -i ${NNUNET_INPUT_DIR:-/root/shared/nnUNet_raw/Dataset137_BraTS2021/imagesTr} \
    -o ${NNUNET_OUTPUT_DIR:-/root/shared/preds} \
    -d 137 \
    -c 3d_fullres \
    -f all