#!/usr/bin/env python3
"""
Patch nnU-Net plans.json to reduce model size by lowering UNet_base_num_features.

Run after nnUNetv2_plan_and_preprocess, before nnUNetv2_train.

Default base_num_features=32 → ~31M params
Setting base_num_features=8  → ~6M params (closest to 7M target)
"""
import json
import os
import shutil
from pathlib import Path

TARGET_BASE_FEATURES = 8  # ~6M params, closest to 7M target
DATASET_ID = "137"
PLANS_NAME = "nnUNetPlans"

preprocessed_dir = Path(os.environ.get(
    "nnUNet_preprocessed",
    "/root/shared/nnUNet_preprocessed",
))
plans_path = preprocessed_dir / f"Dataset{DATASET_ID}_BraTS2021" / f"{PLANS_NAME}.json"

if not plans_path.exists():
    raise FileNotFoundError(
        f"Plans file not found at {plans_path}. "
        "Run nnUNetv2_plan_and_preprocess first."
    )

# Back up original
backup = plans_path.with_suffix(".json.orig")
if not backup.exists():
    shutil.copy(plans_path, backup)
    print(f"Backed up original plans to {backup}")

with open(plans_path) as f:
    plans = json.load(f)

# Patch 3d_fullres configuration
config = plans["configurations"]["3d_fullres"]
old_val = config.get("UNet_base_num_features", 32)
config["UNet_base_num_features"] = TARGET_BASE_FEATURES

with open(plans_path, "w") as f:
    json.dump(plans, f, indent=2)

print(f"Patched {plans_path}")
print(f"  UNet_base_num_features: {old_val} → {TARGET_BASE_FEATURES}")
print(f"  Expected params: ~6M (target: ~7M)")