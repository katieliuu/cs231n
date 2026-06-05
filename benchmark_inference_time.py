#!/usr/bin/env python3
"""
Benchmark nnU-Net inference time and peak GPU memory per volume.
Runs nnUNetv2_predict on a small subset of cases and times each one.
"""
import os
import shutil
import time
import tempfile
from pathlib import Path

import torch
import subprocess

IMAGE_DIR = Path(os.environ.get(
    "NNUNET_IMAGE_DIR",
    "/root/shared/nnUNet_raw/Dataset137_BraTS2021/imagesTr",
))
OUT_DIR = Path("/root/shared/benchmark_preds")
N_CASES = 5  # number of volumes to time; enough for a stable mean

# ── pick N_CASES cases (one case = 4 files _0000–_0003) ──────────────────────
all_images = sorted(IMAGE_DIR.glob("*_0000.nii.gz"))
selected = all_images[:N_CASES]
case_stems = [p.name.replace("_0000.nii.gz", "") for p in selected]
print(f"Benchmarking {N_CASES} cases: {case_stems}", flush=True)

# Copy just those cases to a temp input dir so predict only runs on them
with tempfile.TemporaryDirectory() as tmp_in:
    tmp_in_path = Path(tmp_in)
    for stem in case_stems:
        for suffix in ("_0000", "_0001", "_0002", "_0003"):
            src = IMAGE_DIR / f"{stem}{suffix}.nii.gz"
            if src.exists():
                shutil.copy(src, tmp_in_path / src.name)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── time the full predict call ────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    subprocess.run(
        [
            "nnUNetv2_predict",
            "-i", str(tmp_in_path),
            "-o", str(OUT_DIR),
            "-d", "137",
            "-c", "3d_fullres",
            "-f", "0", "1", "2", "3", "4",
            "-chk", "checkpoint_best.pth",
            "--disable_progress_bar",
        ],
        check=True,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available()
        else float("nan")
    )

per_volume_s = total_s / N_CASES

print("\n===== Inference Benchmark =====")
print(f"  GPU:                  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  Cases timed:          {N_CASES}")
print(f"  Total time:           {total_s:.1f} s")
print(f"  Time per volume:      {per_volume_s:.2f} s")
print(f"  Peak GPU memory:      {peak_mb:.0f} MB")