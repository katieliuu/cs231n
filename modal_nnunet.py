#!/usr/bin/env python3
from pathlib import PurePosixPath

import modal

app = modal.App("nnunet-brats21")

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "build-essential",
    )
    .pip_install(
        "monai[nibabel]",
        "SimpleITK",
        "batchgenerators",
        "nnunetv2",
        "scipy",
        "pandas",
        "matplotlib",
    )
    .add_local_file("prepare_brats21_nnunet.py",       "/workspace/prepare_brats21_nnunet.py")
    .add_local_file("nnunet_inference_utils.py",        "/workspace/nnunet_inference_utils.py")
    .add_local_file("eval_nnunet_sensitivity_hd95.py",  "/workspace/eval_nnunet_sensitivity_hd95.py")
    .add_local_file("eval_nnunet_comprehensive.py",     "/workspace/eval_nnunet_comprehensive.py")
    .add_local_file("eval_comprehensive.py",            "/workspace/eval_comprehensive.py")
    .add_local_file("eval_nnunet_robustness.py",        "/workspace/eval_nnunet_robustness.py")
    .add_local_file("benchmark_inference_time.py",      "/workspace/benchmark_inference_time.py")
    .add_local_file("train_subset_nnunet.py",           "/workspace/train_subset_nnunet.py")
    .add_local_file("patch_nnunet_plans.py",            "/workspace/patch_nnunet_plans.py")
)

volume = modal.Volume.from_name(
    "nnunet-brats-volume",
    create_if_missing=True,
)

VOLUMES: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    "/root/shared": volume,
}
ENV: dict[str, str | None] = {
    "nnUNet_raw":            "/root/shared/nnUNet_raw",
    "nnUNet_preprocessed":   "/root/shared/nnUNet_preprocessed",
    "nnUNet_results":        "/root/shared/nnUNet_results",
    "BRATS_DATA_DIR":        "/root/shared/TrainingData",
    "NNUNET_PRED_DIR":       "/root/shared/preds",
    "NNUNET_IMAGE_DIR":      "/root/shared/nnUNet_raw/Dataset137_BraTS2021/imagesTr",
    "NNUNET_GT_DIR":         "/root/shared/nnUNet_raw/Dataset137_BraTS2021/labelsTr",
    "NNUNET_ROBUSTNESS_DIR": "/root/shared/robustness_preds",
}


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    env=ENV,
)
def prepare_dataset() -> None:
    import subprocess
    subprocess.run(["python", "prepare_brats21_nnunet.py"], check=True)
    volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    env=ENV,
)
def train_fold(fold: int) -> None:
    import subprocess
    # Step 1: preprocess (safe to rerun, skips if already done)
    subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", "137", "--verify_dataset_integrity"],
        check=True,
    )
    # Step 2: patch plans to reduce model to ~6M params (target: 7M)
    subprocess.run(["python", "patch_nnunet_plans.py"], check=True)
    # Step 3: train
    subprocess.run(
        ["nnUNetv2_train", "137", "3d_fullres", str(fold), "--npz"],
        check=True,
    )
    volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    env=ENV,
)
def predict() -> None:
    import subprocess
    subprocess.run(
        [
            "nnUNetv2_predict",
            "-i", "/root/shared/nnUNet_raw/Dataset137_BraTS2021/imagesTr",
            "-o", "/root/shared/preds",
            "-d", "137",
            "-c", "3d_fullres",
            "-f", "0", "1", "2", "3", "4",
            "-chk", "checkpoint_best.pth",
        ],
        check=True,
    )
    volume.commit()


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes=VOLUMES,
    env=ENV,
)
def evaluate() -> None:
    import subprocess
    subprocess.run(["python", "eval_nnunet_sensitivity_hd95.py"], check=True)
    subprocess.run(["python", "eval_nnunet_comprehensive.py"], check=True)
    volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    env=ENV,
)
def evaluate_robustness(condition: str | None = None) -> None:
    import subprocess

    cmd = ["python", "eval_nnunet_robustness.py"]

    if condition is not None:
        cmd.extend(["--condition", condition])

    subprocess.run(cmd, check=True)
    volume.commit()

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes=VOLUMES,
    env=ENV,
)
def benchmark_inference() -> None:
    import subprocess
    subprocess.run(["python", "benchmark_inference_time.py"], check=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    env=ENV,
)
def train_subset(pct: int) -> None:
    """Train on pct% of data (10/25/50), predict on full val set, score Dice."""
    import subprocess
    subprocess.run(["python", "train_subset_nnunet.py", "--pct", str(pct)], check=True)
    volume.commit()


@app.local_entrypoint()
def main() -> None:
    prepare_dataset.remote()
    for fold in range(5):
        train_fold.remote(fold)
    predict.remote()
    evaluate.remote()