#!/usr/bin/env python3
"""
Comprehensive validation metrics for Swin UNETR checkpoints on BraTS21.

Metrics:
  - False-positive voxel counts per case (TC, WT, ET, total)
  - Inference time per volume, peak GPU memory, model size
  - Expected calibration error (ECE) and Brier score (from sigmoid probabilities)
  - Surface Dice (NSD-style, MONAI SurfaceDiceMetric)
  - Average symmetric surface distance (ASSD, MONAI SurfaceDistanceMetric symmetric=True)

Reuses preprocessing / sliding-window settings from eval_sensitivity_hd95.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from monai import data, transforms
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import SurfaceDiceMetric, SurfaceDistanceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

from eval_sensitivity_hd95 import (
    _checkpoint_paths,
    _parse_epochs_arg,
    _voxel_spacing_from_meta,
    datafold_read,
)


def _case_id(entry: dict) -> str:
    label = entry.get("label", "")
    if isinstance(label, str):
        m = re.search(r"(BraTS2021_\d+)", label)
        if m:
            return m.group(1)
        return Path(label).parent.name
    return "unknown"


def _mean_finite(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _ece_brier(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    """ECE and Brier on flattened binary probs/labels in [0,1]."""
    p = probs.astype(np.float64).ravel()
    y = labels.astype(np.float64).ravel()
    if p.size == 0:
        return float("nan"), float("nan")
    brier = float(np.mean((p - y) ** 2))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p > lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += mask.mean() * abs(acc - conf)
    return ece, brier


def _false_positive_voxels(pred_bin: torch.Tensor, gt: torch.Tensor) -> tuple[int, int, int]:
    """Per-channel FP counts for binarized CHWD tensors."""
    pred = pred_bin.squeeze(0) > 0
    lab = gt.squeeze(0) > 0
    fps = []
    for c in range(min(3, pred.shape[0])):
        fps.append(int((pred[c] & ~lab[c]).sum().item()))
    while len(fps) < 3:
        fps.append(0)
    return fps[0], fps[1], fps[2]


def _model_size_stats(model: torch.nn.Module, ckpt_path: Path) -> dict[str, float]:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ckpt_mb = ckpt_path.stat().st_size / (1024**2)
    param_mb_fp32 = n_params * 4 / (1024**2)
    return {
        "n_parameters": float(n_params),
        "n_trainable_parameters": float(n_trainable),
        "checkpoint_size_mb": float(ckpt_mb),
        "parameter_memory_mb_fp32": float(param_mb_fp32),
    }


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: Path,
    val_files: list,
    *,
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    max_cases: int | None,
    surface_dice_tolerance_mm: float,
    ece_bins: int,
    out_dir: Path,
) -> dict:
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    files = val_files if max_cases is None else val_files[: max(0, max_cases)]
    val_ds = data.Dataset(data=files, transform=val_transform)
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    ).to(device)
    try:
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    size_stats = _model_size_stats(model, ckpt_path)

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    inferer = partial(
        sliding_window_inference,
        roi_size=list(roi_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )

    sd_metric = SurfaceDiceMetric(
        class_thresholds=[surface_dice_tolerance_mm] * 3,
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

    per_case_rows: list[dict] = []
    use_cuda = device.type == "cuda"

    for batch_idx, batch in enumerate(val_loader):
        entry = files[batch_idx]
        case = _case_id(entry)
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        spacing = _voxel_spacing_from_meta(batch["image"])

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logits = inferer(image)
        if use_cuda:
            torch.cuda.synchronize(device)
        infer_s = time.perf_counter() - t0
        peak_gpu_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2) if use_cuda else float("nan")
        )

        preds = decollate_batch(logits)
        labs = decollate_batch(label)
        prob_list = [post_sigmoid(p) for p in preds]
        pred_bin = [post_pred(p) for p in prob_list]

        fp_tc, fp_wt, fp_et = _false_positive_voxels(pred_bin[0], labs[0])

        ece_tc = ece_wt = ece_et = brier_tc = brier_wt = brier_et = float("nan")
        prob_np = prob_list[0].detach().cpu().numpy()
        lab_np = labs[0].detach().cpu().numpy()
        if prob_np.shape[0] >= 3:
            ece_tc, brier_tc = _ece_brier(prob_np[0], lab_np[0], n_bins=ece_bins)
            ece_wt, brier_wt = _ece_brier(prob_np[1], lab_np[1], n_bins=ece_bins)
            ece_et, brier_et = _ece_brier(prob_np[2], lab_np[2], n_bins=ece_bins)

        sd_metric.reset()
        sd_metric(y_pred=pred_bin, y=labs, spacing=spacing)
        sd_agg = sd_metric.aggregate()
        sd_vals = sd_agg[0].flatten().cpu().numpy() if isinstance(sd_agg, tuple) else sd_agg.flatten().cpu().numpy()

        assd_metric.reset()
        assd_metric(y_pred=pred_bin, y=labs, spacing=spacing)
        assd_agg = assd_metric.aggregate()
        assd_vals = (
            assd_agg[0].flatten().cpu().numpy()
            if isinstance(assd_agg, tuple)
            else assd_agg.flatten().cpu().numpy()
        )

        row = {
            "case_id": case,
            "fp_tc": fp_tc,
            "fp_wt": fp_wt,
            "fp_et": fp_et,
            "fp_total": fp_tc + fp_wt + fp_et,
            "inference_time_s": infer_s,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "ece_tc": ece_tc,
            "ece_wt": ece_wt,
            "ece_et": ece_et,
            "ece_avg": float(np.nanmean([ece_tc, ece_wt, ece_et])),
            "brier_tc": brier_tc,
            "brier_wt": brier_wt,
            "brier_et": brier_et,
            "brier_avg": float(np.nanmean([brier_tc, brier_wt, brier_et])),
            "surface_dice_tc": float(sd_vals[0]) if sd_vals.size >= 1 else float("nan"),
            "surface_dice_wt": float(sd_vals[1]) if sd_vals.size >= 2 else float("nan"),
            "surface_dice_et": float(sd_vals[2]) if sd_vals.size >= 3 else float("nan"),
            "surface_dice_avg": _mean_finite([float(sd_vals[i]) for i in range(min(3, sd_vals.size))]),
            "assd_tc": float(assd_vals[0]) if assd_vals.size >= 1 else float("nan"),
            "assd_wt": float(assd_vals[1]) if assd_vals.size >= 2 else float("nan"),
            "assd_et": float(assd_vals[2]) if assd_vals.size >= 3 else float("nan"),
            "assd_avg": _mean_finite([float(assd_vals[i]) for i in range(min(3, assd_vals.size))]),
        }
        per_case_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    ep = int(re.search(r"(\d+)", ckpt_path.stem).group(1)) if re.search(r"(\d+)", ckpt_path.stem) else -1
    per_case_path = out_dir / f"eval_comprehensive_epoch{ep}_per_case.csv"
    fieldnames = list(per_case_rows[0].keys()) if per_case_rows else []
    with open(per_case_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_case_rows)

    def col_mean(key: str) -> float:
        return _mean_finite([float(r[key]) for r in per_case_rows])

    summary = {
        "epoch": ep,
        "n_cases": len(per_case_rows),
        "checkpoint": str(ckpt_path),
        **size_stats,
        "mean_fp_tc": col_mean("fp_tc"),
        "mean_fp_wt": col_mean("fp_wt"),
        "mean_fp_et": col_mean("fp_et"),
        "mean_fp_total": col_mean("fp_total"),
        "mean_inference_time_s": col_mean("inference_time_s"),
        "std_inference_time_s": float(np.std([r["inference_time_s"] for r in per_case_rows])),
        "mean_peak_gpu_memory_mb": col_mean("peak_gpu_memory_mb"),
        "max_peak_gpu_memory_mb": float(max(r["peak_gpu_memory_mb"] for r in per_case_rows)),
        "ece_tc": col_mean("ece_tc"),
        "ece_wt": col_mean("ece_wt"),
        "ece_et": col_mean("ece_et"),
        "ece_avg": col_mean("ece_avg"),
        "brier_tc": col_mean("brier_tc"),
        "brier_wt": col_mean("brier_wt"),
        "brier_et": col_mean("brier_et"),
        "brier_avg": col_mean("brier_avg"),
        "surface_dice_tc": col_mean("surface_dice_tc"),
        "surface_dice_wt": col_mean("surface_dice_wt"),
        "surface_dice_et": col_mean("surface_dice_et"),
        "surface_dice_avg": col_mean("surface_dice_avg"),
        "assd_tc_mm": col_mean("assd_tc"),
        "assd_wt_mm": col_mean("assd_wt"),
        "assd_et_mm": col_mean("assd_et"),
        "assd_avg_mm": col_mean("assd_avg"),
        "surface_dice_tolerance_mm": surface_dice_tolerance_mm,
        "ece_bins": ece_bins,
        "per_case_csv": str(per_case_path),
    }
    summary_path = out_dir / f"eval_comprehensive_epoch{ep}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Comprehensive BraTS validation metrics.")
    p.add_argument("--data-dir", type=Path, default=Path("/scratch/users/linika"))
    p.add_argument("--split-json", type=Path, default=Path("/scratch/users/linika/brats21_80_20.json"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("/scratch/users/linika/checkpoints"))
    p.add_argument("--epochs", type=str, default="50")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--sw-batch-size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--surface-dice-tol-mm", type=float, default=1.0, help="NSD tolerance per region (mm).")
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/scratch/users/linika/checkpoints/eval_comprehensive"),
    )
    args = p.parse_args()

    _, val_files = datafold_read(args.split_json, args.data_dir, fold=None)
    pairs = _checkpoint_paths(args.checkpoint_dir, _parse_epochs_arg(args.epochs))
    pairs = [(e, path) for e, path in pairs if path.is_file()]
    if not pairs:
        raise SystemExit(f"No checkpoints for epochs={args.epochs!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ep, ckpt in pairs:
        print(f"=== Epoch {ep}: {ckpt.name} (device={device}) ===", flush=True)
        summary = evaluate_checkpoint(
            ckpt,
            val_files,
            device=device,
            roi_size=tuple(args.roi),
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            max_cases=args.max_cases,
            surface_dice_tolerance_mm=args.surface_dice_tol_mm,
            ece_bins=args.ece_bins,
            out_dir=args.out_dir,
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "per_case_csv"}, indent=2), flush=True)
        print(f"Per-case CSV: {summary['per_case_csv']}", flush=True)


if __name__ == "__main__":
    main()
