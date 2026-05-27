#!/usr/bin/env python3
"""
Full validation benchmark for Swin UNETR: robustness, missing modalities, and reporting.

Uses the same inference settings as training (sliding window 128^3, overlap 0.5).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai import data, transforms
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import (
    ConfusionMatrixMetric,
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDiceMetric,
    SurfaceDistanceMetric,
)
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, MapTransform
from monai.utils.enums import MetricReduction

from eval_sensitivity_hd95 import _voxel_spacing_from_meta, datafold_read

# BraTS JSON channel order: FLAIR, T1ce, T1, T2
CHANNEL_NAMES = ("FLAIR", "T1ce", "T1", "T2")

CONDITION_ORDER = (
    "baseline",
    "gaussian_noise",
    "motion_blur",
    "bias_field",
    "without_FLAIR",
    "without_T1ce",
    "without_T1",
    "without_T2",
)

# Fixed evaluation-time perturbations (documented in output JSON)
PERTURBATION_CONFIG = {
    "gaussian_noise": {
        "transform": "RandGaussianNoise",
        "prob": 1.0,
        "mean": 0.0,
        "std": 0.1,
        "sample_std": False,
    },
    "motion_blur": {
        "transform": "RandGaussianSmooth",
        "prob": 1.0,
        "sigma_x": [1.0, 1.0],
        "sigma_y": [1.0, 1.0],
        "sigma_z": [1.0, 1.0],
        "note": "Fixed isotropic sigma=1.0 voxel (MONAI RandGaussianSmooth, prob=1).",
    },
    "bias_field": {
        "transform": "RandBiasField",
        "prob": 1.0,
        "degree": 3,
        "coeff_range": [0.0, 0.3],
    },
    "missing_modality": {
        "method": "zero_channel_after_load_before_normalize",
        "channels": {name: i for i, name in enumerate(CHANNEL_NAMES)},
    },
}


class ZeroChanneld(MapTransform):
    """Zero out one MRI channel (C,H,W,D) after load."""

    def __init__(self, keys: tuple[str, ...], channel: int):
        super().__init__(keys)
        self.channel = channel

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if isinstance(img, np.ndarray):
                img = img.copy()
                img[self.channel] = 0
            else:
                img = img.clone()
                img[self.channel] = 0
            d[key] = img
        return d


@dataclass
class EvalConfig:
    name: str
    extra_transforms: list = field(default_factory=list)


def _mean_finite(xs: list[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _aggregate_to_numpy(agg: Any) -> np.ndarray:
    """Normalize MONAI metric.aggregate() return (tensor, tuple, or list) to 1d numpy."""
    if isinstance(agg, list):
        agg = agg[0]
    if isinstance(agg, tuple):
        agg = agg[0]
    if hasattr(agg, "flatten"):
        return agg.flatten().detach().cpu().numpy()
    return np.asarray(agg, dtype=np.float64).ravel()


def _ece_brier(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    p = probs.astype(np.float64).ravel()
    y = labels.astype(np.float64).ravel()
    if p.size == 0:
        return float("nan"), float("nan")
    brier = float(np.mean((p - y) ** 2))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return ece, brier


def _false_positive_voxels(pred_bin: torch.Tensor, gt: torch.Tensor) -> tuple[int, int, int]:
    pred = pred_bin.squeeze(0) > 0
    lab = gt.squeeze(0) > 0
    fps = [int((pred[c] & ~lab[c]).sum().item()) for c in range(3)]
    return fps[0], fps[1], fps[2]


def _model_size_stats(model: torch.nn.Module, ckpt_path: Path) -> dict[str, float]:
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "n_parameters": float(n_params),
        "checkpoint_size_mb": float(ckpt_path.stat().st_size / (1024**2)),
        "parameter_memory_mb_fp32": float(n_params * 4 / (1024**2)),
    }


def _build_val_transform(extra: list | None = None) -> transforms.Compose:
    steps: list = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    ]
    if extra:
        steps.extend(extra)
    steps.append(transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True))
    return transforms.Compose(steps)


def _condition_transforms() -> dict[str, EvalConfig]:
    return {
        "baseline": EvalConfig("baseline", []),
        "gaussian_noise": EvalConfig(
            "gaussian_noise",
            [
                transforms.RandGaussianNoised(
                    keys=["image"],
                    prob=1.0,
                    mean=0.0,
                    std=0.1,
                    sample_std=False,
                )
            ],
        ),
        "motion_blur": EvalConfig(
            "motion_blur",
            [
                transforms.RandGaussianSmoothd(
                    keys=["image"],
                    prob=1.0,
                    sigma_x=(1.0, 1.0),
                    sigma_y=(1.0, 1.0),
                    sigma_z=(1.0, 1.0),
                )
            ],
        ),
        "bias_field": EvalConfig(
            "bias_field",
            [
                transforms.RandBiasFieldd(
                    keys=["image"],
                    prob=1.0,
                    degree=3,
                    coeff_range=(0.0, 0.3),
                )
            ],
        ),
        "without_FLAIR": EvalConfig("without_FLAIR", [ZeroChanneld(keys=["image"], channel=0)]),
        "without_T1ce": EvalConfig("without_T1ce", [ZeroChanneld(keys=["image"], channel=1)]),
        "without_T1": EvalConfig("without_T1", [ZeroChanneld(keys=["image"], channel=2)]),
        "without_T2": EvalConfig("without_T2", [ZeroChanneld(keys=["image"], channel=3)]),
    }


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    val_files: list,
    device: torch.device,
    *,
    eval_cfg: EvalConfig,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    max_cases: int | None,
    measure_runtime: bool = False,
    surface_dice_tol_mm: float = 1.0,
    ece_bins: int = 15,
) -> dict[str, Any]:
    val_transform = _build_val_transform(eval_cfg.extra_transforms)
    files = val_files if max_cases is None else val_files[: max(0, max_cases)]
    val_ds = data.Dataset(data=files, transform=val_transform)
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    inferer = partial(
        sliding_window_inference,
        roi_size=list(roi_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )

    dice_m = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    sens_m = ConfusionMatrixMetric(
        include_background=True,
        metric_name="sensitivity",
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    hd_m = HausdorffDistanceMetric(
        include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True
    )
    sd_m = SurfaceDiceMetric(
        class_thresholds=[surface_dice_tol_mm] * 3,
        include_background=True,
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    assd_m = SurfaceDistanceMetric(
        include_background=True, symmetric=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True
    )

    per_case: list[dict[str, Any]] = []
    use_cuda = device.type == "cuda"

    for batch_idx, batch in enumerate(val_loader):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        spacing = _voxel_spacing_from_meta(batch["image"])

        peak_mb = float("nan")
        infer_s = float("nan")
        if measure_runtime:
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            logits = inferer(image)
            if use_cuda:
                torch.cuda.synchronize(device)
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            infer_s = time.perf_counter() - t0
        else:
            logits = inferer(image)

        preds = decollate_batch(logits)
        labs = decollate_batch(label)
        prob_list = [post_sigmoid(p) for p in preds]
        pred_bin = [post_pred(p) for p in prob_list]

        dice_m.reset()
        dice_m(y_pred=pred_bin, y=labs)
        d = _aggregate_to_numpy(dice_m.aggregate())

        sens_m.reset()
        sens_m(y_pred=pred_bin, y=labs)
        s = _aggregate_to_numpy(sens_m.aggregate())

        hd_m.reset()
        hd_m(y_pred=pred_bin, y=labs, spacing=spacing)
        h = _aggregate_to_numpy(hd_m.aggregate())

        sd_m.reset()
        sd_m(y_pred=pred_bin, y=labs, spacing=spacing)
        sd = _aggregate_to_numpy(sd_m.aggregate())

        assd_m.reset()
        assd_m(y_pred=pred_bin, y=labs, spacing=spacing)
        a = _aggregate_to_numpy(assd_m.aggregate())

        fp_tc, fp_wt, fp_et = _false_positive_voxels(pred_bin[0], labs[0])
        prob_np = prob_list[0].detach().cpu().numpy()
        lab_np = labs[0].detach().cpu().numpy()
        ece_tc, brier_tc = _ece_brier(prob_np[0], lab_np[0], n_bins=ece_bins)
        ece_wt, brier_wt = _ece_brier(prob_np[1], lab_np[1], n_bins=ece_bins)
        ece_et, brier_et = _ece_brier(prob_np[2], lab_np[2], n_bins=ece_bins)

        dice_mean = float(np.mean(d[:3])) if d.size >= 3 else float("nan")
        per_case.append(
            {
                "case_idx": batch_idx,
                "dice_tc": float(d[0]),
                "dice_wt": float(d[1]),
                "dice_et": float(d[2]),
                "dice_mean": dice_mean,
                "sens_tc": float(s[0]),
                "sens_wt": float(s[1]),
                "sens_et": float(s[2]),
                "sens_mean": float(np.mean(s[:3])),
                "hd95_tc": float(h[0]),
                "hd95_wt": float(h[1]),
                "hd95_et": float(h[2]),
                "hd95_mean": float(np.mean(h[:3])),
                "surface_dice_mean": float(np.mean(sd[:3])),
                "assd_mean": float(np.mean(a[:3])),
                "fp_total": fp_tc + fp_wt + fp_et,
                "ece_mean": float(np.nanmean([ece_tc, ece_wt, ece_et])),
                "brier_mean": float(np.nanmean([brier_tc, brier_wt, brier_et])),
                "inference_time_s": infer_s,
                "peak_gpu_memory_mb": peak_mb,
            }
        )

    def col(k: str) -> float:
        return _mean_finite([float(r[k]) for r in per_case])

    summary = {
        "condition": eval_cfg.name,
        "n_cases": len(per_case),
        "dice_tc": col("dice_tc"),
        "dice_wt": col("dice_wt"),
        "dice_et": col("dice_et"),
        "dice_mean": col("dice_mean"),
        "sens_tc": col("sens_tc"),
        "sens_wt": col("sens_wt"),
        "sens_et": col("sens_et"),
        "lesion_recall_mean": col("sens_mean"),
        "hd95_tc": col("hd95_tc"),
        "hd95_wt": col("hd95_wt"),
        "hd95_et": col("hd95_et"),
        "hd95_mean": col("hd95_mean"),
        "surface_dice_mean": col("surface_dice_mean"),
        "assd_mean_mm": col("assd_mean"),
        "mean_fp_per_case": col("fp_total"),
        "ece_mean": col("ece_mean"),
        "brier_mean": col("brier_mean"),
        "mean_inference_time_s": col("inference_time_s"),
        "mean_peak_gpu_memory_mb": col("peak_gpu_memory_mb"),
        "per_case": per_case,
    }
    return summary


def _wt_volumes(val_files: list) -> np.ndarray:
    vols = []
    for d in val_files:
        seg = nib.load(d["label"]).get_fdata()
        vols.append(int(((seg == 1) | (seg == 2) | (seg == 4)).sum()))
    return np.array(vols, dtype=np.int64)


def _volume_strata_dice(per_case: list[dict], volumes: np.ndarray) -> dict[str, float]:
    dice = np.array([c["dice_mean"] for c in per_case], dtype=np.float64)
    order = np.argsort(volumes)
    n = len(volumes)
    b1, b2 = n // 3, 2 * (n // 3)
    out = {}
    for name, idx in [("small", order[:b1]), ("medium", order[b1:b2]), ("large", order[b2:])]:
        out[f"dice_{name}_tumors"] = float(dice[idx].mean())
    out["small_large_dice_gap"] = out["dice_large_tumors"] - out["dice_small_tumors"]
    return out


def _load_curriculum_dice(curriculum_dir: Path) -> dict[str, dict]:
    out = {}
    for stage, pct in [("pct10", 10), ("pct25", 25), ("pct50", 50), ("pct100", 100)]:
        p = curriculum_dir / stage / "curriculum_dice.json"
        if p.is_file():
            d = json.loads(p.read_text())
            out[f"dice_{pct}pct"] = {
                "dice_tc": d["final_dice_tc"],
                "dice_wt": d["final_dice_wt"],
                "dice_et": d["final_dice_et"],
                "dice_mean": d["final_dice_avg"],
                "n_train": d["n_train"],
                "source": str(p),
            }
    return out


def _resolve_checkpoint(path: Path, curriculum_dir: Path) -> Path:
    if path.is_file():
        return path
    for rel in ("pct100/model_epoch_50.pth", "pct50/model_epoch_50.pth", "pct25/model_epoch_50.pth"):
        p = curriculum_dir / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"No checkpoint at {path} or under {curriculum_dir}")


def build_final_report(
    benchmark: dict[str, Any],
    curriculum: dict[str, dict],
    model_stats: dict[str, float],
    perturbation_config: dict,
) -> dict[str, Any]:
    base = benchmark["baseline"]
    cur = curriculum

    def g(cond: str, key: str) -> float:
        if cond not in benchmark:
            return float("nan")
        return float(benchmark[cond][key])

    dice_10 = cur.get("dice_10pct", {}).get("dice_mean", float("nan"))
    dice_25 = cur.get("dice_25pct", {}).get("dice_mean", float("nan"))
    dice_50 = cur.get("dice_50pct", {}).get("dice_mean", float("nan"))
    dice_100 = cur.get("dice_100pct", {}).get("dice_mean", float("nan"))
    if np.isnan(dice_100):
        dice_100 = g("baseline", "dice_mean")

    low_data_drop = dice_100 - dice_10 if np.isfinite(dice_10) and np.isfinite(dice_100) else float("nan")

    noise_drops = [
        g("baseline", "dice_mean") - g("gaussian_noise", "dice_mean"),
        g("baseline", "dice_mean") - g("motion_blur", "dice_mean"),
    ]
    if "bias_field" in benchmark:
        noise_drops.append(g("baseline", "dice_mean") - g("bias_field", "dice_mean"))
    mod_drops = [
        g("baseline", "dice_mean") - g("without_FLAIR", "dice_mean"),
        g("baseline", "dice_mean") - g("without_T1ce", "dice_mean"),
        g("baseline", "dice_mean") - g("without_T1", "dice_mean"),
        g("baseline", "dice_mean") - g("without_T2", "dice_mean"),
    ]

    vol = benchmark.get("volume_strata", {})

    report = {
        "checkpoint": benchmark["checkpoint"],
        "perturbation_parameters": perturbation_config,
        "Mean Dice": g("baseline", "dice_mean"),
        "Mean HD95": g("baseline", "hd95_mean"),
        "Surface Dice": g("baseline", "surface_dice_mean"),
        "ASSD": g("baseline", "assd_mean_mm"),
        "Lesion recall": g("baseline", "lesion_recall_mean"),
        "False positives per case": g("baseline", "mean_fp_per_case"),
        "Inference time per volume": g("baseline", "mean_inference_time_s"),
        "Peak inference GPU memory": g("baseline", "mean_peak_gpu_memory_mb"),
        "Model size (checkpoint MB)": model_stats["checkpoint_size_mb"],
        "Model size (parameters)": model_stats["n_parameters"],
        "Dice at 10% training data": dice_10,
        "Dice at 25% training data": dice_25,
        "Dice at 50% training data": dice_50,
        "Dice at 100% training data": dice_100,
        "Low-data Dice drop": low_data_drop,
        "Dice under Gaussian noise": g("gaussian_noise", "dice_mean"),
        "Dice under motion blur": g("motion_blur", "dice_mean"),
        "Dice under bias field": g("bias_field", "dice_mean"),
        "Mean noise Dice drop": float(np.nanmean(noise_drops)),
        "Dice without T1": g("without_T1", "dice_mean"),
        "Dice without T1ce": g("without_T1ce", "dice_mean"),
        "Dice without T2": g("without_T2", "dice_mean"),
        "Dice without FLAIR": g("without_FLAIR", "dice_mean"),
        "Mean missing-modality Dice drop": float(np.nanmean(mod_drops)),
        "Dice small tumors": vol.get("dice_small_tumors", float("nan")),
        "Dice medium tumors": vol.get("dice_medium_tumors", float("nan")),
        "Dice large tumors": vol.get("dice_large_tumors", float("nan")),
        "Small-large Dice gap": vol.get("small_large_dice_gap", float("nan")),
        "ECE": g("baseline", "ece_mean"),
        "Brier score": g("baseline", "brier_mean"),
    }
    report["_benchmark_by_condition"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_case"} for k, v in benchmark.items() if k not in ("checkpoint", "volume_strata")}
    report["_curriculum"] = curriculum
    return report


def _write_final_artifacts(
    final: dict[str, Any],
    out_dir: Path,
    ckpt: Path,
) -> None:
    out_json = out_dir / "final_results.json"
    out_md = out_dir / "final_results.md"
    out_json.write_text(json.dumps(final, indent=2))
    with open(out_md, "w") as f:
        f.write("# Final benchmark results\n\n")
        f.write(f"Checkpoint: `{ckpt}`\n\n")
        f.write("## Perturbation parameters\n\n```json\n")
        f.write(json.dumps(PERTURBATION_CONFIG, indent=2))
        f.write("\n```\n\n## Metrics\n\n| Metric | Value |\n|--------|-------|\n")
        skip = {"_benchmark_by_condition", "_curriculum", "perturbation_parameters", "checkpoint"}
        for k, v in final.items():
            if k in skip or k.startswith("_"):
                continue
            if isinstance(v, float):
                f.write(f"| {k} | {v:.6f} |\n")
            else:
                f.write(f"| {k} | {v} |\n")
    print(f"Wrote {out_json}\nWrote {out_md}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("/scratch/users/linika"))
    p.add_argument("--split-json", type=Path, default=Path("/scratch/users/linika/brats21_80_20.json"))
    p.add_argument("--checkpoint", type=Path, default=Path("/scratch/users/linika/checkpoints/curriculum/pct100/model_epoch_50.pth"))
    p.add_argument("--curriculum-dir", type=Path, default=Path("/scratch/users/linika/checkpoints/curriculum"))
    p.add_argument("--out-dir", type=Path, default=Path("/scratch/users/linika/checkpoints/benchmark"))
    p.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated subset of conditions (default: all). Use with --parts-dir for split jobs.",
    )
    p.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Write one JSON per condition here (for parallel Slurm jobs); skips final report unless all parts exist.",
    )
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--sw-batch-size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _resolve_checkpoint(args.checkpoint, args.curriculum_dir)
    print(f"Using checkpoint: {ckpt}", flush=True)

    _, val_files = datafold_read(args.split_json, args.data_dir)
    volumes = _wt_volumes(val_files)
    curriculum = _load_curriculum_dice(args.curriculum_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
    ).to(device)
    try:
        ck = torch.load(str(ckpt), map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    model_stats = _model_size_stats(model, ckpt)

    all_conditions = _condition_transforms()
    if args.conditions:
        order = [c.strip() for c in args.conditions.split(",") if c.strip()]
        unknown = [c for c in order if c not in all_conditions]
        if unknown:
            raise SystemExit(f"Unknown conditions: {unknown}. Valid: {list(all_conditions)}")
    else:
        order = list(CONDITION_ORDER)

    parts_dir = args.parts_dir
    if parts_dir is not None:
        parts_dir.mkdir(parents=True, exist_ok=True)

    benchmark: dict[str, Any] = {"checkpoint": str(ckpt)}

    for i, name in enumerate(order):
        print(f"[{i+1}/{len(order)}] Evaluating: {name}", flush=True)
        measure_runtime = name == "baseline"
        res = run_evaluation(
            model,
            val_files,
            device,
            eval_cfg=all_conditions[name],
            roi_size=tuple(args.roi),
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            max_cases=args.max_cases,
            measure_runtime=measure_runtime,
        )
        print(f"  dice_mean={res['dice_mean']:.4f} hd95_mean={res['hd95_mean']:.4f}", flush=True)
        if parts_dir is not None:
            part = {"checkpoint": str(ckpt), "condition": name, "result": res}
            if name == "baseline":
                part["volume_strata"] = _volume_strata_dice(res["per_case"], volumes)
            part_path = parts_dir / f"{name}.json"
            part_path.write_text(json.dumps(part, indent=2))
            print(f"Wrote {part_path}", flush=True)
        else:
            benchmark[name] = res
            if name == "baseline":
                benchmark["volume_strata"] = _volume_strata_dice(res["per_case"], volumes)

    if parts_dir is not None:
        return

    final = build_final_report(benchmark, curriculum, model_stats, PERTURBATION_CONFIG)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_final_artifacts(final, args.out_dir, ckpt)


if __name__ == "__main__":
    main()
