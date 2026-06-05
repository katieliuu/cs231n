#!/usr/bin/env python3
"""
Evaluate saved Swin UNETR checkpoints on the BraTS validation split: sensitivity (recall)
and 95th-percentile Hausdorff distance (HD95), without training.

Uses the same preprocessing and sliding-window settings as Copy_of_swin_unetr_brats21_segmentation_3d.py.

HD95 via MONAI requires SciPy (``pip install scipy`` in the same env as MONAI).
"""
from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.utils import decollate_batch
from monai.inferers.utils import sliding_window_inference
from monai.metrics.confusion_matrix import ConfusionMatrixMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ConvertToMultiChannelBasedOnBratsClassesd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.post.array import Activations, AsDiscrete
from monai.utils.enums import MetricReduction
from typing import cast


def _scipy_ok() -> bool:
    try:
        import scipy.ndimage  # noqa: F401

        return True
    except ImportError:
        return False


def datafold_read(datalist: str | Path, basedir: str | Path, fold: int | None = None):
    basedir = Path(basedir)
    with open(datalist) as f:
        json_data = json.load(f)
    if "validation" in json_data:
        tr = json_data["training"]
        val = json_data["validation"]
    else:
        all_data = json_data["training"]
        tr, val = [], []
        for d in all_data:
            if "fold" in d and d["fold"] == fold:
                val.append(d)
            else:
                tr.append(d)
    for split in (tr, val):
        for d in split:
            for k in d:
                if isinstance(d[k], list):
                    d[k] = [str(basedir / iv) for iv in d[k]]
                elif isinstance(d[k], str):
                    d[k] = str(basedir / d[k]) if d[k] else d[k]
    return tr, val


def _voxel_spacing_from_meta(image_tensor) -> tuple[float, float, float]:
    """Voxel spacing (mm) for spatial dims, matching MONAI tensor order (D, H, W)."""
    meta = getattr(image_tensor, "meta", None) or {}
    pix = meta.get("pixdim")
    if pix is not None:
        if hasattr(pix, "detach"):
            pix = pix.detach().cpu().numpy()
        pix = np.asarray(pix, dtype=np.float64).ravel()
        if pix.size >= 4:
            # NIfTI-style: pixdim[0] unused; 1,2,3 often dx,dy,dz — map to D,H,W of array
            return (float(pix[1]), float(pix[2]), float(pix[3]))
    aff = meta.get("affine")
    if aff is None:
        return (1.0, 1.0, 1.0)
    if hasattr(aff, "detach"):
        aff = aff.detach().cpu().numpy()
    aff = np.asarray(aff, dtype=np.float64)
    # unpack explicitly so Pylance infers tuple[float, float, float] not tuple[float, ...]
    d = float(np.linalg.norm(aff[:3, 0]))
    h = float(np.linalg.norm(aff[:3, 1]))
    w = float(np.linalg.norm(aff[:3, 2]))
    return (d, h, w)


def _parse_epochs_arg(s: str) -> list[int] | None:
    if s.strip().lower() == "all":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def _checkpoint_paths(ckpt_dir: Path, epochs: list[int] | None) -> list[tuple[int, Path]]:
    if epochs is None:
        # re.search may return None; use walrus operator with fallback
        paths = sorted(
            ckpt_dir.glob("model_epoch_*.pth"),
            key=lambda p: int(m.group(1)) if (m := re.search(r"(\d+)", p.stem)) else 0,
        )
        out = []
        for p in paths:
            m = re.search(r"model_epoch_(\d+)", p.name)
            if m:
                out.append((int(m.group(1)), p))
        return out
    return [(e, ckpt_dir / f"model_epoch_{e}.pth") for e in epochs]


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
) -> dict[str, float]:
    # use directly imported names instead of transforms.Compose / data.Dataset etc.
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    files = val_files if max_cases is None else val_files[: max(0, max_cases)]
    val_ds = Dataset(data=files, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

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

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    inferer = partial(
        sliding_window_inference,
        roi_size=list(roi_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )

    sens_metric = ConfusionMatrixMetric(
        include_background=True,
        metric_name="sensitivity",
        reduction=MetricReduction.MEAN_BATCH,
        get_not_nans=True,
    )
    do_hd = _scipy_ok()
    hd_metric = None
    if do_hd:
        hd_metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=95,
            reduction=MetricReduction.MEAN_BATCH,
            get_not_nans=True,
        )

    sens_tc, sens_wt, sens_et = [], [], []
    hd_tc, hd_wt, hd_et = [], [], []

    for batch in val_loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        logits = inferer(image)
        preds = cast(list[torch.Tensor], decollate_batch(logits))
        labs  = cast(list[torch.Tensor], decollate_batch(label))
        # ensure pred_bin contains Tensors (not NdarrayOrTensor) to satisfy TensorOrList
        pred_bin: list[torch.Tensor] = [
            torch.as_tensor(post_pred(post_sigmoid(p)))
            for p in preds
        ]
        labs_tensor: list[torch.Tensor] = [torch.as_tensor(lb) for lb in labs]
        spacing = _voxel_spacing_from_meta(batch["image"])

        sens_metric.reset()
        sens_metric(y_pred=pred_bin, y=labs_tensor)
        sens_agg = sens_metric.aggregate()
        if isinstance(sens_agg, list):
            s, _ = sens_agg[0]
        else:
            s = sens_agg
        s = s.flatten().cpu().numpy()
        if s.size >= 3:
            sens_tc.append(float(s[0]))
            sens_wt.append(float(s[1]))
            sens_et.append(float(s[2]))

        if hd_metric is not None:
            hd_metric.reset()
            hd_metric(y_pred=pred_bin, y=labs_tensor, spacing=list(spacing))
            hd_agg = hd_metric.aggregate()
            if isinstance(hd_agg, tuple):
                h, _ = hd_agg
            else:
                h = hd_agg
            h = h.flatten().cpu().numpy()
            if h.size >= 3:
                hd_tc.append(float(h[0]) if np.isfinite(h[0]) else float("nan"))
                hd_wt.append(float(h[1]) if np.isfinite(h[1]) else float("nan"))
                hd_et.append(float(h[2]) if np.isfinite(h[2]) else float("nan"))

    def _mean(xs: list[float]) -> float:
        a = np.asarray(xs, dtype=np.float64)
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    return {
        "n": len(files),
        "sens_tc": _mean(sens_tc),
        "sens_wt": _mean(sens_wt),
        "sens_et": _mean(sens_et),
        "sens_avg": _mean(sens_tc + sens_wt + sens_et),
        "hd95_tc": _mean(hd_tc),
        "hd95_wt": _mean(hd_wt),
        "hd95_et": _mean(hd_et),
        "hd95_avg": _mean(hd_tc + hd_wt + hd_et),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Sensitivity + HD95 on BraTS val for saved checkpoints.")
    p.add_argument("--data-dir", type=Path, default=Path("/scratch/users/linika"))
    p.add_argument("--split-json", type=Path, default=Path("/scratch/users/linika/brats21_80_20.json"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("/scratch/users/linika/checkpoints"))
    p.add_argument(
        "--epochs",
        type=str,
        default="50",
        help='Comma-separated epoch indices (e.g. "5,10,50") or "all" for every model_epoch_*.pth',
    )
    p.add_argument("--max-cases", type=int, default=None, help="Limit validation cases (debug / smoke test).")
    p.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--sw-batch-size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Append one CSV row per checkpoint (default: checkpoints/eval_sensitivity_hd95.csv).",
    )
    p.add_argument("--no-csv", action="store_true", help="Do not write CSV.")
    args = p.parse_args()

    _, val_files = datafold_read(args.split_json, args.data_dir, fold=None)
    ep_list = _parse_epochs_arg(args.epochs)
    pairs = _checkpoint_paths(args.checkpoint_dir, ep_list)
    pairs = [(e, path) for e, path in pairs if path.is_file()]
    if not pairs:
        raise SystemExit(f"No checkpoints found under {args.checkpoint_dir} for epochs={args.epochs!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi = tuple(args.roi)
    if not _scipy_ok():
        print(
            "WARNING: SciPy not installed; HD95 will be NaN. "
            "Install with: pip install scipy",
            flush=True,
        )

    rows: list[dict[str, float | int | str]] = []
    for ep, ckpt in pairs:
        print(f"=== Epoch {ep}: {ckpt.name} (device={device}) ===", flush=True)
        stats: dict[str, float | int | str] = dict(evaluate_checkpoint(
            ckpt,
            val_files,
            device=device,
            roi_size=roi,  # type: ignore[arg-type]
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            max_cases=args.max_cases,
        ))
        stats["epoch"] = ep
        stats["checkpoint"] = str(ckpt)
        rows.append(stats)
        print(
            f"  cases={stats['n']}: "
            f"Sens TC/WT/ET (mean)={stats['sens_tc']:.4f}, {stats['sens_wt']:.4f}, {stats['sens_et']:.4f} | "
            f"Sens_avg={stats['sens_avg']:.4f}",
            flush=True,
        )
        print(
            f"  HD95 (mm) TC/WT/ET (mean)={stats['hd95_tc']:.4f}, {stats['hd95_wt']:.4f}, {stats['hd95_et']:.4f} | "
            f"HD95_avg={stats['hd95_avg']:.4f}",
            flush=True,
        )

    if args.no_csv:
        return
    out_csv = args.out_csv or Path("/scratch/users/linika/checkpoints/eval_sensitivity_hd95.csv")
    if out_csv is not None:
        import csv

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out_csv.is_file()
        fieldnames = [
            "epoch",
            "n",
            "sens_tc",
            "sens_wt",
            "sens_et",
            "sens_avg",
            "hd95_tc",
            "hd95_wt",
            "hd95_et",
            "hd95_avg",
            "checkpoint",
        ]
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                w.writeheader()
            # default to float("nan") not "" so the dict stays numeric-compatible
            for r in rows:
                w.writerow({k: r.get(k, float("nan")) for k in fieldnames})
        print(f"Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
