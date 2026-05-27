#!/usr/bin/env python3
"""Generate training figures from loss.txt and val.txt (produced from run.log)."""
import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.ticker as mticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/scratch/users/linika")

_LOSS_RE = re.compile(r"loss: ([0-9.]+)")
_VAL_EP_RE = re.compile(r"Final validation stats\s+(\d+)/")
_DICE_AVG_RE = re.compile(r"Dice_Avg:\s*([0-9.]+)")
_FINAL_TRAIN_RE = re.compile(r"Final training\s+(\d+)/\d+\s+loss:\s*([0-9.]+)")


def plot_loss():
    losses = []
    loss_path = ROOT / "loss.txt"
    with open(loss_path) as f:
        for line in f:
            m = re.search(r"loss: ([0-9.]+)", line)
            if m:
                losses.append(float(m.group(1)))

    plt.figure(figsize=(10, 4))
    plt.plot(losses, linewidth=0.6, alpha=0.85)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = ROOT / "loss_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out} ({len(losses)} points)")


def plot_val_dice():
    dice = []
    epochs = []
    val_path = ROOT / "val.txt"
    with open(val_path) as f:
        for line in f:
            ep = re.search(r"Final validation stats\s+(\d+)/", line)
            m = re.search(r"Dice_Avg:\s*([0-9.]+)", line)
            if m:
                dice.append(float(m.group(1)))
                if ep:
                    epochs.append(int(ep.group(1)))
                else:
                    epochs.append(len(epochs))

    plt.figure(figsize=(8, 4))
    x = epochs if len(epochs) == len(dice) else range(len(dice))
    plt.plot(x, dice, marker="o", linewidth=1.5, markersize=8)
    plt.title("Validation Dice (Dice_Avg)")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = ROOT / "dice_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out} ({len(dice)} checkpoints: epochs {epochs})")


def plot_dice_scores_per_region(
    out_path: Path | None = None,
) -> Path:
    """Validation Dice vs epoch for TC, WT, ET, and mean (user-reported checkpoint table)."""
    epochs = np.array([0, 4, 9, 14, 24, 34, 44, 49], dtype=float)
    tc = np.array([0.420, 0.692, 0.733, 0.737, 0.750, 0.753, 0.755, 0.755])
    wt = np.array([0.739, 0.798, 0.835, 0.839, 0.851, 0.857, 0.861, 0.862])
    et = np.array([0.380, 0.788, 0.793, 0.796, 0.804, 0.808, 0.812, 0.812])
    avg = np.array([0.513, 0.759, 0.787, 0.790, 0.802, 0.806, 0.809, 0.810])

    series = [
        ("TC", tc, "#0072B2"),
        ("WT", wt, "#D55E00"),
        ("ET", et, "#009E73"),
        ("Average", avg, "#CC79A7"),
    ]

    out = out_path or (ROOT / "dice_scores_regions.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, y, color in series:
        ax.plot(epochs, y, marker="o", linewidth=2.0, markersize=7, label=name, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice score")
    ax.set_title("Validation Dice by tumor sub-region")
    ax.set_xlim(0, 49)
    ax.set_ylim(0.32, 0.92)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_loss_from_brats_log(
    log_path: Path, out_path: Path, *, title: str | None = None, xlabel: str | None = None
) -> int:
    """Per-batch training losses from a BRATS training log (stdout mirror)."""
    losses = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _LOSS_RE.search(line)
            if m:
                losses.append(float(m.group(1)))

    plt.figure(figsize=(10, 4))
    plt.plot(losses, linewidth=0.6, alpha=0.85, color="C0")
    plt.title(title or "Training loss (per batch)")
    plt.xlabel(xlabel or "Batch step (within logged run)")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path} ({len(losses)} points)")
    return len(losses)


def plot_dice_from_brats_log(
    log_path: Path, out_path: Path, *, title: str | None = None, xlabel: str | None = None
) -> None:
    """Validation Dice_Avg vs epoch (1-based) from Final validation stats lines."""
    dice = []
    epoch_1based = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            ep = _VAL_EP_RE.search(line)
            m = _DICE_AVG_RE.search(line)
            if ep and m:
                idx0 = int(ep.group(1))
                dice.append(float(m.group(1)))
                epoch_1based.append(idx0 + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epoch_1based, dice, marker="o", linewidth=1.5, markersize=8)
    plt.title(title or "Validation Dice (Dice_Avg)")
    plt.xlabel(xlabel or "Epoch (1-based, at val_every checkpoints)")
    plt.ylabel("Dice score")
    plt.xticks(epoch_1based)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path} ({len(dice)} points at epochs {epoch_1based})")


def _epoch_ticks(max_epochs: int) -> list[int]:
    """Ticks at 1 and every 5 through max_epochs (e.g. 50 → 1,5,…,50)."""
    return [1] + list(range(5, max_epochs + 1, 5))


def parse_final_training_losses(log_path: Path, max_epochs: int) -> np.ndarray:
    """Per 0-based epoch index, last epoch-average training loss from ``Final training`` lines in *log_path*."""
    y = np.full(max_epochs, np.nan, dtype=float)
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _FINAL_TRAIN_RE.search(line)
            if not m:
                continue
            ei = int(m.group(1))
            if 0 <= ei < max_epochs:
                y[ei] = float(m.group(2))
    return y


def count_loss_lines_in_txt(path: Path) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if _LOSS_RE.search(line):
                n += 1
    return n


def epoch_mean_loss_from_loss_txt(path: Path, n_epochs: int) -> np.ndarray:
    """Mean batch loss for epoch indices ``0 .. n_epochs-1`` from ``Epoch e/... loss:`` lines."""
    pat = re.compile(r"Epoch (\d+)/\d+ \d+/\d+ loss: ([0-9.]+)")
    buckets: dict[int, list[float]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            e = int(m.group(1))
            if e >= n_epochs:
                continue
            buckets.setdefault(e, []).append(float(m.group(2)))
    out = np.full(n_epochs, np.nan)
    for e in range(n_epochs):
        if e in buckets and buckets[e]:
            out[e] = float(np.mean(buckets[e]))
    return out


def digitize_loss_png_early_epochs(
    png_path: Path,
    n_iterations: int,
    *,
    n_early_epochs: int = 10,
    samples_per_epoch: int = 1000,
) -> np.ndarray:
    """Trace ``loss_curve.png``: per epoch bin, mean vertical position of the ink (approximate).

    Assumes the plot x-axis is linear in iteration order and ``samples_per_epoch`` lines per epoch
    (e.g. 1000). *n_iterations* should match the number of loss points used when the PNG was built.
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(str(png_path))
    if img.ndim != 3:
        raise ValueError(f"Expected RGB/RGBA image, got shape {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    if float(np.max(img)) > 1.5:
        img = img.astype(float) / 255.0
    H, W = img.shape[:2]
    gray = np.mean(img, axis=2)
    dark = 1.0 - gray
    xm0, xm1 = int(0.08 * W), int(0.97 * W)
    ym0, ym1 = int(0.12 * H), int(0.88 * H)
    dc = dark[ym0:ym1, xm0:xm1]
    Hc, Wc = dc.shape

    rows = np.empty(Wc, dtype=float)
    for x in range(Wc):
        col = dc[:, x]
        thr = float(np.percentile(col, 96))
        hit = np.where(col >= thr)[0]
        rows[x] = float(np.median(hit)) if hit.size else np.nan
    good = ~np.isnan(rows)
    if np.sum(good) < 2:
        raise RuntimeError("Could not detect a curve in the PNG (try a higher-resolution export).")
    idx = np.arange(Wc)
    rows[~good] = np.interp(idx[~good], idx[good], rows[good])

    raw = np.empty(n_early_epochs, dtype=float)
    for k in range(n_early_epochs):
        vals: list[float] = []
        for it in range(k * samples_per_epoch, min((k + 1) * samples_per_epoch, n_iterations)):
            xc = int((it / max(n_iterations - 1, 1)) * (Wc - 1))
            vals.append(float(rows[xc]))
        raw[k] = float(np.mean(vals)) if vals else np.nan
    return raw


def early_loss_from_png_calibrated(
    png_path: Path,
    *,
    n_iterations: int,
    calibrate_txt: Path | None,
    n_early_epochs: int = 10,
) -> tuple[np.ndarray, str]:
    """Digitize ``loss_curve.png``; optionally affine-calibrate to ``loss.txt`` (same underlying series)."""
    raw = digitize_loss_png_early_epochs(png_path, n_iterations, n_early_epochs=n_early_epochs)
    note = "early: digitized from PNG trace"
    if calibrate_txt is not None and calibrate_txt.is_file():
        tgt = epoch_mean_loss_from_loss_txt(calibrate_txt, n_early_epochs)
        if not np.any(np.isnan(tgt)):
            A = np.vstack([raw, np.ones_like(raw)]).T
            a, b = np.linalg.lstsq(A, tgt, rcond=None)[0]
            pred = a * raw + b
            rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
            note = f"early: digitized PNG + affine fit to {calibrate_txt.name} (rmse≈{rmse:.3f})"
            return pred, note
    rmin, rmax = float(np.min(raw)), float(np.max(raw))
    loss_hi, loss_lo = 0.92, 0.35
    pred = loss_hi - (raw - rmin) / (rmax - rmin + 1e-9) * (loss_hi - loss_lo)
    note = "early: digitized PNG + heuristic loss scale (no calibration txt)"
    return pred, note


def plot_combined_loss_png_and_brats_log(
    png_path: Path | None,
    brats_log_path: Path,
    out_path: Path,
    *,
    max_epochs: int = 50,
    calibrate_txt: Path | None = None,
    n_iterations: int | None = None,
    early_losses: Sequence[float] | None = None,
) -> None:
    """Epoch-average loss: epochs 1–10 from PNG digitization or explicit *early_losses*; rest from BRATS log."""
    if early_losses is not None:
        early = np.asarray(early_losses, dtype=float)
        if early.shape != (10,):
            raise ValueError(f"early_losses must have length 10, got {early.shape[0]}")
        note_early = "early: explicit loss samples (iters 0, 1k, …, 9k; ~1k iterations per epoch)"
    else:
        if png_path is None or not png_path.is_file():
            raise FileNotFoundError(
                "combined plot needs --combined-loss-png (existing file) or --early-epoch-losses"
            )
        n_iter = n_iterations
        if n_iter is None:
            if calibrate_txt and calibrate_txt.is_file():
                n_iter = count_loss_lines_in_txt(calibrate_txt)
            else:
                n_iter = 25_000
        early, note_early = early_loss_from_png_calibrated(
            png_path,
            n_iterations=max(n_iter, 1),
            calibrate_txt=calibrate_txt,
            n_early_epochs=min(10, max_epochs),
        )

    y = np.full(max_epochs, np.nan, dtype=float)
    y[:10] = early[:10]
    brats = parse_final_training_losses(brats_log_path, max_epochs)
    for i in range(10, max_epochs):
        if not np.isnan(brats[i]):
            y[i] = brats[i]

    x = np.arange(1, max_epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o", linewidth=1.2, markersize=4, color="C0")
    title_mid = (
        "epochs 1–10: explicit table (iters 0, 1k, …, 9k)"
        if early_losses is not None
        else "epochs 1–10: digitized from loss_curve.png"
    )
    plt.title("Training loss (epoch average)\n" + title_mid + " · epochs 11–50: BRATS log (Final training)")
    plt.xlabel("Epoch (1–{})".format(max_epochs))
    plt.ylabel("Loss")
    plt.xlim(0.5, max_epochs + 0.5)
    plt.xticks(_epoch_ticks(max_epochs))
    plt.grid(True, alpha=0.3)
    plt.figtext(0.5, 0.02, note_early, ha="center", fontsize=8, color="0.35")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    n_ok = int(np.sum(~np.isnan(y)))
    print(f"Wrote {out_path} ({n_ok}/{max_epochs} epochs; {note_early})")


def merge_loss_by_filling_nan(primary: np.ndarray, *extras: np.ndarray) -> np.ndarray:
    """Start from *primary*; for each extra array, fill entries that are still NaN."""
    y = primary.astype(float, copy=True)
    for e in extras:
        mask = np.isnan(y) & ~np.isnan(e)
        y[mask] = e[mask]
    return y


def _checkpoint_epochs_saved(max_epochs: int, val_every: int) -> list[int]:
    """1-based epoch indices for model_epoch_{n}.pth (matches trainer validation schedule)."""
    out: list[int] = []
    for epoch in range(max_epochs):
        if epoch == 0 or (epoch + 1) % val_every == 0:
            out.append(epoch + 1)
    return out


def plot_full_run_loss_epochs(
    log_path: Path,
    out_path: Path,
    *,
    max_epochs: int = 50,
    title: str | None = None,
    merge_log_paths: list[Path] | None = None,
) -> None:
    """One point per epoch: epoch-average training loss from ``Final training e/max loss:`` lines.

    *log_path* is parsed first (e.g. resume run). Optional *merge_log_paths* are applied in order,
    filling only indices that are still NaN (e.g. Slurm stdout from an earlier job that died mid-run).
    """
    y = parse_final_training_losses(log_path, max_epochs)
    merged_from: list[str] = []
    for extra in merge_log_paths or []:
        if not extra.is_file():
            raise FileNotFoundError(f"merge log not found: {extra}")
        before = np.sum(~np.isnan(y))
        y = merge_loss_by_filling_nan(y, parse_final_training_losses(extra, max_epochs))
        after = np.sum(~np.isnan(y))
        if after > before:
            merged_from.append(f"{extra.name}(+{after - before})")

    x = np.arange(1, max_epochs + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker="o", linewidth=1.2, markersize=4, color="C0")
    plt.title(
        title
        or "Training loss (epoch average)\nfrom Final training lines · max_epochs={}".format(max_epochs)
    )
    plt.xlabel("Epoch (1–{})".format(max_epochs))
    plt.ylabel("Loss")
    plt.xlim(0.5, max_epochs + 0.5)
    plt.xticks(_epoch_ticks(max_epochs))
    plt.grid(True, alpha=0.3)
    missing = [i + 1 for i in range(max_epochs) if np.isnan(y[i])]
    if missing:
        if len(missing) > 12:
            note = "Missing epoch-average loss (no Final training line): " + ", ".join(
                str(e) for e in missing[:12]
            ) + " …"
        else:
            note = "Missing epoch-average loss (no Final training line): " + ", ".join(str(e) for e in missing)
        plt.figtext(0.5, 0.02, note, ha="center", fontsize=9, color="0.35")
        plt.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    n_ok = int(np.sum(~np.isnan(y)))
    extra_msg = f"; merged fills from {', '.join(merged_from)}" if merged_from else ""
    print(f"Wrote {out_path} ({n_ok}/{max_epochs} epochs with loss{extra_msg})")


def plot_full_run_dice_checkpoints(
    checkpoint_dir: Path,
    out_path: Path,
    *,
    max_epochs: int = 50,
    val_every: int = 5,
    title: str | None = None,
) -> None:
    """Dice_Avg at each validation checkpoint (model_epoch_*.pth), spanning epochs 1–max_epochs."""
    import torch

    epochs_cp = _checkpoint_epochs_saved(max_epochs, val_every)
    dice_vals: list[float] = []
    for ep in epochs_cp:
        ckpt = checkpoint_dir / f"model_epoch_{ep}.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint for validation epoch {ep}: {ckpt}")
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "val_acc" not in data:
            raise KeyError(f"No val_acc in {ckpt}")
        dice_vals.append(float(data["val_acc"]))

    plt.figure(figsize=(8, 4))
    plt.plot(epochs_cp, dice_vals, marker="o", linewidth=1.5, markersize=8)
    plt.title(title or "Validation Dice (Dice_Avg)\nfrom checkpoint val_acc · val_every={}".format(val_every))
    plt.xlabel("Epoch (1–{})".format(max_epochs))
    plt.ylabel("Dice score")
    plt.xlim(0.5, max_epochs + 0.5)
    plt.xticks(_epoch_ticks(max_epochs))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path} ({len(epochs_cp)} checkpoints: epochs {epochs_cp})")


def plot_curves_from_brats_log(
    log_path: Path,
    tag: str,
    out_dir: Path | None = None,
    *,
    full_epochs: bool = False,
    checkpoint_dir: Path | None = None,
    max_epochs: int = 50,
    val_every: int = 5,
    merge_log_paths: list[Path] | None = None,
) -> None:
    """Write loss_curve_{tag}.png and dice_curve_{tag}.png (does not overwrite loss_curve.png)."""
    base = out_dir or ROOT
    ckpt = checkpoint_dir or (ROOT / "checkpoints")
    if full_epochs:
        plot_full_run_loss_epochs(
            log_path,
            base / f"loss_curve_{tag}.png",
            max_epochs=max_epochs,
            merge_log_paths=merge_log_paths,
        )
        plot_full_run_dice_checkpoints(
            ckpt,
            base / f"dice_curve_{tag}.png",
            max_epochs=max_epochs,
            val_every=val_every,
        )
        return

    plot_loss_from_brats_log(
        log_path,
        base / f"loss_curve_{tag}.png",
        title="Training loss (per batch)\nmax_epochs=50, resume from epoch 10",
        xlabel="Batch step (epochs 10–50 in this log)",
    )
    plot_dice_from_brats_log(
        log_path,
        base / f"dice_curve_{tag}.png",
        title="Validation Dice (Dice_Avg)\nval every 5 epochs · max 50 epochs",
        xlabel="Epoch (1-based, validation checkpoints)",
    )


def _run_swin_unetr_brats_inference(
    data_dir: Path,
    case_id: str,
    checkpoint_path: Path,
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 1,
    overlap: float = 0.5,
) -> np.ndarray:
    """Full-volume BraTS-style label map (0,1,2,4) from Swin UNETR checkpoint; matches training script."""
    import torch
    from functools import partial

    from monai import data, transforms
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import SwinUNETR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td = data_dir / "TrainingData" / case_id
    test_files = [
        {
            "image": [
                str(td / f"{case_id}_flair.nii.gz"),
                str(td / f"{case_id}_t1ce.nii.gz"),
                str(td / f"{case_id}_t1.nii.gz"),
                str(td / f"{case_id}_t2.nii.gz"),
            ],
            "label": str(td / f"{case_id}_seg.nii.gz"),
        }
    ]
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_loader = data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    seg_model = SwinUNETR(
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
    ).to(device)
    try:
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(checkpoint_path), map_location=device)
    seg_model.load_state_dict(ckpt["model_state_dict"])
    seg_model.eval()
    inferer = partial(
        sliding_window_inference,
        roi_size=[roi_size[0], roi_size[1], roi_size[2]],
        sw_batch_size=sw_batch_size,
        predictor=seg_model,
        overlap=overlap,
    )
    with torch.no_grad():
        for batch_data in test_loader:
            image = batch_data["image"].to(device)
            prob = torch.sigmoid(inferer(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]), dtype=np.int16)
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            return seg_out
    raise RuntimeError("inference produced no batch")


def plot_segmentation_qualitative(
    case: str = "BraTS2021_00000",
    checkpoint_path: Path | None = None,
    checkpoint_epoch: int = 10,
):
    import nibabel as nib

    ckpt = checkpoint_path or (ROOT / "checkpoints" / f"model_epoch_{checkpoint_epoch}.pth")
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    base = ROOT / "TrainingData" / case
    flair = base / f"{case}_flair.nii.gz"
    seg_path = base / f"{case}_seg.nii.gz"
    img = nib.load(flair).get_fdata()
    gt = nib.load(seg_path).get_fdata()
    z = img.shape[2] // 2

    seg_out = _run_swin_unetr_brats_inference(ROOT, case, ckpt)

    flair_slice = np.rot90(img[:, :, z], k=-1)
    gt_slice = np.rot90(gt[:, :, z], k=-1)
    pr_slice = np.rot90(seg_out[:, :, z].astype(np.float64), k=-1)

    f_sl = np.asarray(img[:, :, z], dtype=np.float64)
    f_sl = (f_sl - np.min(f_sl)) / (np.ptp(f_sl) + 1e-8)
    pr_sl = seg_out[:, :, z].astype(np.float64)
    overlay = np.stack([f_sl, f_sl, f_sl], axis=-1)
    ed = pr_sl == 2
    nc = pr_sl == 1
    et = pr_sl == 4
    overlay[..., 0] = np.where(et, 0.82 * overlay[..., 0] + 0.18 * 1.0, overlay[..., 0])
    overlay[..., 1] = np.where(ed, 0.72 * overlay[..., 1] + 0.28 * 1.0, overlay[..., 1])
    overlay[..., 2] = np.where(nc, 0.72 * overlay[..., 2] + 0.28 * 1.0, overlay[..., 2])
    overlay = np.clip(overlay, 0, 1)
    overlay = np.rot90(overlay, k=-1)

    ep = ckpt.stem.replace("model_epoch_", "")
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.title("MRI (FLAIR)")
    plt.imshow(flair_slice, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Overlay on FLAIR")
    plt.imshow(overlay)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Ground Truth")
    plt.imshow(gt_slice)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title(f"Prediction (epoch {ep})")
    plt.imshow(pr_slice, vmin=0, vmax=4)
    plt.axis("off")

    plt.tight_layout()
    out = ROOT / "segmentation_example.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Wrote {out} (case={case}, slice z={z}, checkpoint={ckpt.name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training curves from loss.txt/val.txt or a BRATS log.")
    parser.add_argument(
        "--brats-log",
        type=Path,
        help="Path to BRATS training log; writes loss_curve_TAG.png and dice_curve_TAG.png",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Filename suffix (default: stem of --brats-log)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: /scratch/users/linika)",
    )
    parser.add_argument(
        "--full-epochs",
        action="store_true",
        help="X-axis epochs 1–N: loss from Final training lines; dice from checkpoint val_acc",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory with model_epoch_*.pth (default: ROOT/checkpoints)",
    )
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument(
        "--merge-log",
        type=Path,
        action="append",
        default=None,
        help="With --full-epochs: extra log(s) whose Final training lines fill epochs still NaN after --brats-log (repeatable)",
    )
    parser.add_argument(
        "--combined-loss-png",
        type=Path,
        default=None,
        help="With --brats-log: build loss curve from this PNG for epochs 1–10 + log for 11–50; writes loss_curve_TAG_combined.png",
    )
    parser.add_argument(
        "--png-calibrate-txt",
        type=Path,
        default=None,
        help="Optional loss.txt used to count iterations and affine-calibrate digitized PNG (default: ROOT/loss.txt if present)",
    )
    parser.add_argument(
        "--png-iterations",
        type=int,
        default=None,
        help="Number of batch lines matching PNG x-axis (default: count lines in --png-calibrate-txt)",
    )
    parser.add_argument(
        "--early-epoch-losses",
        type=str,
        default=None,
        help="Comma-separated 10 losses for epochs 1–10 (overrides PNG digitization for combined plot)",
    )
    parser.add_argument(
        "--dice-regions-curve",
        action="store_true",
        help="Write dice_scores_regions.png (TC, WT, ET, Average vs epoch) and exit",
    )
    args = parser.parse_args()

    if args.dice_regions_curve:
        base = args.out_dir or ROOT
        plot_dice_scores_per_region(out_path=base / "dice_scores_regions.png")
        raise SystemExit(0)

    if (args.combined_loss_png or args.early_epoch_losses) and not args.brats_log:
        parser.error("--brats-log is required for --combined-loss-png / --early-epoch-losses")

    if args.brats_log and (args.combined_loss_png or args.early_epoch_losses):
        tag = args.tag or args.brats_log.stem
        base = args.out_dir or ROOT
        cal = args.png_calibrate_txt if args.png_calibrate_txt is not None else (ROOT / "loss.txt")
        if not cal.is_file():
            cal = None
        early_list = None
        if args.early_epoch_losses:
            parts = [p.strip() for p in args.early_epoch_losses.split(",") if p.strip()]
            try:
                early_list = [float(p) for p in parts]
            except ValueError as e:
                parser.error(f"invalid --early-epoch-losses: {e}")
            if len(early_list) != 10:
                parser.error(f"--early-epoch-losses: expected 10 comma-separated values, got {len(early_list)}")
        png_resolved = args.combined_loss_png.resolve() if args.combined_loss_png else None
        plot_combined_loss_png_and_brats_log(
            png_resolved,
            args.brats_log.resolve(),
            base / f"loss_curve_{tag}_combined.png",
            max_epochs=args.max_epochs,
            calibrate_txt=cal,
            n_iterations=args.png_iterations,
            early_losses=early_list,
        )
    elif args.brats_log:
        tag = args.tag or args.brats_log.stem
        merge_logs = [p.resolve() for p in (args.merge_log or [])]
        plot_curves_from_brats_log(
            args.brats_log.resolve(),
            tag,
            args.out_dir,
            full_epochs=args.full_epochs,
            checkpoint_dir=args.checkpoint_dir,
            max_epochs=args.max_epochs,
            val_every=args.val_every,
            merge_log_paths=merge_logs or None,
        )
    else:
        plot_loss()
        plot_val_dice()
        try:
            plot_segmentation_qualitative()
        except ImportError:
            print("Skipping segmentation_example.png: nibabel / MONAI / torch not available")
