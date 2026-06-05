#!/usr/bin/env python3
"""
Parse nnU-Net training_log.txt files and plot loss + pseudo-dice curves.
Run locally after:
  modal volume get nnunet-brats-volume nnUNet_results ./local_results
Then:
  python plot_nnunet_training.py --results-dir ./local_results
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path: Path) -> dict[str, list]:
    epochs, train_loss, val_loss = [], [], []
    dice_tc, dice_wt, dice_et = [], [], []

    epoch_re    = re.compile(r"Epoch\s+(\d+)")
    train_re    = re.compile(r"train_loss\s+([-\d.]+)")
    val_re      = re.compile(r"val_loss\s+([-\d.]+)")
    dice_re     = re.compile(r"Pseudo dice\s+\[([^\]]+)\]")

    current: dict = {}
    for line in log_path.read_text().splitlines():
        if m := epoch_re.search(line):
            current["epoch"] = int(m.group(1))
        if m := train_re.search(line):
            current["train_loss"] = float(m.group(1))
        if m := val_re.search(line):
            current["val_loss"] = float(m.group(1))
        if m := dice_re.search(line):
            vals = [float(x.strip().replace("np.float32(","").replace(")",""))
                    for x in m.group(1).split(",")]
            current["dice"] = vals
            # flush once we have everything
            if all(k in current for k in ("epoch", "train_loss", "val_loss", "dice")):
                epochs.append(current["epoch"])
                train_loss.append(current["train_loss"])
                val_loss.append(current["val_loss"])
                dice_tc.append(current["dice"][0])
                dice_wt.append(current["dice"][1])
                dice_et.append(current["dice"][2])
                current = {}

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "dice_tc": dice_tc,
        "dice_wt": dice_wt,
        "dice_et": dice_et,
    }


def plot_fold(data: dict, fold: int, out_dir: Path) -> None:
    epochs = data["epochs"]
    if not epochs:
        print(f"  fold {fold}: no data parsed, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"nnU-Net — Fold {fold}", fontsize=13)

    # Loss
    ax1.plot(epochs, data["train_loss"], label="Train loss")
    ax1.plot(epochs, data["val_loss"],   label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (negative Dice)")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Dice
    ax2.plot(epochs, data["dice_tc"], label="TC")
    ax2.plot(epochs, data["dice_wt"], label="WT")
    ax2.plot(epochs, data["dice_et"], label="ET")
    avg = [np.mean([tc, wt, et])
           for tc, wt, et in zip(data["dice_tc"], data["dice_wt"], data["dice_et"])]
    ax2.plot(epochs, avg, label="Avg", linestyle="--", color="black")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Pseudo Dice")
    ax2.set_title("Validation Pseudo Dice per Region")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / f"training_curves_fold{fold}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./local_results"),
        help="Local path to downloaded nnUNet_results folder.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("./plots"))
    args = p.parse_args()

    base = args.results_dir
    trainer = "nnUNetTrainer__nnUNetPlans__3d_fullres"
    task = "Dataset137_BraTS2021"
    fold_root = base / task / trainer

    if not fold_root.exists():
        # try one level up in case user passed the parent
        fold_root = base / "nnUNet_results" / task / trainer

    if not fold_root.exists():
        raise SystemExit(
            f"Cannot find results at {fold_root}.\n"
            f"Run: modal volume get nnunet-brats-volume nnUNet_results ./local_results"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    found = False
    for fold in range(5):
        fold_dir = fold_root / f"fold_{fold}"
        logs = sorted(fold_dir.glob("training_log*.txt"))
        if not logs:
            print(f"fold {fold}: no training_log*.txt found in {fold_dir}")
            continue
        log_path = logs[-1]  # use most recent if multiple
        found = True
        print(f"Parsing fold {fold}...")
        data = parse_log(log_path)
        plot_fold(data, fold, args.out_dir)

    if not found:
        raise SystemExit("No training_log.txt files found. Check --results-dir path.")

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()