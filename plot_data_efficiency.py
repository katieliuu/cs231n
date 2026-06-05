#!/usr/bin/env python3
"""
Plot data efficiency curve: Dice vs % training data.
Run locally after:
  modal volume get nnunet-brats-volume data_efficiency ./local_data_efficiency

Then:
  python plot_data_efficiency.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("./local_results")

# 100% results from your main eval logs — paste your values here
FULL_RESULTS = {
    "dice_tc":  0.9215,
    "dice_wt":  0.9353,
    "dice_et":  0.8681,
    "dice_avg": 0.9083,
}

def load_results(pct: int) -> dict:
    p = DATA_DIR / f"pct{pct}" / "results.json"
    if not p.exists():
        print(f"  WARNING: {p} not found, skipping pct={pct}")
        return {}
    with open(p) as f:
        return json.load(f)

# Load all results
all_results = {}
for pct in [10, 25, 50]:
    r = load_results(pct)
    if r:
        all_results[pct] = r
all_results[100] = FULL_RESULTS

if len(all_results) < 2:
    raise SystemExit("Not enough data points found. Check ./local_data_efficiency/")

pcts     = sorted(all_results.keys())
dice_avg = [all_results[p]["dice_avg"] for p in pcts]
dice_tc  = [all_results[p]["dice_tc"]  for p in pcts]
dice_wt  = [all_results[p]["dice_wt"]  for p in pcts]
dice_et  = [all_results[p]["dice_et"]  for p in pcts]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("nnU-Net Data Efficiency: Dice vs Training Data Size", fontsize=13)

# Left: avg dice only
ax = axes[0]
ax.plot(pcts, dice_avg, "o-", color="black", linewidth=2, markersize=7, label="Avg Dice")
for x, y in zip(pcts, dice_avg):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.set_xlabel("Training data (%)")
ax.set_ylabel("Mean Dice")
ax.set_title("Average Dice")
ax.set_xticks(pcts)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
ax.legend()

# Right: per-region
ax = axes[1]
ax.plot(pcts, dice_tc,  "o-", label="TC",  color="tab:blue")
ax.plot(pcts, dice_wt,  "s-", label="WT",  color="tab:orange")
ax.plot(pcts, dice_et,  "^-", label="ET",  color="tab:green")
ax.plot(pcts, dice_avg, "D--", label="Avg", color="black", linewidth=2)
ax.set_xlabel("Training data (%)")
ax.set_ylabel("Dice")
ax.set_title("Dice per Region (TC / WT / ET)")
ax.set_xticks(pcts)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
out_path = Path("./plots/data_efficiency.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")

# ── print table ───────────────────────────────────────────────────────────────
print("\n===== Data Efficiency Results =====")
print(f"{'PCT':>6}  {'dice_tc':>8}  {'dice_wt':>8}  {'dice_et':>8}  {'dice_avg':>8}")
for p in pcts:
    r = all_results[p]
    print(f"{p:>5}%  {r['dice_tc']:>8.4f}  {r['dice_wt']:>8.4f}  {r['dice_et']:>8.4f}  {r['dice_avg']:>8.4f}")