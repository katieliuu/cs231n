#!/usr/bin/env python3
"""Build nested 10% / 25% / 50% training subsets (same validation set)."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path("/scratch/users/linika")
SRC = ROOT / "brats21_80_20.json"
SEED = 42
FRACTIONS = (0.10, 0.25, 0.50)


def main() -> None:
    spec = json.loads(SRC.read_text())
    train = list(spec["training"])
    val = spec["validation"]
    rng = random.Random(SEED)
    rng.shuffle(train)
    n = len(train)
    out_dir = ROOT / "curriculum_splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    for frac in FRACTIONS:
        k = max(1, int(round(n * frac)))
        subset = train[:k]
        pct = int(frac * 100)
        out = {
            "training": subset,
            "validation": val,
            "meta": {
                "source": str(SRC),
                "seed": SEED,
                "train_fraction": frac,
                "n_train": k,
                "n_val": len(val),
            },
        }
        path = out_dir / f"brats21_train_{pct}pct.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"Wrote {path} ({k} train / {len(val)} val)")


if __name__ == "__main__":
    main()
