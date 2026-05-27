#!/usr/bin/env python3
"""Merge per-condition benchmark JSON parts into final_results.json / .md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_full_benchmark import (
    CONDITION_ORDER,
    PERTURBATION_CONFIG,
    _load_curriculum_dice,
    _model_size_stats,
    _write_final_artifacts,
    build_final_report,
)


def merge_parts(
    parts_dir: Path,
    curriculum_dir: Path,
    out_dir: Path,
    *,
    skip: frozenset[str] = frozenset(),
) -> None:
    parts_dir = parts_dir.resolve()
    required = [c for c in CONDITION_ORDER if c not in skip]
    missing = [c for c in required if not (parts_dir / f"{c}.json").is_file()]
    if missing:
        raise SystemExit(f"Missing part files in {parts_dir}: {missing}")

    benchmark: dict = {}
    ckpt_path: Path | None = None
    for name in required:
        data = json.loads((parts_dir / f"{name}.json").read_text())
        ckpt_path = Path(data["checkpoint"])
        benchmark[name] = data["result"]
        if name == "baseline" and "volume_strata" in data:
            benchmark["volume_strata"] = data["volume_strata"]

    assert ckpt_path is not None
    benchmark["checkpoint"] = str(ckpt_path)
    curriculum = _load_curriculum_dice(curriculum_dir)

    import torch
    from monai.networks.nets import SwinUNETR

    device = torch.device("cpu")
    model = SwinUNETR(in_channels=4, out_channels=3, feature_size=48, use_checkpoint=False)
    try:
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model_stats = _model_size_stats(model, ckpt_path)

    final = build_final_report(benchmark, curriculum, model_stats, PERTURBATION_CONFIG)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_final_artifacts(final, out_dir, ckpt_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parts-dir", type=Path, default=Path("/scratch/users/linika/checkpoints/benchmark/parts"))
    p.add_argument("--curriculum-dir", type=Path, default=Path("/scratch/users/linika/checkpoints/curriculum"))
    p.add_argument("--out-dir", type=Path, default=Path("/scratch/users/linika/checkpoints/benchmark"))
    p.add_argument(
        "--skip",
        nargs="*",
        default=(),
        help="Conditions to omit (e.g. bias_field if not run yet).",
    )
    args = p.parse_args()
    merge_parts(args.parts_dir, args.curriculum_dir, args.out_dir, skip=frozenset(args.skip))


if __name__ == "__main__":
    main()
