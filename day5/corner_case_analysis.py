#!/usr/bin/env python3
"""
Corner-case analysis for segmentation results.

Definition (per model):
  Dice > dice_threshold
  AND at least one of:
    obj_count_error > obj_count_threshold
    merge_rate      > merge_rate_threshold
    split_rate      > split_rate_threshold

Outputs:
  evaluation/results/{model}_corner_cases.csv
  evaluation/results/{model}_corner_case_summary.json
  evaluation/results/corner_cases_combined.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.json"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corner-case analysis from per-image evaluation CSVs.")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config.json.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=["unet", "nnunet"],
        default=["unet", "nnunet"],
        help="Models to analyze.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override results directory (default from config).",
    )
    p.add_argument(
        "--dice-threshold",
        type=float,
        default=0.85,
        help="Dice foreground threshold for corner-case candidate selection.",
    )
    p.add_argument(
        "--obj-count-threshold",
        type=float,
        default=4.0,
        help="Object-count-error cutoff for a corner case.",
    )
    p.add_argument(
        "--merge-rate-threshold",
        type=float,
        default=0.0,
        help="Merge-rate cutoff for a corner case.",
    )
    p.add_argument(
        "--split-rate-threshold",
        type=float,
        default=0.0,
        help="Split-rate cutoff for a corner case.",
    )
    return p.parse_args()

def required_columns() -> List[str]:
    return ["case_id", "dice_fg", "obj_count_error", "merge_rate", "split_rate"]

def load_config(path: Path) -> Dict:
    return json.loads(path.read_text())

def analyze_model(
    df: pd.DataFrame,
    dice_threshold: float,
    obj_count_threshold: float,
    merge_rate_threshold: float,
    split_rate_threshold: float,
) -> tuple[pd.DataFrame, Dict]:
    cols = required_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[cols].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)

    if work.empty:
        summary = {
            "cases_total": int(len(df)),
            "cases_usable": 0,
            "corner_cases": 0,
            "corner_rate": 0.0,
            "dice_threshold": float(dice_threshold),
            "obj_count_threshold": float(obj_count_threshold),
            "merge_rate_threshold": float(merge_rate_threshold),
            "split_rate_threshold": float(split_rate_threshold),
        }
        return work.iloc[0:0], summary

    work["high_dice"] = work["dice_fg"] > dice_threshold
    work["fail_obj_count_error"] = work["obj_count_error"] > obj_count_threshold
    work["fail_merge_rate"] = work["merge_rate"] > merge_rate_threshold
    work["fail_split_rate"] = work["split_rate"] > split_rate_threshold

    fail_any = (
        work["fail_obj_count_error"]
        | work["fail_merge_rate"]
        | work["fail_split_rate"]
    )
    work["corner_case"] = work["high_dice"] & fail_any

    corner = work[work["corner_case"]].copy()
    if not corner.empty:
        corner["failure_flags"] = (
            corner[[
                "fail_obj_count_error",
                "fail_merge_rate",
                "fail_split_rate",
            ]]
            .apply(
                lambda r: ",".join(
                    flag
                    for flag, ok in [
                        ("obj_count_error", r["fail_obj_count_error"]),
                        ("merge_rate", r["fail_merge_rate"]),
                        ("split_rate", r["fail_split_rate"]),
                    ]
                    if bool(ok)
                ),
                axis=1,
            )
        )
        corner["failure_count"] = (
            corner[[
                "fail_obj_count_error",
                "fail_merge_rate",
                "fail_split_rate",
            ]]
            .sum(axis=1)
            .astype(int)
        )
        corner = corner.sort_values(["failure_count", "dice_fg"], ascending=[False, False])

    n_high_dice = int(work["high_dice"].sum())
    summary = {
        "cases_total": int(len(df)),
        "cases_usable": int(len(work)),
        "corner_cases": int(len(corner)),
        "corner_rate": float(len(corner) / len(work)) if len(work) else 0.0,
        "corner_rate_of_high_dice": float(len(corner) / n_high_dice) if n_high_dice else 0.0,
        "dice_threshold": float(dice_threshold),
        "obj_count_threshold": float(obj_count_threshold),
        "merge_rate_threshold": float(merge_rate_threshold),
        "split_rate_threshold": float(split_rate_threshold),
        "counts": {
            "high_dice_cases": n_high_dice,
            "fail_obj_count_error": int(work["fail_obj_count_error"].sum()),
            "fail_merge_rate": int(work["fail_merge_rate"].sum()),
            "fail_split_rate": int(work["fail_split_rate"].sum()),
        },
    }
    return corner, summary

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = args.results_dir or Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[pd.DataFrame] = []

    print("=" * 72)
    print("Corner-Case Analysis")
    print(f"  results dir    : {results_dir}")
    print(f"  models         : {args.models}")
    print(f"  dice threshold : {args.dice_threshold}")
    print(f"  obj count thr  : {args.obj_count_threshold}")
    print(f"  merge rate thr : {args.merge_rate_threshold}")
    print(f"  split rate thr : {args.split_rate_threshold}")
    print("=" * 72)

    for model in args.models:
        per_image_path = results_dir / f"{model}_per_image.csv"
        if not per_image_path.exists():
            print(f"[{model}] missing per-image CSV: {per_image_path} (skipping)")
            continue

        df = pd.read_csv(per_image_path)
        corner, summary = analyze_model(
            df,
            args.dice_threshold,
            args.obj_count_threshold,
            args.merge_rate_threshold,
            args.split_rate_threshold,
        )

        corner_path = results_dir / f"{model}_corner_cases.csv"
        summary_path = results_dir / f"{model}_corner_case_summary.json"

        corner.insert(1, "model", model)
        corner.to_csv(corner_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2))

        print(
            f"[{model}] corner cases: {summary['corner_cases']} / {summary['cases_usable']} "
            f"({summary['corner_rate']:.2%})"
        )
        print(f"  - {corner_path}")
        print(f"  - {summary_path}")

        if not corner.empty:
            combined_rows.append(corner)

    combined_path = results_dir / "corner_cases_combined.csv"
    if combined_rows:
        pd.concat(combined_rows, ignore_index=True).to_csv(combined_path, index=False)
        print(f"\nCombined corner-case CSV: {combined_path}")
    else:
        pd.DataFrame().to_csv(combined_path, index=False)
        print(f"\nNo corner cases found; wrote empty file: {combined_path}")

if __name__ == "__main__":
    main()
