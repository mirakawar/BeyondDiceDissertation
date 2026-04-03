"""
Statistical significance tests: Wilcoxon signed-rank + Benjamini-Hochberg FDR correction.
Compares U-Net vs nnU-Net across all key metrics on the paired resplit test set
(shared cases are determined automatically from the per-image result tables).

Outputs:
  evaluation/results/stats_tests.csv   -- full results per metric
  evaluation/results/stats_tests_summary.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "evaluation/results"

METRICS = [

    "dice_fg_t50",
    "dice_bg_t50",
    "iou_t50",
    "precision_t50",
    "recall_t50",
    "hd95_t50",
    "boundary_f1_2px_t50",
    "boundary_f1_5px_t50",

    "dice_fg_t30",
    "dice_fg_t40",
    "dice_fg_t60",
    "dice_fg_t70",

    "ece",

    "obj_count_error",
    "merge_rate",
    "split_rate",
    "connectivity_error",
    "pq_iou50",
    "matched_iou_mean",
    "obj_precision_t30",
    "obj_recall_t30",
    "obj_precision_t50",
    "obj_recall_t50",
    "obj_precision_t70",
    "obj_recall_t70",
    "obj_precision_t90",
    "obj_recall_t90",
]

LOWER_IS_BETTER = {
    "hd95_t50",
    "ece",
    "obj_count_error",
    "merge_rate",
    "split_rate",
    "connectivity_error",
}

def bh_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values and rejected flags."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)
    for rank_i, original_i in enumerate(order):
        ranked[original_i] = rank_i + 1

    p_adj = np.minimum(1.0, p * n / ranked)

    for i in range(n - 2, -1, -1):
        orig = order[i]
        orig_next = order[i + 1]
        p_adj[orig] = min(p_adj[orig], p_adj[orig_next])

    rejected = p_adj <= alpha
    return p_adj, rejected

def main():
    unet = pd.read_csv(RESULTS_DIR / "unet_per_image.csv", index_col="case_id")
    nnunet = pd.read_csv(RESULTS_DIR / "nnunet_per_image.csv", index_col="case_id")

    shared = unet.index.intersection(nnunet.index)
    print(f"Shared cases: {len(shared)}")
    unet = unet.loc[shared]
    nnunet = nnunet.loc[shared]

    rows = []
    raw_pvals = []
    valid_metrics = []

    for metric in METRICS:
        if metric not in unet.columns or metric not in nnunet.columns:
            print(f"  SKIP (missing): {metric}")
            continue

        u = unet[metric].values.astype(float)
        n = nnunet[metric].values.astype(float)

        mask = np.isfinite(u) & np.isfinite(n)
        u_clean = u[mask]
        n_clean = n[mask]

        if len(u_clean) < 10:
            print(f"  SKIP (too few valid pairs): {metric}")
            continue

        diff = u_clean - n_clean

        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(u_clean, n_clean, zero_method="wilcox", alternative="two-sided")

        rows.append({
            "metric": metric,
            "n_pairs": int(mask.sum()),
            "mean_unet": float(np.mean(u_clean)),
            "median_unet": float(np.median(u_clean)),
            "mean_nnunet": float(np.mean(n_clean)),
            "median_nnunet": float(np.median(n_clean)),
            "mean_diff_unet_minus_nnunet": float(np.mean(diff)),
            "wilcoxon_statistic": float(stat),
            "p_value": float(p),
        })
        raw_pvals.append(float(p))
        valid_metrics.append(metric)

    p_adj, rejected = bh_correction(raw_pvals, alpha=0.05)
    for i, row in enumerate(rows):
        row["p_adjusted_bh"] = float(p_adj[i])
        row["significant_bh005"] = bool(rejected[i])
        if row["metric"] in LOWER_IS_BETTER:
            row["better_model"] = (
                "U-Net" if row["mean_diff_unet_minus_nnunet"] < 0 else "nnU-Net"
            )
            row["better_direction"] = "lower"
        else:
            row["better_model"] = (
                "U-Net" if row["mean_diff_unet_minus_nnunet"] > 0 else "nnU-Net"
            )
            row["better_direction"] = "higher"

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "stats_tests.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    sig = df[df["significant_bh005"]]
    summary = {
        "n_metrics_tested": len(df),
        "n_significant_bh005": int(sig["significant_bh005"].sum()),
        "n_nnunet_better": int((df["better_model"] == "nnU-Net").sum()),
        "n_unet_better": int((df["better_model"] == "U-Net").sum()),
        "significant_metrics": sig["metric"].tolist(),
    }
    out_json = RESULTS_DIR / "stats_tests_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")

    print("\n=== STATISTICAL RESULTS ===")
    print(f"{'Metric':<30} {'mean U-Net':>12} {'mean nnU-Net':>13} {'p_adj':>10} {'Sig':>5} {'Better':>8}")
    print("-" * 85)
    for _, row in df.iterrows():
        print(
            f"{row['metric']:<30} {row['mean_unet']:>12.4f} {row['mean_nnunet']:>13.4f}"
            f" {row['p_adjusted_bh']:>10.4f} {'*' if row['significant_bh005'] else '':>5}"
            f" {row['better_model']:>8}"
        )

if __name__ == "__main__":
    main()
