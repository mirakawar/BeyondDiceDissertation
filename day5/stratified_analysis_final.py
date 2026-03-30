"""
Stratified analysis of U-Net vs nnU-Net metrics from the `_final` evaluator outputs.

Stratifications:
  1. Foreground ratio quartile (Q1–Q4 of GT fg pixels, excluding empty slices)
  2. GT object count quartile (Q1–Q4 of gt_objects, excluding empty slices)
  3. Sample type proxy: _2D / _3D / other (from sample name suffix in nnunet_id_map)

Outputs to evaluation/results/:
  stratified_fg_quartile_final.csv
  stratified_obj_count_quartile_final.csv
  stratified_sample_type_final.csv
  stratified_empty_vs_nonempty_final.csv
  stratified_summary_final.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "evaluation/results"
ID_MAP_PATH = REPO_ROOT / "data/nnunet_id_map.json"
INPUT_SUFFIX = "_final"
OUTPUT_SUFFIX = "_final"

# Key metrics to report in stratified tables
KEY_METRICS = [
    "dice_fg_t50",
    "iou_t50",
    "precision_t50",
    "recall_t50",
    "hd95_t50",
    "boundary_f1_2px_t50",
    "boundary_f1_5px_t50",
    "ece",
    "obj_count_error",
    "merge_rate",
    "split_rate",
    "pq_iou50",
    "obj_precision_t50",
    "obj_recall_t50",
]


def load_and_merge():
    """Load per-image results and attach metadata from id_map."""
    unet = pd.read_csv(RESULTS_DIR / f"unet_per_image{INPUT_SUFFIX}.csv")
    nnunet = pd.read_csv(RESULTS_DIR / f"nnunet_per_image{INPUT_SUFFIX}.csv")

    with open(ID_MAP_PATH) as f:
        id_map = json.load(f)

    # Build metadata dataframe
    meta_rows = []
    for case_id, info in id_map.items():
        if info["split"] == "test":
            sample = info["sample"]
            if sample.endswith("_3D"):
                stype = "3D_volumeEM"
            elif sample.endswith("_2D"):
                stype = "2D_TEM"
            else:
                stype = "other"
            meta_rows.append({
                "case_id": case_id,
                "sample": sample,
                "fg_ratio": info["fg_ratio"],
                "n_instances": info["n_instances"],
                "is_empty": info["is_empty"],
                "sample_type": stype,
            })
    meta = pd.DataFrame(meta_rows)

    unet = unet.merge(meta, on="case_id", how="left")
    nnunet = nnunet.merge(meta, on="case_id", how="left")

    return unet, nnunet, meta


def stratified_table(unet, nnunet, group_col, group_labels=None):
    """For each stratum, compute mean of key metrics for both models."""
    rows = []
    groups = sorted(unet[group_col].dropna().unique())
    for g in groups:
        label = group_labels.get(g, str(g)) if group_labels else str(g)
        u_g = unet[unet[group_col] == g]
        n_g = nnunet[nnunet[group_col] == g]
        row = {"stratum": label, "n_images": len(u_g)}
        for m in KEY_METRICS:
            if m in u_g.columns:
                u_vals = pd.to_numeric(u_g[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
                n_vals = pd.to_numeric(n_g[m], errors="coerce").replace([np.inf, -np.inf], np.nan)
                row[f"unet_{m}_mean"] = float(u_vals.mean())
                row[f"unet_{m}_median"] = float(u_vals.median())
                row[f"nnunet_{m}_mean"] = float(n_vals.mean())
                row[f"nnunet_{m}_median"] = float(n_vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def save_fg_quartile_plot(fg_table: pd.DataFrame) -> None:
    """Save the Chapter 4 foreground-density stratification plot from `_final` results."""
    out_dir = REPO_ROOT / "figures" / "stratified"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the Q1->Q4 order used in the table.
    plot_df = fg_table.copy()
    x = np.arange(len(plot_df))
    labels = ["Q1", "Q2", "Q3", "Q4"]

    unet_c = "#c94c4c"
    nnunet_c = "#4f79bd"

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0))
    metrics = [
        ("dice_fg_t50_mean", "Foreground Dice", (0.35, 0.80)),
        ("pq_iou50_mean", "Panoptic Quality", (0.25, 0.65)),
        ("split_rate_mean", "Split Rate", (0.00, 0.25)),
    ]

    for ax, (metric, title, ylim) in zip(axes, metrics):
        ax.plot(x, plot_df[f"unet_{metric}"], marker="o", linewidth=2.0,
                markersize=6.5, color=unet_c, label="U-Net")
        ax.plot(x, plot_df[f"nnunet_{metric}"], marker="o", linewidth=2.0,
                markersize=6.5, color=nnunet_c, label="nnU-Net")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    plt.tight_layout()
    out_path = out_dir / "fg_density_quartiles.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("Loading data...")
    unet, nnunet, meta = load_and_merge()
    print(f"  U-Net rows: {len(unet)}, nnU-Net rows: {len(nnunet)}")

    # ----------------------------------------------------------------
    # 1. Foreground ratio quartile (exclude empty slices)
    # ----------------------------------------------------------------
    non_empty_mask = unet["is_empty"] == False  # noqa: E712
    u_ne = unet[non_empty_mask].copy()
    n_ne = nnunet[non_empty_mask].copy()

    fg_quartiles = pd.qcut(u_ne["fg_ratio"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    u_ne = u_ne.copy()
    n_ne = n_ne.copy()
    u_ne["fg_quartile"] = fg_quartiles.values
    n_ne["fg_quartile"] = fg_quartiles.values

    fg_table = stratified_table(u_ne, n_ne, "fg_quartile")

    # Add quartile fg_ratio ranges
    bins = pd.qcut(u_ne["fg_ratio"], q=4, retbins=True)[1]
    fg_table["fg_ratio_range"] = [
        f"{bins[i]:.3f}–{bins[i+1]:.3f}" for i in range(4)
    ]

    out = RESULTS_DIR / f"stratified_fg_quartile{OUTPUT_SUFFIX}.csv"
    fg_table.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    _print_compact(fg_table, "FOREGROUND RATIO QUARTILE")
    save_fg_quartile_plot(fg_table)

    # ----------------------------------------------------------------
    # 2. GT object count quartile (exclude empty slices)
    # ----------------------------------------------------------------
    obj_quartiles = pd.qcut(u_ne["gt_objects"], q=4,
                            labels=["Q1 (few)", "Q2", "Q3", "Q4 (many)"],
                            duplicates="drop")
    u_ne2 = u_ne.copy()
    n_ne2 = n_ne.copy()
    u_ne2["obj_quartile"] = obj_quartiles.values
    n_ne2["obj_quartile"] = obj_quartiles.values

    obj_table = stratified_table(u_ne2, n_ne2, "obj_quartile")

    bins_obj = pd.qcut(u_ne["gt_objects"], q=4, retbins=True, duplicates="drop")[1]
    obj_table["gt_obj_range"] = [
        f"{int(bins_obj[i])}–{int(bins_obj[i+1])}" for i in range(len(bins_obj) - 1)
    ]

    out = RESULTS_DIR / f"stratified_obj_count_quartile{OUTPUT_SUFFIX}.csv"
    obj_table.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    _print_compact(obj_table, "GT OBJECT COUNT QUARTILE")

    # ----------------------------------------------------------------
    # 3. Sample type proxy (_3D / _2D / other)
    # ----------------------------------------------------------------
    type_table = stratified_table(unet, nnunet, "sample_type")
    out = RESULTS_DIR / f"stratified_sample_type{OUTPUT_SUFFIX}.csv"
    type_table.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    _print_compact(type_table, "SAMPLE TYPE (proxy for imaging modality)")

    # ----------------------------------------------------------------
    # Also: include ALL images (including empty) for FG quartile with
    # a dedicated "empty" stratum
    # ----------------------------------------------------------------
    unet["has_fg"] = ~unet["is_empty"]
    nnunet["has_fg"] = ~nnunet["is_empty"]
    empty_table = stratified_table(unet, nnunet, "has_fg",
                                   group_labels={True: "non-empty", False: "empty"})
    out = RESULTS_DIR / f"stratified_empty_vs_nonempty{OUTPUT_SUFFIX}.csv"
    empty_table.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary JSON
    summary = {
        "stratifications": [
            "fg_quartile (non-empty slices only)",
            "obj_count_quartile (non-empty slices only)",
            "sample_type (_2D/_3D/other proxy)",
            "empty_vs_nonempty",
        ],
        "key_metrics": KEY_METRICS,
        "n_test_images": len(unet),
        "n_non_empty": int(non_empty_mask.sum()),
    }
    with open(RESULTS_DIR / f"stratified_summary{OUTPUT_SUFFIX}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR}/stratified_summary{OUTPUT_SUFFIX}.json")


def _print_compact(df, title):
    """Print a compact readable version of key metrics."""
    print(f"\n=== {title} ===")
    cols = ["stratum", "n_images",
            "unet_dice_fg_t50_mean", "nnunet_dice_fg_t50_mean",
            "unet_obj_count_error_mean", "nnunet_obj_count_error_mean",
            "unet_pq_iou50_mean", "nnunet_pq_iou50_mean",
            "unet_split_rate_mean", "nnunet_split_rate_mean"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
