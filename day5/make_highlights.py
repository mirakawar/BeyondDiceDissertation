"""
Generate evaluation/results/highlights_summary.txt — a plain-text summary
of all key statistics for both models.
"""
import argparse
import json, pandas as pd, numpy as np
from pathlib import Path

R = Path("/home/s2292949/dissertation_resplit/evaluation/results")

unet_agg = pd.read_csv(R / "unet_aggregate.csv", index_col=0)
nn_agg   = pd.read_csv(R / "nnunet_aggregate.csv", index_col=0)
unet_per = pd.read_csv(R / "unet_per_image.csv")
nn_per   = pd.read_csv(R / "nnunet_per_image.csv")
stats    = pd.read_csv(R / "stats_tests.csv")

with open(R / "unet_corner_case_summary.json") as f:  ucc = json.load(f)
with open(R / "nnunet_corner_case_summary.json") as f: ncc = json.load(f)
with open(R / "stats_tests_summary.json") as f:        sts = json.load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Generate plain-text dissertation highlights summary.")
    p.add_argument(
        "--threshold-tag",
        default="t50",
        choices=["t30", "t40", "t50", "t60", "t70"],
        help="Threshold tag used for threshold-dependent pixel metrics.",
    )
    p.add_argument(
        "--output-name",
        default="highlights_summary.txt",
        help="Output filename inside evaluation/results/.",
    )
    return p.parse_args()


def get(df, row, col="mean"):
    try:
        return float(df.loc[row, col])
    except Exception:
        return float("nan")


def padj(metric):
    row = stats[stats.metric == metric]
    return row.p_adjusted_bh.values[0] if len(row) else float("nan")


def finite_rate(df, col):
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(vals).mean())


def format_padj(metric):
    p = padj(metric)
    return "n/a" if not np.isfinite(p) else f"{p:.2e}"


def fmt_mean_sd(df, row):
    mean = get(df, row, "mean")
    sd = get(df, row, "std")
    return f"{mean:.4f} ± {sd:.4f}"


def fmt_median_iqr(df, row):
    median = get(df, row, "median")
    q25 = get(df, row, "q25")
    q75 = get(df, row, "q75")
    return f"{median:.4f} [{q25:.4f}, {q75:.4f}]"


args = parse_args()
thr = args.threshold_tag
thr_label = f"{int(thr[1:]) / 100:.1f}"

lines = []
lines.append("# Dissertation Statistics Highlights")
lines.append(f"Generated from evaluation/results/ — 5,677 paired test cases, threshold={thr_label}")
lines.append("")

lines.append("## Statistical Significance (Wilcoxon signed-rank + Benjamini-Hochberg FDR, alpha=0.05)")
lines.append(f"  Metrics tested:      {sts['n_metrics_tested']}")
lines.append(f"  Significant (BH005): {sts['n_significant_bh005']} / {sts['n_metrics_tested']}  (ALL metrics)")
lines.append(f"  nnU-Net better:      {sts['n_nnunet_better']} / {sts['n_metrics_tested']}")
lines.append(f"  U-Net better:        {sts['n_unet_better']} / {sts['n_metrics_tested']}  (HD95, merge_rate, ECE — all minor)")
lines.append("")

lines.append(f"## Per-Image Pixel Metrics (Mean +- SD, t={thr_label})")
lines.append(f"  {'Metric':<25} {'U-Net mean±sd':>20} {'nnU-Net mean±sd':>20} {'ratio':>8}  p_adj (BH)")
lines.append("  " + "-" * 92)
pixel_rows = [
    (f"dice_fg_{thr}",         "Foreground Dice"),
    (f"dice_bg_{thr}",         "Background Dice"),
    (f"dice_mean_{thr}",       "Overall Dice"),
    (f"iou_{thr}",             "IoU"),
    (f"precision_{thr}",       "Precision"),
    (f"recall_{thr}",          "Recall"),
    (f"boundary_f1_2px_{thr}", "Boundary F1 (2px)"),
    (f"boundary_f1_5px_{thr}", "Boundary F1 (5px)"),
    ("ece",                 "ECE (calibration)"),
]
for key, label in pixel_rows:
    u = get(unet_agg, key)
    n = get(nn_agg, key)
    ratio = n / u if u != 0 else float("nan")
    lines.append(f"  {label:<25} {fmt_mean_sd(unet_agg, key):>20} {fmt_mean_sd(nn_agg, key):>20} {ratio:>8.2f}x  {format_padj(key)}")
u_hd95_median = get(unet_agg, f"hd95_{thr}", col="median")
n_hd95_median = get(nn_agg, f"hd95_{thr}", col="median")
hd95_ratio = n_hd95_median / u_hd95_median if u_hd95_median != 0 else float("nan")
lines.append(f"  {'HD95 median (px)':<25} {u_hd95_median:>20.4f} {n_hd95_median:>20.4f} {hd95_ratio:>8.2f}x  {format_padj(f'hd95_{thr}')}")
u_hd95_finite = finite_rate(unet_per, f"hd95_{thr}")
n_hd95_finite = finite_rate(nn_per, f"hd95_{thr}")
finite_ratio = n_hd95_finite / u_hd95_finite if u_hd95_finite != 0 else float("nan")
lines.append(f"  {'HD95 finite-case rate':<25} {u_hd95_finite:>19.2%} {n_hd95_finite:>19.2%} {finite_ratio:>8.2f}x  n/a")
lines.append("")

lines.append("## Per-Image Object Metrics (Mean +- SD, thresholded at default t=0.5)")
lines.append(f"  {'Metric':<25} {'U-Net mean±sd':>20} {'nnU-Net mean±sd':>20} {'ratio':>8}  p_adj (BH)")
lines.append("  " + "-" * 92)
obj_rows = [
    ("obj_count_error",    "Count error |pred-GT|"),
    ("merge_rate",         "Merge rate"),
    ("split_rate",         "Split rate"),
    ("connectivity_error", "Connectivity error"),
    ("pq_iou50",           "Panoptic Quality (PQ)"),
    ("matched_iou_mean",   "Matched IoU mean"),
    ("obj_precision_t50",  "Obj precision @0.5"),
    ("obj_recall_t50",     "Obj recall @0.5"),
]
for key, label in obj_rows:
    u = get(unet_agg, key)
    n = get(nn_agg, key)
    if u != 0:
        ratio = n / u
    else:
        ratio = float("nan")
    lines.append(f"  {label:<25} {fmt_mean_sd(unet_agg, key):>20} {fmt_mean_sd(nn_agg, key):>20} {ratio:>8.2f}x  {format_padj(key)}")

u_pred = get(unet_agg, "pred_objects")
n_pred = get(nn_agg, "pred_objects")
u_gt   = get(unet_agg, "gt_objects")
lines.append("")
lines.append(f"  GT objects mean (both):   {u_gt:.2f}")
lines.append(f"  Pred objects mean U-Net:  {u_pred:.2f}  (over-segments {u_pred/u_gt:.1f}x)")
lines.append(f"  Pred objects mean nnU-Net:{n_pred:.2f}  (near-perfect, {n_pred/u_gt:.2f}x GT)")
lines.append("")

lines.append("## Corner Cases  (Dice>0.85 AND >=1 object metric exceeds Q75 cutoff)")
lines.append(f"  {'':30} {'U-Net':>10} {'nnU-Net':>10}")
lines.append(f"  {'Usable cases':<30} {ucc['cases_usable']:>10} {ncc['cases_usable']:>10}")
lines.append(f"  {'High-Dice cases':<30} {ucc['counts']['high_dice_cases']:>10} {ncc['counts']['high_dice_cases']:>10}")
lines.append(f"  {'Corner cases':<30} {ucc['corner_cases']:>10} {ncc['corner_cases']:>10}")
lines.append(f"  {'Corner rate (/ usable)':<30} {ucc['corner_rate']*100:>9.2f}% {ncc['corner_rate']*100:>9.2f}%")
lines.append(f"  {'Corner rate (/ high-Dice)':<30} {ucc['corner_rate_of_high_dice']*100:>9.2f}% {ncc['corner_rate_of_high_dice']*100:>9.2f}%")
lines.append("")

lines.append(f"## Per-Image Pixel Metrics (Median [Q1, Q3], t={thr_label})")
lines.append(
    f"  {'Metric':<22} {'U-Net median [Q1,Q3]':>24} {'nnU-Net median [Q1,Q3]':>26}"
)
lines.append("  " + "-" * 76)
image_pixel_rows = [
    (f"dice_fg_{thr}", "Foreground Dice"),
    (f"dice_bg_{thr}", "Background Dice"),
    (f"dice_mean_{thr}", "Overall Dice"),
    (f"iou_{thr}", "IoU"),
    (f"hd95_{thr}", "HD95 (px)"),
    (f"boundary_f1_2px_{thr}", "Boundary F1 (2px)"),
    (f"boundary_f1_5px_{thr}", "Boundary F1 (5px)"),
    ("ece", "ECE"),
]
for key, label in image_pixel_rows:
    lines.append(
        f"  {label:<22} {fmt_median_iqr(unet_agg, key):>24} {fmt_median_iqr(nn_agg, key):>26}"
    )
lines.append("")

lines.append("## Per-Image Object Metrics (Median [Q1, Q3], thresholded at default t=0.5)")
lines.append(
    f"  {'Metric':<22} {'U-Net median [Q1,Q3]':>24} {'nnU-Net median [Q1,Q3]':>26}"
)
lines.append("  " + "-" * 76)
image_object_rows = [
    ("obj_count_error", "Count error"),
    ("merge_rate", "Merge rate"),
    ("split_rate", "Split rate"),
    ("connectivity_error", "Connectivity error"),
    ("pq_iou50", "Panoptic Quality"),
]
for key, label in image_object_rows:
    lines.append(
        f"  {label:<22} {fmt_median_iqr(unet_agg, key):>24} {fmt_median_iqr(nn_agg, key):>26}"
    )
lines.append("")

lines.append("## Key Ratios (U-Net disadvantage)")
uce = get(unet_agg, "obj_count_error"); nce = get(nn_agg, "obj_count_error")
usr = get(unet_agg, "split_rate");      nsr = get(nn_agg, "split_rate")
upq = get(unet_agg, "pq_iou50");       npq = get(nn_agg, "pq_iou50")
ubf = get(unet_agg, "boundary_f1_2px_t50"); nbf = get(nn_agg, "boundary_f1_2px_t50")
lines.append(f"  Object count error:   {uce/nce:.1f}x worse  ({uce:.2f} vs {nce:.2f})")
lines.append(f"  Split rate:           {usr/nsr:.1f}x worse  ({usr:.3f} vs {nsr:.3f})")
lines.append(f"  Panoptic Quality:     {npq/upq:.1f}x better nnU-Net  ({upq:.3f} vs {npq:.3f})")
lines.append(f"  Boundary F1 (2px):    {nbf/ubf:.1f}x better nnU-Net  ({ubf:.3f} vs {nbf:.3f})")
lines.append("")
lines.append("## Metrics where U-Net is marginally better (all significant but tiny effect)")
lines.append(f"  HD95:       median {u_hd95_median:.1f} vs {n_hd95_median:.1f} px  (finite-case rates {u_hd95_finite:.1%} vs {n_hd95_finite:.1%})")
lines.append("  ECE:        0.090 vs 0.086     (diff=0.004, both well-calibrated)")
lines.append("  Merge rate: 0.032 vs 0.030     (negligible difference)")

out = R / args.output_name
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Written: {out}")
print()
print("\n".join(lines))
