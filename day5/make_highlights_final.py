#!/usr/bin/env python3
"""
Generate reproducible documentation for the `_final` no-filter evaluation.

Outputs by default:
  evaluation/results/highlights_summary_final.txt
  evaluation/results/highlights_summary_final.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "evaluation/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate highlights summaries for the `_final` results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing the `_final` result files.",
    )
    parser.add_argument(
        "--txt-name",
        default="highlights_summary_final.txt",
        help="Output plain-text filename inside results-dir.",
    )
    parser.add_argument(
        "--md-name",
        default="highlights_summary_final.md",
        help="Output markdown filename inside results-dir.",
    )
    return parser.parse_args()


def get(df: pd.DataFrame, metric: str, column: str = "mean") -> float:
    return float(df.loc[metric, column])


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}%"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def markdown_table(headers: list[str], rows: Iterable[Iterable[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main() -> None:
    args = parse_args()
    r = args.results_dir

    unet_agg = pd.read_csv(r / "unet_aggregate_final.csv", index_col=0)
    nn_agg = pd.read_csv(r / "nnunet_aggregate_final.csv", index_col=0)
    unet_gp = pd.read_csv(r / "unet_aggregate_gt_positive_final.csv", index_col=0)
    nn_gp = pd.read_csv(r / "nnunet_aggregate_gt_positive_final.csv", index_col=0)
    unet_old = pd.read_csv(r / "unet_aggregate.csv", index_col=0)
    nn_old = pd.read_csv(r / "nnunet_aggregate.csv", index_col=0)

    with open(r / "unet_summary_final.json") as f:
        unet_summary = json.load(f)
    with open(r / "nnunet_summary_final.json") as f:
        nnunet_summary = json.load(f)
    with open(r / "unet_corner_case_summary_final.json") as f:
        unet_cc = json.load(f)
    with open(r / "nnunet_corner_case_summary_final.json") as f:
        nnunet_cc = json.load(f)
    with open(r / "stats_tests_summary_final.json") as f:
        stats_summary = json.load(f)

    fg_quart = pd.read_csv(r / "stratified_fg_quartile_final.csv")
    q4 = fg_quart[fg_quart["stratum"] == "Q4 (high)"].iloc[0]

    output_files = sorted(
        p.name
        for p in r.glob("*_final.csv")
    ) + sorted(
        p.name
        for p in r.glob("*_final.json")
        if p.name not in {args.txt_name, args.md_name}
    )

    pixel_rows = [
        ("Foreground Dice", fmt(get(unet_agg, "dice_fg_t50")), fmt(get(nn_agg, "dice_fg_t50"))),
        ("IoU", fmt(get(unet_agg, "iou_t50")), fmt(get(nn_agg, "iou_t50"))),
        ("Precision", fmt(get(unet_agg, "precision_t50")), fmt(get(nn_agg, "precision_t50"))),
        ("Recall", fmt(get(unet_agg, "recall_t50")), fmt(get(nn_agg, "recall_t50"))),
        ("HD95", f"{get(unet_agg, 'hd95_t50'):.2f}", f"{get(nn_agg, 'hd95_t50'):.2f}"),
        ("Boundary F1 (2 px)", fmt(get(unet_agg, "boundary_f1_2px_t50")), fmt(get(nn_agg, "boundary_f1_2px_t50"))),
        ("Boundary F1 (5 px)", fmt(get(unet_agg, "boundary_f1_5px_t50")), fmt(get(nn_agg, "boundary_f1_5px_t50"))),
        ("ECE", fmt(get(unet_agg, "ece")), fmt(get(nn_agg, "ece"))),
    ]

    object_rows = [
        ("Predicted objects per slice", fmt(get(unet_agg, "pred_objects")), fmt(get(nn_agg, "pred_objects"))),
        ("Ground-truth objects per slice", fmt(get(unet_agg, "gt_objects")), fmt(get(nn_agg, "gt_objects"))),
        ("Object count error", fmt(get(unet_agg, "obj_count_error")), fmt(get(nn_agg, "obj_count_error"))),
        ("Signed count error", fmt(get(unet_agg, "obj_count_error_signed")), fmt(get(nn_agg, "obj_count_error_signed"))),
        ("Merge rate", fmt(get(unet_agg, "merge_rate")), fmt(get(nn_agg, "merge_rate"))),
        ("Split rate", fmt(get(unet_agg, "split_rate")), fmt(get(nn_agg, "split_rate"))),
        ("Connectivity error", fmt(get(unet_agg, "connectivity_error")), fmt(get(nn_agg, "connectivity_error"))),
        ("Matched IoU mean", fmt(get(unet_agg, "matched_iou_mean")), fmt(get(nn_agg, "matched_iou_mean"))),
        ("PQ (IoU > 0.5)", fmt(get(unet_agg, "pq_iou50")), fmt(get(nn_agg, "pq_iou50"))),
        ("Object precision @0.5", fmt(get(unet_agg, "obj_precision_t50")), fmt(get(nn_agg, "obj_precision_t50"))),
        ("Object recall @0.5", fmt(get(unet_agg, "obj_recall_t50")), fmt(get(nn_agg, "obj_recall_t50"))),
        ("Object precision @0.9", fmt(get(unet_agg, "obj_precision_t90")), fmt(get(nn_agg, "obj_precision_t90"))),
        ("Object recall @0.9", fmt(get(unet_agg, "obj_recall_t90")), fmt(get(nn_agg, "obj_recall_t90"))),
    ]

    gp_rows = [
        ("Predicted objects per slice", fmt(get(unet_gp, "pred_objects")), fmt(get(nn_gp, "pred_objects"))),
        ("Ground-truth objects per slice", fmt(get(unet_gp, "gt_objects")), fmt(get(nn_gp, "gt_objects"))),
        ("Object count error", fmt(get(unet_gp, "obj_count_error")), fmt(get(nn_gp, "obj_count_error"))),
        ("Merge rate", fmt(get(unet_gp, "merge_rate")), fmt(get(nn_gp, "merge_rate"))),
        ("Split rate", fmt(get(unet_gp, "split_rate")), fmt(get(nn_gp, "split_rate"))),
        ("Connectivity error", fmt(get(unet_gp, "connectivity_error")), fmt(get(nn_gp, "connectivity_error"))),
        ("Matched IoU mean", fmt(get(unet_gp, "matched_iou_mean")), fmt(get(nn_gp, "matched_iou_mean"))),
        ("PQ (IoU > 0.5)", fmt(get(unet_gp, "pq_iou50")), fmt(get(nn_gp, "pq_iou50"))),
        ("Object precision @0.5", fmt(get(unet_gp, "obj_precision_t50")), fmt(get(nn_gp, "obj_precision_t50"))),
        ("Object recall @0.5", fmt(get(unet_gp, "obj_recall_t50")), fmt(get(nn_gp, "obj_recall_t50"))),
        ("Object precision @0.9", fmt(get(unet_gp, "obj_precision_t90")), fmt(get(nn_gp, "obj_precision_t90"))),
        ("Object recall @0.9", fmt(get(unet_gp, "obj_recall_t90")), fmt(get(nn_gp, "obj_recall_t90"))),
    ]

    txt_lines: list[str] = []
    txt_lines.append("No-Filter Evaluation Highlights (_final)")
    txt_lines.append("======================================")
    txt_lines.append("")
    txt_lines.append("Evaluation setting:")
    txt_lines.append("- min-object-size = 1")
    txt_lines.append(f"- {unet_summary['cases_evaluated']} / {unet_summary['cases_total']} test cases evaluated for both models")
    txt_lines.append(f"- {unet_summary['cases_gt_positive']} GT-positive cases")
    txt_lines.append("")
    txt_lines.append("Main statistical result:")
    txt_lines.append(f"- {stats_summary['n_significant_bh005']} / {stats_summary['n_metrics_tested']} metrics significant after BH correction")
    txt_lines.append(f"- {stats_summary['n_nnunet_better']} / {stats_summary['n_metrics_tested']} metrics favour nnU-Net")
    txt_lines.append(f"- {stats_summary['n_unet_better']} / {stats_summary['n_metrics_tested']} metrics favour U-Net")
    txt_lines.append("")
    txt_lines.append("Main pixel metrics at threshold 0.5 (means):")
    for label, u, n in pixel_rows:
        txt_lines.append(f"- {label + ':':<24} U-Net {u:<8} | nnU-Net {n}")
    txt_lines.append("")
    txt_lines.append("Main object metrics, all cases (means):")
    for label, u, n in object_rows:
        txt_lines.append(f"- {label + ':':<24} U-Net {u:<8} | nnU-Net {n}")
    txt_lines.append("")
    txt_lines.append("GT-positive object metrics (means):")
    for label, u, n in gp_rows:
        txt_lines.append(f"- {label + ':':<24} U-Net {u:<8} | nnU-Net {n}")
    txt_lines.append("")
    txt_lines.append("High-Dice corner cases:")
    txt_lines.append("- Rule: Dice > 0.85 and (obj_count_error > 4 or merge_rate > 0 or split_rate > 0)")
    txt_lines.append(f"- U-Net high-Dice cases: {unet_cc['counts']['high_dice_cases']}")
    txt_lines.append(f"- U-Net corner cases among high-Dice cases: {unet_cc['corner_cases']} / {unet_cc['counts']['high_dice_cases']} = {pct(unet_cc['corner_rate_of_high_dice'])}")
    txt_lines.append(f"- nnU-Net high-Dice cases: {nnunet_cc['counts']['high_dice_cases']}")
    txt_lines.append(f"- nnU-Net corner cases among high-Dice cases: {nnunet_cc['corner_cases']} / {nnunet_cc['counts']['high_dice_cases']} = {pct(nnunet_cc['corner_rate_of_high_dice'])}")
    txt_lines.append(f"- All-case corner rate: U-Net {pct(unet_cc['corner_rate'], 2)}, nnU-Net {pct(nnunet_cc['corner_rate'], 2)}")
    txt_lines.append("")
    txt_lines.append("Foreground-density quartiles (non-empty cases):")
    for quartile in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
        row = fg_quart[fg_quart["stratum"] == quartile].iloc[0]
        txt_lines.append(f"- {quartile} Dice:  U-Net {row['unet_dice_fg_t50_mean']:.3f} | nnU-Net {row['nnunet_dice_fg_t50_mean']:.3f}")
        txt_lines.append(f"- {quartile} PQ:    U-Net {row['unet_pq_iou50_mean']:.3f} | nnU-Net {row['nnunet_pq_iou50_mean']:.3f}")
        txt_lines.append(f"- {quartile} split: U-Net {row['unet_split_rate_mean']:.3f} | nnU-Net {row['nnunet_split_rate_mean']:.3f}")
        txt_lines.append("")
    txt_lines.append("Interpretation:")
    txt_lines.append("- The Dice gap narrows in dense slices, but the object-level gap remains large.")
    txt_lines.append("- The no-filter analysis therefore strengthens the claim that voxel-level metrics alone understate the biological difference between the models.")
    txt_lines.append("")
    txt_lines.append("Sensitivity relative to the original filtered evaluation:")
    txt_lines.append("")
    txt_lines.append("U-Net:")
    txt_lines.append(f"- Predicted objects:    {fmt(get(unet_old, 'pred_objects'))} -> {fmt(get(unet_agg, 'pred_objects'))}")
    txt_lines.append(f"- Count error:         {fmt(get(unet_old, 'obj_count_error'))} -> {fmt(get(unet_agg, 'obj_count_error'))}")
    txt_lines.append(f"- Split rate:          {fmt(get(unet_old, 'split_rate'))} -> {fmt(get(unet_agg, 'split_rate'))}")
    txt_lines.append(f"- Connectivity error:  {fmt(get(unet_old, 'connectivity_error'))} -> {fmt(get(unet_agg, 'connectivity_error'))}")
    txt_lines.append(f"- PQ:                  {fmt(get(unet_old, 'pq_iou50'))} -> {fmt(get(unet_agg, 'pq_iou50'))}")
    txt_lines.append("")
    txt_lines.append("nnU-Net:")
    txt_lines.append(f"- Predicted objects:    {fmt(get(nn_old, 'pred_objects'))} -> {fmt(get(nn_agg, 'pred_objects'))}")
    txt_lines.append(f"- Count error:         {fmt(get(nn_old, 'obj_count_error'))} -> {fmt(get(nn_agg, 'obj_count_error'))}")
    txt_lines.append(f"- Split rate:          {fmt(get(nn_old, 'split_rate'))} -> {fmt(get(nn_agg, 'split_rate'))}")
    txt_lines.append(f"- Connectivity error:  {fmt(get(nn_old, 'connectivity_error'))} -> {fmt(get(nn_agg, 'connectivity_error'))}")
    txt_lines.append(f"- PQ:                  {fmt(get(nn_old, 'pq_iou50'))} -> {fmt(get(nn_agg, 'pq_iou50'))}")
    txt_lines.append("")
    txt_lines.append("Bottom line:")
    txt_lines.append("- Removing the small-object filter affects U-Net much more strongly than nnU-Net.")
    txt_lines.append("- The ranking does not reverse.")
    txt_lines.append("- The no-filter evaluation is stronger and more defensible for the thesis.")

    md_lines: list[str] = []
    md_lines.append("# No-Filter Evaluation Highlights (`_final`)")
    md_lines.append("")
    md_lines.append("This summary documents the no-filter object-level evaluation produced by the `_final` pipeline.")
    md_lines.append("")
    md_lines.append("## Scope")
    md_lines.append("")
    md_lines.append("- Evaluation setting: `min-object-size = 1`")
    md_lines.append(f"- Test cases evaluated: `{unet_summary['cases_evaluated']} / {unet_summary['cases_total']}` for both models")
    md_lines.append(f"- Ground-truth-positive cases: `{unet_summary['cases_gt_positive']}`")
    md_lines.append("")
    md_lines.append("## Output Files")
    md_lines.append("")
    for name in output_files:
        md_lines.append(f"- `{name}`")
    md_lines.append("")
    md_lines.append("## Headline Result")
    md_lines.append("")
    md_lines.append("- Metrics tested: `{}`".format(stats_summary["n_metrics_tested"]))
    md_lines.append("- Significant after BH correction: `{} / {}`".format(stats_summary["n_significant_bh005"], stats_summary["n_metrics_tested"]))
    md_lines.append("- Metrics favouring nnU-Net: `{} / {}`".format(stats_summary["n_nnunet_better"], stats_summary["n_metrics_tested"]))
    md_lines.append("- Metrics favouring U-Net: `{} / {}`".format(stats_summary["n_unet_better"], stats_summary["n_metrics_tested"]))
    md_lines.append("")
    md_lines.append("## Main Pixel Metrics at Threshold 0.5")
    md_lines.append("")
    md_lines.extend(markdown_table(["Metric", "U-Net", "nnU-Net"], pixel_rows))
    md_lines.append("")
    md_lines.append("## Main Object Metrics")
    md_lines.append("")
    md_lines.extend(markdown_table(["Metric", "U-Net", "nnU-Net"], object_rows))
    md_lines.append("")
    md_lines.append("## GT-Positive Object Metrics")
    md_lines.append("")
    md_lines.extend(markdown_table(["Metric", "U-Net", "nnU-Net"], gp_rows))
    md_lines.append("")
    md_lines.append("## High-Dice Corner Cases")
    md_lines.append("")
    md_lines.append("- Rule: `Dice > 0.85` and (`obj_count_error > 4` or `merge_rate > 0` or `split_rate > 0`)")
    md_lines.append(f"- U-Net high-Dice cases: `{unet_cc['counts']['high_dice_cases']}`")
    md_lines.append(f"- U-Net corner cases among high-Dice cases: `{unet_cc['corner_cases']} / {unet_cc['counts']['high_dice_cases']} = {pct(unet_cc['corner_rate_of_high_dice'])}`")
    md_lines.append(f"- nnU-Net high-Dice cases: `{nnunet_cc['counts']['high_dice_cases']}`")
    md_lines.append(f"- nnU-Net corner cases among high-Dice cases: `{nnunet_cc['corner_cases']} / {nnunet_cc['counts']['high_dice_cases']} = {pct(nnunet_cc['corner_rate_of_high_dice'])}`")
    md_lines.append(f"- All-case corner rate: U-Net `{pct(unet_cc['corner_rate'], 2)}`, nnU-Net `{pct(nnunet_cc['corner_rate'], 2)}`")
    md_lines.append("")
    md_lines.append("## Stratified Analysis Highlights")
    md_lines.append("")
    quart_rows = []
    for quartile in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
        row = fg_quart[fg_quart["stratum"] == quartile].iloc[0]
        quart_rows.append([
            quartile,
            f"{row['unet_dice_fg_t50_mean']:.3f}",
            f"{row['nnunet_dice_fg_t50_mean']:.3f}",
            f"{row['unet_pq_iou50_mean']:.3f}",
            f"{row['nnunet_pq_iou50_mean']:.3f}",
            f"{row['unet_split_rate_mean']:.3f}",
            f"{row['nnunet_split_rate_mean']:.3f}",
        ])
    md_lines.extend(markdown_table(["Quartile", "Dice U-Net", "Dice nnU-Net", "PQ U-Net", "PQ nnU-Net", "Split U-Net", "Split nnU-Net"], quart_rows))
    md_lines.append("")
    md_lines.append(f"- In dense slices (`Q4`), Dice is close: `{q4['unet_dice_fg_t50_mean']:.3f}` vs `{q4['nnunet_dice_fg_t50_mean']:.3f}`.")
    md_lines.append(f"- The object-level gap remains large in the same slices: PQ `{q4['unet_pq_iou50_mean']:.3f}` vs `{q4['nnunet_pq_iou50_mean']:.3f}`, split rate `{q4['unet_split_rate_mean']:.3f}` vs `{q4['nnunet_split_rate_mean']:.3f}`.")
    md_lines.append("")
    md_lines.append("## Sensitivity Relative to the Original Filtered Evaluation")
    md_lines.append("")
    md_lines.append("### U-Net")
    md_lines.append("")
    md_lines.append(f"- Predicted objects: `{fmt(get(unet_old, 'pred_objects'))} -> {fmt(get(unet_agg, 'pred_objects'))}`")
    md_lines.append(f"- Object count error: `{fmt(get(unet_old, 'obj_count_error'))} -> {fmt(get(unet_agg, 'obj_count_error'))}`")
    md_lines.append(f"- Split rate: `{fmt(get(unet_old, 'split_rate'))} -> {fmt(get(unet_agg, 'split_rate'))}`")
    md_lines.append(f"- Connectivity error: `{fmt(get(unet_old, 'connectivity_error'))} -> {fmt(get(unet_agg, 'connectivity_error'))}`")
    md_lines.append(f"- PQ: `{fmt(get(unet_old, 'pq_iou50'))} -> {fmt(get(unet_agg, 'pq_iou50'))}`")
    md_lines.append("")
    md_lines.append("### nnU-Net")
    md_lines.append("")
    md_lines.append(f"- Predicted objects: `{fmt(get(nn_old, 'pred_objects'))} -> {fmt(get(nn_agg, 'pred_objects'))}`")
    md_lines.append(f"- Object count error: `{fmt(get(nn_old, 'obj_count_error'))} -> {fmt(get(nn_agg, 'obj_count_error'))}`")
    md_lines.append(f"- Split rate: `{fmt(get(nn_old, 'split_rate'))} -> {fmt(get(nn_agg, 'split_rate'))}`")
    md_lines.append(f"- Connectivity error: `{fmt(get(nn_old, 'connectivity_error'))} -> {fmt(get(nn_agg, 'connectivity_error'))}`")
    md_lines.append(f"- PQ: `{fmt(get(nn_old, 'pq_iou50'))} -> {fmt(get(nn_agg, 'pq_iou50'))}`")
    md_lines.append("")
    md_lines.append("## Bottom Line")
    md_lines.append("")
    md_lines.append("- Removing the small-object filter affects U-Net much more strongly than nnU-Net.")
    md_lines.append("- The ranking does not reverse.")
    md_lines.append("- The no-filter evaluation is stronger and more defensible for the thesis.")

    txt_path = r / args.txt_name
    md_path = r / args.md_name
    txt_path.write_text("\n".join(txt_lines) + "\n")
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Written: {txt_path}")
    print(f"Written: {md_path}")


if __name__ == "__main__":
    main()
