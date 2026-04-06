"""
Compute per-slice and per-sample statistics for CEM-MitoLab.

Masks are instance labels (0=background, 1-255=individual mitochondria).
Masks are binarised only for computing fg_ratio — instance counts are
preserved and written out for use in the object-level metrics later.

Reads:  data/cem_mitolab, data/inspection_report.csv
Output: data/dataset_stats.csv (one row per slice)
        data/sample_stats.csv  (one row per sample)
"""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread

REPO_ROOT      = Path(__file__).resolve().parents[1]
DATA_ROOT      = REPO_ROOT / "data" / "cem_mitolab"
INSPECTION_CSV = REPO_ROOT / "data" / "inspection_report.csv"
OUTPUT_DIR     = REPO_ROOT / "data"

def compute_dataset_stats(data_root: str, inspection_csv: str, output_dir: str):
    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inspection = pd.read_csv(inspection_csv)
    broken = set(
        zip(
            inspection[inspection["shape_match"] == False]["sample"],
            inspection[inspection["shape_match"] == False]["filename"],
        )
    )
    print(f"Skipping {len(broken)} broken pairs\n")

    sample_folders = sorted([f for f in data_root.iterdir() if f.is_dir()])
    slice_rows  = []
    sample_rows = []

    for i, sample in enumerate(sample_folders):
        images_dir = sample / "images"
        masks_dir  = sample / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            continue

        image_files = sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff"))
        sample_fg_ratios    = []
        sample_inst_counts  = []
        sample_slice_count  = 0

        for img_path in image_files:
            fname     = img_path.name
            mask_path = masks_dir / fname

            if not mask_path.exists():
                continue
            if (sample.name, fname) in broken:
                continue

            img  = imread(str(img_path))
            mask = imread(str(mask_path))

            unique_vals  = np.unique(mask)
            n_instances  = int(len([v for v in unique_vals if v > 0]))

            mask_bin     = (mask > 0).astype(np.uint8)
            h, w         = img.shape[:2]
            total_pixels = h * w
            fg_pixels    = int(np.sum(mask_bin))
            fg_ratio     = fg_pixels / total_pixels if total_pixels > 0 else 0.0
            is_empty     = fg_pixels == 0

            slice_rows.append({
                "sample":       sample.name,
                "filename":     fname,
                "img_height":   h,
                "img_width":    w,
                "total_pixels": total_pixels,
                "fg_pixels":    fg_pixels,
                "bg_pixels":    total_pixels - fg_pixels,
                "fg_ratio":     round(fg_ratio, 6),
                "bg_ratio":     round(1.0 - fg_ratio, 6),
                "n_instances":  n_instances,
                "is_empty":     is_empty,
                "image_mean":   round(float(np.mean(img)), 4),
                "image_std":    round(float(np.std(img)), 4),
                "image_min":    int(np.min(img)),
                "image_max":    int(np.max(img)),
                "image_dtype":  str(img.dtype),
            })

            sample_fg_ratios.append(fg_ratio)
            sample_inst_counts.append(n_instances)
            sample_slice_count += 1

        if sample_fg_ratios:
            sample_rows.append({
                "sample":              sample.name,
                "num_slices":          sample_slice_count,
                "mean_fg_ratio":       round(float(np.mean(sample_fg_ratios)), 6),
                "median_fg_ratio":     round(float(np.median(sample_fg_ratios)), 6),
                "min_fg_ratio":        round(float(np.min(sample_fg_ratios)), 6),
                "max_fg_ratio":        round(float(np.max(sample_fg_ratios)), 6),
                "std_fg_ratio":        round(float(np.std(sample_fg_ratios)), 6),
                "total_instances":     int(sum(sample_inst_counts)),
                "mean_instances":      round(float(np.mean(sample_inst_counts)), 2),
                "empty_slice_count":   sum(1 for r in sample_fg_ratios if r == 0.0),
            })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sample_folders)} samples...")

    slice_csv = output_dir / "dataset_stats.csv"
    if slice_rows:
        with open(slice_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=slice_rows[0].keys())
            writer.writeheader()
            writer.writerows(slice_rows)
        print(f"\nSlice stats saved  → {slice_csv}  ({len(slice_rows)} slices)")

    sample_csv = output_dir / "sample_stats.csv"
    if sample_rows:
        with open(sample_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sample_rows[0].keys())
            writer.writeheader()
            writer.writerows(sample_rows)
        print(f"Sample stats saved → {sample_csv}  ({len(sample_rows)} samples)")

    all_fg      = [r["fg_ratio"] for r in slice_rows]
    total_fg_px = sum(r["fg_pixels"] for r in slice_rows)
    total_px    = sum(r["total_pixels"] for r in slice_rows)
    all_inst    = [r["n_instances"] for r in slice_rows]
    empty_count = sum(1 for r in slice_rows if r["is_empty"])

    print(f"\n── Global Summary ────────────────────────────────────────")
    print(f"Total slices          : {len(slice_rows)}")
    print(f"Empty slices          : {empty_count} ({100*empty_count/len(slice_rows):.1f}%)")
    print(f"Total instances       : {sum(all_inst)}")
    print(f"Mean instances/slice  : {np.mean([x for x in all_inst if x > 0]):.2f}")
    print(f"Global fg ratio       : {total_fg_px/total_px:.4f} fg  /  "
          f"{1 - total_fg_px/total_px:.4f} bg")
    print(f"Per-slice fg ratio    : mean={np.mean(all_fg):.4f}  "
          f"median={np.median(all_fg):.4f}  "
          f"min={np.min(all_fg):.4f}  max={np.max(all_fg):.4f}")

if __name__ == "__main__":
    compute_dataset_stats(DATA_ROOT, INSPECTION_CSV, OUTPUT_DIR)
