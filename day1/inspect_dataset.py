"""
Day 1 - Script 1: inspect_dataset.py
=====================================
Inspects all sample folders and their image/mask pairs.

CEM-MitoLab specific:
  - Masks are INSTANCE labels (0 = background, 1-255 = individual mitochondria)
  - Instance labels are EXPECTED and not flagged as errors
  - n_instances per slice is recorded for use in biological metrics later

Reports:
  - Folder structure validity
  - Image/mask shape and dtype
  - Mask encoding (instance labels, binary, empty)
  - Mismatched filenames between images/ and masks/
  - Empty masks (all background — no mitochondria in patch)
  - Shape mismatches between image and mask

Outputs:
  /home/s2292949/dissertation_resplit/data/inspection_report.csv
  /home/s2292949/dissertation_resplit/data/inspection_summary.txt
"""

import csv
import numpy as np
from pathlib import Path
from tifffile import imread
from collections import defaultdict

DATA_ROOT  = "/home/s2292949/dissertation_resplit/data/cem_mitolab"
OUTPUT_DIR = "/home/s2292949/dissertation_resplit/data"

def detect_mask_encoding(mask: np.ndarray):
    """
    Returns (encoding_label, n_instances, is_issue, issue_message).
    Instance labels (0-255) are EXPECTED for CEM-MitoLab — not flagged as errors.
    """
    unique_vals  = np.unique(mask).tolist()
    n_instances  = len([v for v in unique_vals if v > 0])

    if unique_vals == [0]:
        return "empty", 0, True, "all background"

    elif set(unique_vals) == {0, 1}:
        return "binary_semantic", 1, False, ""

    elif set(unique_vals) == {0, 255}:
        return "0/255_semantic", 1, True, "mask is 0/255 — needs converting to 0/1"

    elif min(unique_vals) == 0 and max(unique_vals) <= 255 and len(unique_vals) > 2:

        return f"instance_{n_instances}_objects", n_instances, False, ""

    else:
        return f"unknown:{unique_vals}", 0, True, f"unexpected mask values {unique_vals}"

def inspect_dataset(data_root: str, output_dir: str):
    data_root  = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_folders = sorted([f for f in data_root.iterdir() if f.is_dir()])
    print(f"Found {len(sample_folders)} sample folders\n")

    issues = []
    rows   = []

    for sample in sample_folders:
        images_dir = sample / "images"
        masks_dir  = sample / "masks"

        if not images_dir.exists():
            issues.append(f"[MISSING] {sample.name}: no images/ subfolder")
            continue
        if not masks_dir.exists():
            issues.append(f"[MISSING] {sample.name}: no masks/ subfolder")
            continue

        image_files = sorted(images_dir.glob("*.tif")) + sorted(images_dir.glob("*.tiff"))
        mask_files  = sorted(masks_dir.glob("*.tif"))  + sorted(masks_dir.glob("*.tiff"))

        image_names = {f.name for f in image_files}
        mask_names  = {f.name for f in mask_files}

        only_in_images = image_names - mask_names
        only_in_masks  = mask_names  - image_names
        if only_in_images:
            issues.append(f"[MISMATCH] {sample.name}: images without masks: {only_in_images}")
        if only_in_masks:
            issues.append(f"[MISMATCH] {sample.name}: masks without images: {only_in_masks}")

        for fname in sorted(image_names & mask_names):
            img  = imread(str(images_dir / fname))
            mask = imread(str(masks_dir  / fname))

            shape_match = img.shape == mask.shape
            if not shape_match:
                issues.append(
                    f"[SHAPE] {sample.name}/{fname}: "
                    f"image {img.shape} vs mask {mask.shape}"
                )

            encoding, n_inst, is_issue, issue_msg = detect_mask_encoding(mask)
            if is_issue:
                issues.append(f"[{encoding.upper()}] {sample.name}/{fname}: {issue_msg}")

            mask_bin     = (mask > 0).astype(np.uint8)
            fg_pixels    = int(np.sum(mask_bin))
            total_pixels = int(mask.size)
            fg_ratio     = fg_pixels / total_pixels if total_pixels > 0 else 0.0

            rows.append({
                "sample":        sample.name,
                "filename":      fname,
                "image_shape":   str(img.shape),
                "image_dtype":   str(img.dtype),
                "mask_dtype":    str(mask.dtype),
                "mask_encoding": encoding,
                "n_instances":   n_inst,
                "fg_pixels":     fg_pixels,
                "total_pixels":  total_pixels,
                "fg_ratio":      round(fg_ratio, 6),
                "is_empty":      n_inst == 0,
                "shape_match":   shape_match,
            })

    csv_path = output_dir / "inspection_report.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Inspection report saved → {csv_path}")

    summary_path = output_dir / "inspection_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"DATASET INSPECTION SUMMARY\n{'='*50}\n\n")
        f.write(f"Total sample folders  : {len(sample_folders)}\n")
        f.write(f"Total slices inspected: {len(rows)}\n\n")

        encodings = defaultdict(int)
        for r in rows:
            encodings[r["mask_encoding"]] += 1
        f.write("Mask encoding breakdown:\n")
        for enc, count in sorted(encodings.items()):
            f.write(f"  {enc}: {count} slices\n")

        inst_counts = [r["n_instances"] for r in rows if r["n_instances"] > 0]
        if inst_counts:
            f.write(f"\nMitochondria instance counts (non-empty slices):\n")
            f.write(f"  total instances : {sum(inst_counts)}\n")
            f.write(f"  mean per slice  : {np.mean(inst_counts):.2f}\n")
            f.write(f"  median per slice: {np.median(inst_counts):.2f}\n")
            f.write(f"  min per slice   : {min(inst_counts)}\n")
            f.write(f"  max per slice   : {max(inst_counts)}\n")

        empty = [r for r in rows if r["is_empty"]]
        f.write(f"\nEmpty masks (no mitochondria): {len(empty)}\n")
        for r in empty[:20]:
            f.write(f"  {r['sample']}/{r['filename']}\n")
        if len(empty) > 20:
            f.write(f"  ... and {len(empty) - 20} more\n")

        mismatches = [r for r in rows if not r["shape_match"]]
        f.write(f"\nShape mismatches: {len(mismatches)}\n")
        for r in mismatches:
            f.write(f"  {r['sample']}/{r['filename']}: {r['image_shape']}\n")

        fg_ratios = [r["fg_ratio"] for r in rows if not r["is_empty"]]
        if fg_ratios:
            f.write(f"\nForeground ratio stats (non-empty slices only):\n")
            f.write(f"  min    : {min(fg_ratios):.4f}\n")
            f.write(f"  max    : {max(fg_ratios):.4f}\n")
            f.write(f"  mean   : {np.mean(fg_ratios):.4f}\n")
            f.write(f"  median : {np.median(fg_ratios):.4f}\n")

        f.write(f"\nAll issues found ({len(issues)}):\n")
        for issue in issues:
            f.write(f"  {issue}\n")
        if not issues:
            f.write("  None — dataset looks clean.\n")

    print(f"Summary saved        → {summary_path}")
    print(f"\nIssues found: {len(issues)}")
    for issue in issues[:10]:
        print(f"  {issue}")
    if len(issues) > 10:
        print(f"  ... and {len(issues) - 10} more (see inspection_summary.txt)")

    inst_counts = [r["n_instances"] for r in rows if r["n_instances"] > 0]
    print(f"\nTotal slices     : {len(rows)}")
    print(f"Empty slices     : {sum(1 for r in rows if r['is_empty'])}")
    print(f"Total instances  : {sum(inst_counts)}")
    print(f"Mean inst/slice  : {np.mean(inst_counts):.2f}")

if __name__ == "__main__":
    inspect_dataset(DATA_ROOT, OUTPUT_DIR)
