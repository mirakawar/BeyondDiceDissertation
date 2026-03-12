#!/usr/bin/env python3
"""
figures/visualise.py
====================
Comprehensive visualisation script for the CEM-MitoLab dissertation.

Figures produced (in figures/<subdir>/):
  training_panels/  — 4-panel per case: raw EM | GT binary | prediction | error map
  eval_instances/   — instance-label view: raw EM | GT instances | pred binary
  contour_overlap/  — predicted contour overlaid on GT instance colourmap
  corner_cases/     — high Dice but biological metric failure
  scatter/          — Dice vs each biological metric
  threshold_curves/ — metric vs threshold for each model
  convergence/      — train/val curves, both models side by side

Usage:
    python figures/visualise.py                          # all figures
    python figures/visualise.py --mode convergence
    python figures/visualise.py --mode scatter
    python figures/visualise.py --mode training_panels --model nnunet --n 4

Run from any directory; paths are resolved via config.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_erosion, label as scipy_label

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "monospace",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

CONFIG_DEFAULT = Path("/home/s2292949/dissertation_resplit/config.json")

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_cfg(path: Path) -> Dict:
    return json.loads(path.read_text())


def load_nifti(path: Path) -> np.ndarray:
    arr = nib.load(str(path)).get_fdata()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return np.asarray(arr)


def norm01(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img, dtype=np.float32)


def load_probs_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    key = "probabilities" if "probabilities" in data.files else data.files[0]
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 4:
        fg = arr[1, 0] if arr.shape[0] >= 2 else arr[0, 0]
    elif arr.ndim == 3:
        fg = arr[1] if arr.shape[0] >= 2 else arr[0]
    else:
        fg = arr
    return np.clip(fg, 0.0, 1.0)


def maybe_T(arr: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    if arr.shape == target:
        return arr
    if arr.T.shape == target:
        return arr.T
    raise ValueError(f"Cannot align shape {arr.shape} to {target}")


def contour_mask(binary: np.ndarray) -> np.ndarray:
    """Boolean mask of the boundary pixels of a binary mask."""
    binary = binary.astype(bool)
    eroded = binary_erosion(binary, structure=np.ones((3, 3), bool), border_value=0)
    return np.logical_xor(binary, eroded)


def instance_colourmap(inst: np.ndarray, bg_gray: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Map integer instance labels to an RGBA image.
    Label 0 → transparent (or background grey).
    """
    ids = np.unique(inst)
    ids = ids[ids > 0]
    # Use a qualitative colourmap with enough distinct colours
    cmap = plt.get_cmap("tab20", max(len(ids), 1))
    rgb = np.zeros((*inst.shape, 4), dtype=np.float32)
    if bg_gray is not None:
        bg = norm01(bg_gray)
        rgb[..., :3] = np.stack([bg, bg, bg], axis=-1)
        rgb[..., 3] = 1.0
    for i, inst_id in enumerate(ids):
        mask = inst == inst_id
        r, g, b, a = cmap(i % 20)
        rgb[mask, 0] = r
        rgb[mask, 1] = g
        rgb[mask, 2] = b
        rgb[mask, 3] = 1.0
    return rgb


def error_map_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Returns an RGB image:
      TP  → white
      TN  → near-black (#111)
      FP  → red  (#e74c3c)
      FN  → blue (#3498db)
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    rgb  = np.zeros((*pred.shape, 3), dtype=np.float32)
    # TN
    rgb[~pred & ~gt] = [0.07, 0.07, 0.07]
    # TP
    rgb[ pred &  gt] = [1.0,  1.0,  1.0 ]
    # FP
    rgb[ pred & ~gt] = [0.91, 0.30, 0.24]
    # FN
    rgb[~pred &  gt] = [0.20, 0.60, 0.86]
    return rgb


# ── prediction path helpers ───────────────────────────────────────────────────

def find_pred_binary(case_id: str, cfg: Dict, model: str) -> Optional[Path]:
    base = Path(cfg["paths"][f"predictions_{model}"])
    for p in [base / "binary" / f"{case_id}.nii.gz",
              base / f"{case_id}.nii.gz"]:
        if p.exists():
            return p
    return None


def find_pred_probs(case_id: str, cfg: Dict, model: str) -> Optional[Path]:
    base = Path(cfg["paths"][f"predictions_{model}"])
    for p in [base / "probs" / f"{case_id}.npz",
              base / f"{case_id}.npz"]:
        if p.exists():
            return p
    return None


def collect_test_cases(cfg: Dict) -> List[str]:
    inst_dir = Path(cfg["paths"]["instances_ts"])
    return sorted(p.name.replace("_inst.nii.gz", "") for p in inst_dir.glob("*_inst.nii.gz"))


def collect_cases_with_pred(cfg: Dict, model: str, limit: Optional[int] = None) -> List[str]:
    all_cases = collect_test_cases(cfg)
    result = [c for c in all_cases if find_pred_binary(c, cfg, model) is not None]
    if limit is not None:
        result = result[:limit]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 1. Training panels  —  raw EM | GT binary | prediction | error map
# ══════════════════════════════════════════════════════════════════════════════

def fig_training_panels(cfg: Dict, out_root: Path, model: str, n: int = 6) -> None:
    out_dir = out_root / "training_panels" / model
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir  = Path(cfg["paths"]["nnunet_raw"]) / "Dataset001_Mitochondria" / "imagesTs"
    inst_dir = Path(cfg["paths"]["instances_ts"])

    cases = collect_cases_with_pred(cfg, model)
    if not cases:
        print(f"  [training_panels/{model}] no predictions found — skipping")
        return

    # Stratified sample: spread across sorted case list
    step = max(1, len(cases) // n)
    selected = cases[::step][:n]

    print(f"  [training_panels/{model}] {len(selected)} panels → {out_dir}")
    for case_id in selected:
        raw_path  = img_dir / f"{case_id}_0000.nii.gz"
        inst_path = inst_dir / f"{case_id}_inst.nii.gz"
        if not raw_path.exists() or not inst_path.exists():
            print(f"    skip {case_id}: raw or GT missing")
            continue

        raw  = load_nifti(raw_path).astype(np.float32)
        inst = np.rint(load_nifti(inst_path)).astype(np.int32)
        gt   = (inst > 0).astype(np.uint8)

        # Load prediction
        bin_path = find_pred_binary(case_id, cfg, model)
        pred = (load_nifti(bin_path) > 0).astype(np.uint8)
        pred = maybe_T(pred, gt.shape)

        em   = norm01(raw)
        err  = error_map_rgb(pred, gt)

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.04)

        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, ~gt.astype(bool)).sum())
        fn = int(np.logical_and(~pred.astype(bool), gt).sum())
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        fig.suptitle(f"CEM-MitoLab  [{model.upper()}]  —  {case_id}",
                     fontsize=13, fontweight="bold", y=0.97)
        fig.text(0.5, 0.875,
                 f"shape={raw.shape[0]}×{raw.shape[1]}   "
                 f"GT fg={gt.mean()*100:.1f}%   "
                 f"pred fg={pred.mean()*100:.1f}%   "
                 f"Dice={dice:.3f}",
                 ha="center", fontsize=8, color="#555")

        titles = ["Raw EM", "Ground Truth (binary)", "Prediction (binary)", "Error Map"]
        imgs   = [em, gt.astype(np.float32), pred.astype(np.float32), err]
        cmaps  = ["gray", "gray", "gray", None]
        for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
            if cmap is not None:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img, interpolation="nearest")
            ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
            ax.axis("off")

        # Error-map legend
        axes[3].legend(handles=[
            mpatches.Patch(color=(1.0, 1.0, 1.0),       label="TP"),
            mpatches.Patch(color=(0.07, 0.07, 0.07),     label="TN"),
            mpatches.Patch(color=(0.91, 0.30, 0.24),     label="FP"),
            mpatches.Patch(color=(0.20, 0.60, 0.86),     label="FN"),
        ], loc="lower right", fontsize=7, framealpha=0.7, edgecolor="none",
           facecolor="white")

        out_path = out_dir / f"{case_id}_panel.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    {case_id}  Dice={dice:.3f}  → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Evaluation figures — instance labels (not just binary)
# ══════════════════════════════════════════════════════════════════════════════

def fig_eval_instances(cfg: Dict, out_root: Path, n: int = 4,
                        models: Optional[List[str]] = None) -> None:
    out_dir = out_root / "eval_instances"
    out_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ["unet", "nnunet"]

    img_dir  = Path(cfg["paths"]["nnunet_raw"]) / "Dataset001_Mitochondria" / "imagesTs"
    inst_dir = Path(cfg["paths"]["instances_ts"])

    cases_by_model = {m: set(collect_cases_with_pred(cfg, m)) for m in models}
    all_cases = sorted(set().union(*cases_by_model.values()))

    step = max(1, len(all_cases) // n)
    selected = all_cases[::step][:n]
    if not selected:
        print("  [eval_instances] no predictions found — skipping")
        return

    print(f"  [eval_instances] {len(selected)} figures → {out_dir}")
    for case_id in selected:
        raw_path  = img_dir / f"{case_id}_0000.nii.gz"
        inst_path = inst_dir / f"{case_id}_inst.nii.gz"
        if not raw_path.exists() or not inst_path.exists():
            continue

        raw  = load_nifti(raw_path).astype(np.float32)
        inst = np.rint(load_nifti(inst_path)).astype(np.int32)
        gt_bin = (inst > 0).astype(np.uint8)
        em = norm01(raw)

        active_models = [m for m in models if case_id in cases_by_model[m]]
        n_cols = 2 + len(active_models)
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.05)
        model_title = " vs ".join(m.upper().replace("NNUNET", "nnU-Net").replace("UNET", "U-Net")
                                    for m in active_models)
        fig.suptitle(f"Instance Evaluation  [{model_title}]  —  {case_id}",
                     fontsize=13, fontweight="bold", y=0.97)

        ax_idx = 0
        axes[ax_idx].imshow(em, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[ax_idx].set_title("Raw EM", fontsize=10, fontweight="bold", pad=7)
        axes[ax_idx].axis("off")
        ax_idx += 1

        inst_rgb = instance_colourmap(inst, bg_gray=raw)
        axes[ax_idx].imshow(inst_rgb, interpolation="nearest")
        n_inst = int(np.unique(inst[inst > 0]).size)
        axes[ax_idx].set_title(f"GT Instances  ({n_inst} objects)", fontsize=10, fontweight="bold", pad=7)
        axes[ax_idx].axis("off")
        ax_idx += 1

        model_labels = {"unet": "U-Net pred", "nnunet": "nnU-Net pred"}
        for model in active_models:
            label = model_labels[model]
            bin_path = find_pred_binary(case_id, cfg, model)
            pred = (load_nifti(bin_path) > 0).astype(np.uint8)
            pred = maybe_T(pred, gt_bin.shape)
            tp = int(np.logical_and(pred, gt_bin).sum())
            fp = int(np.logical_and(pred, ~gt_bin.astype(bool)).sum())
            fn = int(np.logical_and(~pred.astype(bool), gt_bin).sum())
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            # Label connected components in prediction
            pred_inst, n_pred = scipy_label(pred, structure=np.ones((3, 3), np.uint8))
            pred_rgb = instance_colourmap(pred_inst, bg_gray=raw)
            axes[ax_idx].imshow(pred_rgb, interpolation="nearest")
            axes[ax_idx].set_title(
                f"{label}  ({n_pred} obj, Dice={dice:.3f})",
                fontsize=10, fontweight="bold", pad=7)
            axes[ax_idx].axis("off")
            ax_idx += 1

        model_suffix = "_vs_".join(active_models)
        out_path = out_dir / f"{case_id}_{model_suffix}_instances.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    {case_id}  inst={n_inst}  → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Contour overlap — predicted contour on GT instance colourmap
# ══════════════════════════════════════════════════════════════════════════════

def fig_contour_overlap(cfg: Dict, out_root: Path, n: int = 4,
                         models: Optional[List[str]] = None) -> None:
    out_dir = out_root / "contour_overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ["unet", "nnunet"]

    img_dir  = Path(cfg["paths"]["nnunet_raw"]) / "Dataset001_Mitochondria" / "imagesTs"
    inst_dir = Path(cfg["paths"]["instances_ts"])

    cases_by_model = {m: set(collect_cases_with_pred(cfg, m)) for m in models}
    all_cases = sorted(set().union(*cases_by_model.values()))
    step  = max(1, len(all_cases) // n)
    selected = all_cases[::step][:n]
    if not selected:
        print("  [contour_overlap] no predictions found — skipping")
        return

    print(f"  [contour_overlap] {len(selected)} figures → {out_dir}")
    for case_id in selected:
        raw_path  = img_dir / f"{case_id}_0000.nii.gz"
        inst_path = inst_dir / f"{case_id}_inst.nii.gz"
        if not raw_path.exists() or not inst_path.exists():
            continue

        raw  = load_nifti(raw_path).astype(np.float32)
        inst = np.rint(load_nifti(inst_path)).astype(np.int32)
        gt_bin = (inst > 0)

        models_present = [m for m in models if case_id in cases_by_model[m]]
        if not models_present:
            continue

        n_cols = len(models_present)
        fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 7))
        if n_cols == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.06)
        fig.suptitle(f"Contour Overlap  —  {case_id}", fontsize=13, fontweight="bold", y=0.97)

        for ax, model in zip(axes, models_present):
            bin_path = find_pred_binary(case_id, cfg, model)
            pred = (load_nifti(bin_path) > 0).astype(bool)
            pred = maybe_T(pred, gt_bin.shape)

            inst_rgb = instance_colourmap(inst, bg_gray=raw)
            ax.imshow(inst_rgb, interpolation="nearest")

            # Predicted contour overlay (yellow)
            pred_contour = contour_mask(pred)
            # GT contour overlay (white)
            gt_contour = contour_mask(gt_bin)

            h, w = pred_contour.shape
            contour_layer = np.zeros((h, w, 4), dtype=np.float32)
            contour_layer[pred_contour] = [1.0, 0.85, 0.0, 0.95]   # yellow = pred
            contour_layer[gt_contour]   = [1.0, 1.0,  1.0, 0.75]   # white  = GT

            ax.imshow(contour_layer, interpolation="nearest")

            tp = int((pred & gt_bin).sum())
            fp = int((pred & ~gt_bin).sum())
            fn = int((~pred & gt_bin).sum())
            dice = (2*tp) / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0.0

            ax.set_title(f"{model.upper()} — Dice={dice:.3f}", fontsize=11, fontweight="bold", pad=7)
            ax.axis("off")
            ax.legend(handles=[
                mpatches.Patch(color=(1.0, 0.85, 0.0), label="Pred contour"),
                mpatches.Patch(color=(1.0, 1.0,  1.0), label="GT contour"),
            ], loc="lower right", fontsize=8, framealpha=0.7, edgecolor="none", facecolor="#333")

        out_path = out_dir / f"{case_id}_contour.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    {case_id}  → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Corner cases — high Dice but biological metric failure
# ══════════════════════════════════════════════════════════════════════════════

def fig_corner_cases(cfg: Dict, out_root: Path, results_dir: Path, n: int = 4) -> None:
    out_dir = out_root / "corner_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir  = Path(cfg["paths"]["nnunet_raw"]) / "Dataset001_Mitochondria" / "imagesTs"
    inst_dir = Path(cfg["paths"]["instances_ts"])

    found_any = False
    for model in ["unet", "nnunet"]:
        csv_path = results_dir / f"{model}_per_image.csv"
        if not csv_path.exists():
            print(f"  [corner_cases/{model}] {csv_path} not found — skipping")
            continue

        df = pd.read_csv(csv_path)
        if "dice_fg" not in df.columns:
            print(f"  [corner_cases/{model}] no dice_fg column — skipping")
            continue

        bio_cols = [c for c in ["merge_rate", "split_rate", "connectivity_error",
                                 "obj_count_error", "pq_iou50"]
                    if c in df.columns]
        if not bio_cols:
            print(f"  [corner_cases/{model}] no biological metric columns — skipping")
            continue

        df_clean = df.dropna(subset=["dice_fg"] + bio_cols)
        if df_clean.empty:
            continue

        # Score: high Dice AND high connectivity_error (or low pq_iou50)
        dice_thresh = df_clean["dice_fg"].quantile(0.60)
        if "connectivity_error" in bio_cols:
            bio_fail = df_clean["connectivity_error"] > df_clean["connectivity_error"].quantile(0.70)
        elif "pq_iou50" in bio_cols:
            bio_fail = df_clean["pq_iou50"] < df_clean["pq_iou50"].quantile(0.30)
        else:
            bio_fail = df_clean[bio_cols[0]] > df_clean[bio_cols[0]].quantile(0.70)

        high_dice = df_clean["dice_fg"] >= dice_thresh
        corners = df_clean[high_dice & bio_fail].sort_values("dice_fg", ascending=False)

        if corners.empty:
            print(f"  [corner_cases/{model}] no corner cases found with current thresholds")
            continue

        found_any = True
        selected = corners.head(n)
        print(f"  [corner_cases/{model}] {len(selected)} corner cases → {out_dir}")

        for _, row in selected.iterrows():
            case_id = row["case_id"]
            raw_path  = img_dir / f"{case_id}_0000.nii.gz"
            inst_path = inst_dir / f"{case_id}_inst.nii.gz"
            bin_path  = find_pred_binary(case_id, cfg, model)
            if not raw_path.exists() or not inst_path.exists() or bin_path is None:
                print(f"    skip {case_id}: files missing")
                continue

            raw  = load_nifti(raw_path).astype(np.float32)
            inst = np.rint(load_nifti(inst_path)).astype(np.int32)
            gt   = (inst > 0).astype(np.uint8)
            pred = (load_nifti(bin_path) > 0).astype(np.uint8)
            pred = maybe_T(pred, gt.shape)
            em   = norm01(raw)
            err  = error_map_rgb(pred, gt)

            fig, axes = plt.subplots(1, 4, figsize=(22, 6))
            fig.patch.set_facecolor("white")
            fig.subplots_adjust(left=0.01, right=0.99, top=0.74, bottom=0.02, wspace=0.04)

            dice   = float(row["dice_fg"])
            merges = int(row.get("merge_events", 0))
            splits = int(row.get("split_events", 0))
            conn_e = float(row.get("connectivity_error", 0.0))
            pq     = float(row.get("pq_iou50", float("nan")))

            fig.suptitle(
                f"Corner Case  [{model.upper()}]  —  {case_id}",
                fontsize=13, fontweight="bold", y=0.98)
            fig.text(0.5, 0.885,
                     f"Dice={dice:.3f}   merges={merges}   splits={splits}   "
                     f"conn_error={conn_e:.3f}   PQ@0.5={pq:.3f}",
                     ha="center", fontsize=9, color="#c0392b", fontweight="bold")

            inst_rgb = instance_colourmap(inst, bg_gray=raw)
            pred_inst, n_pred = scipy_label(pred, structure=np.ones((3, 3), np.uint8))
            pred_inst_rgb = instance_colourmap(pred_inst, bg_gray=raw)

            panels = [
                ("Raw EM",              em,           "gray"),
                ("GT Instances",        inst_rgb,     None),
                ("Pred Instances",      pred_inst_rgb, None),
                ("Error Map",           err,           None),
            ]
            for ax, (title, img, cmap) in zip(axes, panels):
                if cmap is not None:
                    ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
                else:
                    ax.imshow(img, interpolation="nearest")
                ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
                ax.axis("off")

            axes[3].legend(handles=[
                mpatches.Patch(color=(1.0, 1.0, 1.0),     label=f"TP"),
                mpatches.Patch(color=(0.07, 0.07, 0.07),  label=f"TN"),
                mpatches.Patch(color=(0.91, 0.30, 0.24),  label=f"FP"),
                mpatches.Patch(color=(0.20, 0.60, 0.86),  label=f"FN"),
            ], loc="lower right", fontsize=7, framealpha=0.7, edgecolor="none", facecolor="white")

            out_path = out_dir / f"{model}_{case_id}_corner.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close()
            print(f"    {case_id}  Dice={dice:.3f}  conn_err={conn_e:.3f}  → {out_path.name}")

    if not found_any:
        print("  [corner_cases] no per-image CSVs available yet — run evaluate.py first")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Scatter plots — Dice vs each biological metric
# ══════════════════════════════════════════════════════════════════════════════

def fig_scatter(cfg: Dict, out_root: Path, results_dir: Path,
                models: Optional[List[str]] = None) -> None:
    out_dir = out_root / "scatter"
    out_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ["unet", "nnunet"]

    MODEL_STYLES = {
        "unet":   dict(color="#2980b9", marker="o", label="U-Net",   zorder=3),
        "nnunet": dict(color="#e74c3c", marker="s", label="nnU-Net", zorder=4),
    }

    bio_metrics = [
        ("iou",                "IoU"),
        ("hd95",               "HD95 (px)"),
        ("boundary_f1_2px",    "Boundary F1 (2px)"),
        ("boundary_f1_5px",    "Boundary F1 (5px)"),
        ("pq_iou50",           "PQ @ IoU 0.5"),
        ("merge_rate",         "Merge rate"),
        ("split_rate",         "Split rate"),
        ("connectivity_error", "Connectivity error"),
        ("matched_iou_mean",   "Mean matched IoU"),
    ]

    frames = {}
    for model in models:
        csv_path = results_dir / f"{model}_per_image.csv"
        if csv_path.exists():
            frames[model] = pd.read_csv(csv_path)

    if not frames:
        print("  [scatter] no per-image CSVs — skipping")
        return

    available_metrics = [(col, label) for col, label in bio_metrics
                         if any(col in df.columns for df in frames.values())]
    if not available_metrics:
        print("  [scatter] no biological metric columns found — skipping")
        return

    print(f"  [scatter] {len(available_metrics)} scatter plots → {out_dir}")

    for col, ylabel in available_metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(f"Dice vs {ylabel}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Dice (FG, t=0.5)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        for model, df in frames.items():
            if "dice_fg" not in df.columns or col not in df.columns:
                continue
            sub = df[["dice_fg", col]].dropna()
            style = MODEL_STYLES[model]
            ax.scatter(sub["dice_fg"], sub[col],
                       color=style["color"], marker=style["marker"],
                       label=style["label"], alpha=0.65, s=35,
                       edgecolors="white", linewidths=0.4,
                       zorder=style["zorder"])

            # Correlation annotation
            if len(sub) >= 3:
                corr = sub["dice_fg"].corr(sub[col])
                ax.annotate(f"r={corr:.2f}", xy=(0.03, 0.97 - list(frames).index(model) * 0.07),
                            xycoords="axes fraction", fontsize=8,
                            color=style["color"], va="top")

        ax.legend(fontsize=9, framealpha=0.5)
        plt.tight_layout()
        out_path = out_dir / f"scatter_dice_vs_{col}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    dice vs {col}  → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Threshold sensitivity curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_threshold_curves(cfg: Dict, out_root: Path, results_dir: Path,
                          models: Optional[List[str]] = None) -> None:
    out_dir = out_root / "threshold_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ["unet", "nnunet"]

    thresholds = cfg["shared"]["threshold_sensitivity"]   # e.g. [0.3,0.4,0.5,0.6,0.7]

    def tag(t: float) -> str:
        return f"t{int(round(t * 100)):02d}"

    metric_groups = [
        ("Dice (FG)",         [f"dice_fg_{tag(t)}"         for t in thresholds]),
        ("IoU",               [f"iou_{tag(t)}"             for t in thresholds]),
        ("Boundary F1 (2px)", [f"boundary_f1_2px_{tag(t)}" for t in thresholds]),
        ("Boundary F1 (5px)", [f"boundary_f1_5px_{tag(t)}" for t in thresholds]),
        ("Precision",         [f"precision_{tag(t)}"       for t in thresholds]),
        ("Recall",            [f"recall_{tag(t)}"          for t in thresholds]),
    ]

    MODEL_STYLES = {
        "unet":   dict(color="#2980b9", marker="o", label="U-Net"),
        "nnunet": dict(color="#e74c3c", marker="s", label="nnU-Net"),
    }

    frames = {}
    for model in models:
        csv_path = results_dir / f"{model}_per_image.csv"
        if csv_path.exists():
            frames[model] = pd.read_csv(csv_path)

    if not frames:
        print("  [threshold_curves] no per-image CSVs — skipping")
        return

    print(f"  [threshold_curves] {len(metric_groups)} curves → {out_dir}")

    # Compact multi-metric figure
    n_metrics = len(metric_groups)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    fig.suptitle("Threshold Sensitivity", fontsize=14, fontweight="bold", y=0.99)

    for ax, (metric_name, col_names) in zip(axes, metric_groups):
        for model, df in frames.items():
            avail_cols  = [c for c in col_names if c in df.columns]
            avail_thrs  = [t for t, c in zip(thresholds, col_names) if c in avail_cols]
            if not avail_cols:
                continue
            means = df[avail_cols].mean()
            stds  = df[avail_cols].std()
            style = MODEL_STYLES[model]
            ax.plot(avail_thrs, means.values,
                    color=style["color"], marker=style["marker"],
                    label=style["label"], linewidth=2, markersize=6)
            ax.fill_between(avail_thrs,
                            means.values - stds.values,
                            means.values + stds.values,
                            color=style["color"], alpha=0.15)

        ax.set_title(metric_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=9)
        ax.set_xticks(thresholds)
        ax.set_xlim(thresholds[0] - 0.03, thresholds[-1] + 0.03)
        ax.legend(fontsize=8, framealpha=0.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_path = out_dir / "threshold_sensitivity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Convergence curves — both models side by side
# ══════════════════════════════════════════════════════════════════════════════

def _parse_unet_log(log_path: Path) -> pd.DataFrame:
    """Parse U-Net training log into a DataFrame."""
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|"
        r"\s+train_loss=([\d.]+)"
        r"\s+train_dice=([\d.]+)"
        r".*?\|"
        r"\s+val_loss=([^\s]+)"
        r"\s+val_dice=([^\s]+)"
    )
    rows = []
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            epoch     = int(m.group(1))
            tr_loss   = float(m.group(2))
            tr_dice   = float(m.group(3))
            val_loss  = None if m.group(4) == "----" else float(m.group(4))
            val_dice  = None if m.group(5) == "----" else float(m.group(5))
            rows.append(dict(epoch=epoch, train_loss=tr_loss, train_dice=tr_dice,
                             val_loss=val_loss, val_dice=val_dice))
    return pd.DataFrame(rows)


def _parse_nnunet_log(log_path: Path) -> pd.DataFrame:
    """
    Parse nnU-Net stdout log.
    Looks for blocks of:
      Epoch N
      train_loss X
      val_loss X
      Pseudo dice [X]
    """
    lines = log_path.read_text().splitlines()
    rows = []
    i = 0
    epoch_re    = re.compile(r"Epoch\s+(\d+)\s*$")
    tloss_re    = re.compile(r"train_loss\s+([-\d.]+)")
    vloss_re    = re.compile(r"val_loss\s+([-\d.]+)")
    pdice_re    = re.compile(r"Pseudo dice\s+\[([\d., ]+)\]")

    while i < len(lines):
        m_ep = epoch_re.search(lines[i])
        if m_ep:
            epoch = int(m_ep.group(1))
            tr_loss = val_loss = pseudo_dice = None
            # scan next ~10 lines for metrics
            for j in range(i + 1, min(i + 12, len(lines))):
                if tloss_re.search(lines[j]):
                    tr_loss = float(tloss_re.search(lines[j]).group(1))
                elif vloss_re.search(lines[j]):
                    val_loss = float(vloss_re.search(lines[j]).group(1))
                elif pdice_re.search(lines[j]):
                    vals = [float(v.strip()) for v in pdice_re.search(lines[j]).group(1).split(",")]
                    pseudo_dice = float(np.mean(vals))
                elif epoch_re.search(lines[j]):
                    break
            rows.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss,
                             pseudo_dice=pseudo_dice))
        i += 1

    return pd.DataFrame(rows)


def fig_convergence(cfg: Dict, out_root: Path, logs_dir: Path) -> None:
    out_dir = out_root / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find log files
    unet_log_candidates = sorted(logs_dir.glob("unet_train_*.out"))
    nnunet_log_candidates = sorted(logs_dir.glob("nnunet_fold0_*.out"))

    unet_df = nnunet_df = None

    if unet_log_candidates:
        dfs = []
        for p in unet_log_candidates:
            try:
                dfs.append(_parse_unet_log(p))
            except Exception as e:
                print(f"    warning: could not parse {p.name}: {e}")
        if dfs:
            unet_df = pd.concat(dfs).sort_values("epoch").drop_duplicates("epoch").reset_index(drop=True)
            print(f"  [convergence] U-Net: {len(unet_df)} epochs parsed from {len(unet_log_candidates)} log(s)")

    if nnunet_log_candidates:
        dfs = []
        for p in nnunet_log_candidates:
            try:
                dfs.append(_parse_nnunet_log(p))
            except Exception as e:
                print(f"    warning: could not parse {p.name}: {e}")
        if dfs:
            nnunet_df = pd.concat(dfs).sort_values("epoch").drop_duplicates("epoch").reset_index(drop=True)
            print(f"  [convergence] nnU-Net: {len(nnunet_df)} epochs parsed from {len(nnunet_log_candidates)} log(s)")

    if unet_df is None and nnunet_df is None:
        print("  [convergence] no training logs found — skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Convergence  —  U-Net vs nnU-Net",
                 fontsize=14, fontweight="bold", y=1.01)

    UNET_C   = "#2980b9"
    NNUNET_C = "#e74c3c"

    # ── top-left: train loss ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_title("Training Loss", fontsize=11, fontweight="bold")
    if unet_df is not None and "train_loss" in unet_df:
        ax.plot(unet_df["epoch"], unet_df["train_loss"],
                color=UNET_C, linewidth=1.5, label="U-Net")
    if nnunet_df is not None and "train_loss" in nnunet_df:
        nn_tr = nnunet_df.dropna(subset=["train_loss"])
        ax.plot(nn_tr["epoch"], nn_tr["train_loss"],
                color=NNUNET_C, linewidth=1.5, label="nnU-Net")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # ── top-right: val loss ──────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.set_title("Validation Loss", fontsize=11, fontweight="bold")
    if unet_df is not None:
        val_u = unet_df.dropna(subset=["val_loss"])
        ax.plot(val_u["epoch"], val_u["val_loss"],
                color=UNET_C, linewidth=1.5, label="U-Net")
    if nnunet_df is not None and "val_loss" in nnunet_df:
        val_n = nnunet_df.dropna(subset=["val_loss"])
        ax.plot(val_n["epoch"], val_n["val_loss"],
                color=NNUNET_C, linewidth=1.5, label="nnU-Net")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # ── bottom-left: train dice / pseudo dice ────────────────────────────────
    ax = axes[1, 0]
    ax.set_title("Training Dice", fontsize=11, fontweight="bold")
    if unet_df is not None and "train_dice" in unet_df:
        ax.plot(unet_df["epoch"], unet_df["train_dice"],
                color=UNET_C, linewidth=1.5, label="U-Net train Dice")
    if nnunet_df is not None and "pseudo_dice" in nnunet_df:
        nn_pd = nnunet_df.dropna(subset=["pseudo_dice"])
        ax.plot(nn_pd["epoch"], nn_pd["pseudo_dice"],
                color=NNUNET_C, linewidth=1.5, label="nnU-Net pseudo Dice")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # ── bottom-right: val dice ───────────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_title("Validation Dice", fontsize=11, fontweight="bold")
    if unet_df is not None:
        val_ud = unet_df.dropna(subset=["val_dice"])
        ax.plot(val_ud["epoch"], val_ud["val_dice"],
                color=UNET_C, linewidth=1.5, label="U-Net val Dice")
    if nnunet_df is not None and "pseudo_dice" in nnunet_df:
        # nnU-Net val Dice is the same pseudo_dice column (reported after val)
        nn_vd = nnunet_df.dropna(subset=["pseudo_dice"])
        ax.plot(nn_vd["epoch"], nn_vd["pseudo_dice"],
                color=NNUNET_C, linewidth=1.5, linestyle="--",
                label="nnU-Net pseudo Dice (val)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_path = out_dir / "convergence.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [convergence] → {out_path}")

    # Also save individual model plots
    for model_name, df, color in [("unet", unet_df, UNET_C), ("nnunet", nnunet_df, NNUNET_C)]:
        if df is None:
            continue
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig2.suptitle(f"Convergence  —  {model_name.upper()}", fontsize=12, fontweight="bold")

        if "train_loss" in df:
            ax1.plot(df["epoch"], df["train_loss"], color=color, linewidth=1.5, label="train")
        if "val_loss" in df.columns:
            vl = df.dropna(subset=["val_loss"])
            ax1.plot(vl["epoch"], vl["val_loss"], color=color, linewidth=1.5,
                     linestyle="--", label="val")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(fontsize=9)
        ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

        dice_col = "train_dice" if model_name == "unet" else "pseudo_dice"
        if dice_col in df.columns:
            dd = df.dropna(subset=[dice_col])
            ax2.plot(dd["epoch"], dd[dice_col], color=color, linewidth=1.5, label=dice_col)
        if model_name == "unet" and "val_dice" in df.columns:
            vd = df.dropna(subset=["val_dice"])
            ax2.plot(vd["epoch"], vd["val_dice"], color=color, linewidth=1.5,
                     linestyle="--", label="val Dice")
        ax2.set_title("Dice"); ax2.set_xlabel("Epoch")
        ax2.set_ylim(bottom=0.0); ax2.legend(fontsize=9)
        ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        out_path2 = out_dir / f"convergence_{model_name}.png"
        plt.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  [convergence] → {out_path2.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

ALL_MODES = [
    "training_panels",
    "eval_instances",
    "contour_overlap",
    "corner_cases",
    "scatter",
    "threshold_curves",
    "convergence",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate all dissertation figures.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    ap.add_argument("--mode",   nargs="+", choices=ALL_MODES + ["all"],
                    default=["all"],
                    help="Which figure sets to produce (default: all).")
    ap.add_argument("--model",  choices=["unet", "nnunet", "both"], default="both",
                    help="Which model(s) to include in all figures (default: both).")
    ap.add_argument("--n",      type=int, default=6,
                    help="Number of sample images for panel figures.")
    ap.add_argument("--out",    type=Path, default=None,
                    help="Output root (default: figures_dir from config).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_cfg(args.config)

    out_root    = args.out or Path(cfg["paths"]["figures_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    logs_dir    = Path(cfg["paths"]["logs_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    modes  = ALL_MODES if "all" in args.mode else args.mode
    active_models = ["unet", "nnunet"] if args.model == "both" else [args.model]

    print(f"Output root : {out_root}")
    print(f"Modes       : {modes}")
    print(f"Models      : {active_models}")
    print()

    if "training_panels" in modes:
        for m in active_models:
            fig_training_panels(cfg, out_root, model=m, n=args.n)

    if "eval_instances" in modes:
        fig_eval_instances(cfg, out_root, n=args.n, models=active_models)

    if "contour_overlap" in modes:
        fig_contour_overlap(cfg, out_root, n=args.n, models=active_models)

    if "corner_cases" in modes:
        fig_corner_cases(cfg, out_root, results_dir, n=args.n)

    if "scatter" in modes:
        fig_scatter(cfg, out_root, results_dir, models=active_models)

    if "threshold_curves" in modes:
        fig_threshold_curves(cfg, out_root, results_dir, models=active_models)

    if "convergence" in modes:
        fig_convergence(cfg, out_root, logs_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
