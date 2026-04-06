#!/usr/bin/env python3
"""
Per-image evaluation for U-Net and nnU-Net predictions on the test set.

Inputs:
  - Binary predictions (.nii.gz)
  - Probability maps (.npz, key='probabilities')
  - Instance ground truth masks from instancesTs/ (0..255, non-binary)

Outputs:
  - evaluation/results/{model}_per_image.csv
  - evaluation/results/{model}_aggregate.csv
  - evaluation/results/{model}_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = REPO_ROOT / "config.json"

def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)

def load_config(config_path: Path) -> Dict:
    cfg = json.loads(config_path.read_text())
    base_dir = config_path.resolve().parent
    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                cfg["paths"][key] = str(base_dir / path)
    return cfg

def threshold_tag(threshold: float) -> str:
    return f"t{int(round(threshold * 100)):02d}"

def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)

def load_nifti_array(path: Path) -> np.ndarray:
    array = nib.load(str(path)).get_fdata()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    return np.asarray(array)

def maybe_transpose_to_match(array: np.ndarray, target_shape: Tuple[int, int], name: str) -> np.ndarray:
    if tuple(array.shape) == tuple(target_shape):
        return array
    if tuple(array.T.shape) == tuple(target_shape):
        return array.T
    raise ValueError(f"{name} shape {array.shape} cannot be aligned to target shape {target_shape}.")

def align_prob_to_binary(prob_fg: np.ndarray, pred_binary: np.ndarray, threshold: float) -> np.ndarray:
    """Fix transposition for square images.

    maybe_transpose_to_match compares shapes, but for a square image (H == W) the
    transposed shape is identical, so a row/column swap goes undetected.  This
    function uses the already-correct binary prediction (loaded from nifti) as a
    reference: if thresholding the probability map agrees better with pred_binary
    after transposing, the transposed map is returned.

    For non-square images the check is skipped because maybe_transpose_to_match
    already handles them correctly via shape comparison.
    """
    if prob_fg.shape[0] != prob_fg.shape[1]:
        return prob_fg
    pred_ref = pred_binary.astype(bool)
    thr_orig = (prob_fg >= threshold)
    thr_T    = (prob_fg.T >= threshold)
    agree      = np.equal(thr_orig, pred_ref).sum()
    agree_T    = np.equal(thr_T,    pred_ref).sum()
    if agree_T > agree:
        return np.ascontiguousarray(prob_fg.T)
    return prob_fg

def extract_probabilities(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "probabilities" in data.files:
        arr = data["probabilities"]
    elif data.files:
        arr = data[data.files[0]]
    else:
        raise ValueError(f"No arrays found in {npz_path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:

        if arr.shape[0] >= 2:
            fg = arr[1, 0]
        elif arr.shape[1] >= 2:
            fg = arr[0, 1]
        else:
            fg = arr[0, 0]
    elif arr.ndim == 3:
        if arr.shape[0] >= 2:
            fg = arr[1]
        elif arr.shape[-1] >= 2:
            fg = arr[..., 1]
        else:
            fg = arr[0]
    elif arr.ndim == 2:
        fg = arr
    else:
        raise ValueError(f"Unsupported probability array shape {arr.shape} in {npz_path}")

    return np.clip(fg.astype(np.float32), 0.0, 1.0)

def confusion_counts(pred: np.ndarray, target: np.ndarray) -> Tuple[int, int, int, int]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    return tp, tn, fp, fn

def compute_hd95(pred: np.ndarray, target: np.ndarray, spacing: Tuple[float, float] = (1.0, 1.0)) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")

    structure = np.ones((3, 3), dtype=bool)
    pred_boundary = np.logical_xor(pred, binary_erosion(pred, structure=structure, border_value=0))
    target_boundary = np.logical_xor(target, binary_erosion(target, structure=structure, border_value=0))

    if pred_boundary.sum() == 0:
        pred_boundary = pred
    if target_boundary.sum() == 0:
        target_boundary = target

    dt_pred = distance_transform_edt(~pred_boundary, sampling=spacing)
    dt_target = distance_transform_edt(~target_boundary, sampling=spacing)

    d_pred_to_target = dt_target[pred_boundary]
    d_target_to_pred = dt_pred[target_boundary]
    if d_pred_to_target.size == 0 or d_target_to_pred.size == 0:
        return float("inf")
    all_dist = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_dist, 95))

def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance_px: int) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    structure = np.ones((3, 3), dtype=bool)
    pred_boundary = np.logical_xor(pred, binary_erosion(pred, structure=structure, border_value=0))
    target_boundary = np.logical_xor(target, binary_erosion(target, structure=structure, border_value=0))

    if pred_boundary.sum() == 0:
        pred_boundary = pred
    if target_boundary.sum() == 0:
        target_boundary = target

    dt_to_target = distance_transform_edt(~target_boundary)
    dt_to_pred = distance_transform_edt(~pred_boundary)

    pred_hits = dt_to_target[pred_boundary] <= tolerance_px
    target_hits = dt_to_pred[target_boundary] <= tolerance_px

    precision = pred_hits.mean() if pred_hits.size > 0 else 0.0
    recall = target_hits.mean() if target_hits.size > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))

def voxel_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return {
            "dice_fg": float("nan"),
            "dice_bg": 1.0,
            "dice_mean": float("nan"),
            "iou": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }

    tp, tn, fp, fn = confusion_counts(pred, target)

    dice_fg = safe_div(2 * tp, 2 * tp + fp + fn)
    dice_bg = safe_div(2 * tn, 2 * tn + fp + fn)
    dice_mean = 0.5 * (dice_fg + dice_bg)
    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "dice_fg": dice_fg,
        "dice_bg": dice_bg,
        "dice_mean": dice_mean,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }

def build_aggregate(df: pd.DataFrame) -> pd.DataFrame:

    numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({
        "mean": numeric_df.mean(numeric_only=True),
        "std": numeric_df.std(numeric_only=True),
        "median": numeric_df.median(numeric_only=True),
        "q25": numeric_df.quantile(0.25, numeric_only=True),
        "q75": numeric_df.quantile(0.75, numeric_only=True),
        "min": numeric_df.min(numeric_only=True),
        "max": numeric_df.max(numeric_only=True),
    })

def expected_calibration_error(prob_fg: np.ndarray, target: np.ndarray, n_bins: int = 15) -> float:
    target_bool = target.astype(bool).ravel()
    prob_fg = np.clip(prob_fg.astype(np.float32).ravel(), 0.0, 1.0)

    pred_bool = prob_fg >= 0.5
    confidence = np.where(pred_bool, prob_fg, 1.0 - prob_fg)
    correct = (pred_bool == target_bool).astype(np.float32)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        if i == 0:
            in_bin = (confidence >= left) & (confidence <= right)
        else:
            in_bin = (confidence > left) & (confidence <= right)
        if not np.any(in_bin):
            continue
        bin_acc = float(correct[in_bin].mean())
        bin_conf = float(confidence[in_bin].mean())
        ece += float(in_bin.mean()) * abs(bin_acc - bin_conf)
    return float(ece)

def compute_object_metrics(
    pred_binary: np.ndarray,
    gt_instance: np.ndarray,
    iou_thresholds: Sequence[float],
) -> Dict[str, float]:
    pred_binary = pred_binary.astype(bool)
    gt_instance = gt_instance.astype(np.int32)

    structure = np.ones((3, 3), dtype=np.uint8)
    pred_labeled, n_pred = label(pred_binary, structure=structure)
    gt_ids = np.unique(gt_instance)
    gt_ids = gt_ids[gt_ids > 0]
    n_gt = int(gt_ids.size)

    if n_gt == 0 and n_pred == 0:
        out: Dict[str, float] = {
            "gt_objects": 0,
            "pred_objects": 0,
            "obj_count_error": 0,
            "obj_count_error_signed": 0,
            "merge_events": 0,
            "split_events": 0,
            "merge_rate": 0.0,
            "split_rate": 0.0,
            "connectivity_error": 0.0,
            "pq_iou50": 1.0,
            "matched_iou_mean": 0.0,
        }
        for thr in iou_thresholds:
            tag = threshold_tag(thr)
            out[f"obj_precision_{tag}"] = 1.0
            out[f"obj_recall_{tag}"] = 1.0
        return out

    if n_gt == 0 or n_pred == 0:
        denom_gt = max(n_gt, 1)
        out = {
            "gt_objects": n_gt,
            "pred_objects": int(n_pred),
            "obj_count_error": abs(int(n_pred) - n_gt),
            "obj_count_error_signed": int(n_pred) - n_gt,
            "merge_events": 0,
            "split_events": 0,
            "merge_rate": 0.0,
            "split_rate": 0.0,
            "connectivity_error": 0.0,
            "pq_iou50": 0.0,
            "matched_iou_mean": 0.0,
        }
        for thr in iou_thresholds:
            tag = threshold_tag(thr)
            tp = 0
            fp = int(n_pred)
            fn = n_gt
            out[f"obj_precision_{tag}"] = safe_div(tp, tp + fp)
            out[f"obj_recall_{tag}"] = safe_div(tp, tp + fn)
        return out

    max_gt_label = int(gt_ids.max())
    gt_map = np.zeros(max_gt_label + 1, dtype=np.int32)
    gt_map[gt_ids] = np.arange(1, n_gt + 1, dtype=np.int32)

    gt_flat = gt_instance.ravel()
    pred_flat = pred_labeled.ravel()
    gt_mapped = gt_map[gt_flat]

    overlap_mask = (gt_mapped > 0) & (pred_flat > 0)
    pair_idx = (gt_mapped[overlap_mask] - 1) * n_pred + (pred_flat[overlap_mask] - 1)
    intersections = np.bincount(pair_idx, minlength=n_gt * n_pred).reshape(n_gt, n_pred)

    gt_areas = np.bincount(gt_mapped[gt_mapped > 0] - 1, minlength=n_gt)
    pred_areas = np.bincount(pred_flat[pred_flat > 0] - 1, minlength=n_pred)

    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou_matrix = np.zeros_like(intersections, dtype=np.float32)
    valid_union = unions > 0
    iou_matrix[valid_union] = intersections[valid_union] / unions[valid_union]

    row_idx, col_idx = linear_sum_assignment(1.0 - iou_matrix)
    matched_iou = iou_matrix[row_idx, col_idx]

    real_matches = matched_iou[matched_iou > 0]
    matched_iou_mean = float(real_matches.mean()) if real_matches.size > 0 else 0.0

    overlap_gt_per_pred = np.count_nonzero(intersections > 0, axis=0)
    overlap_pred_per_gt = np.count_nonzero(intersections > 0, axis=1)
    merge_events = int(np.clip(overlap_gt_per_pred - 1, 0, None).sum())
    split_events = int(np.clip(overlap_pred_per_gt - 1, 0, None).sum())

    denom_gt = max(n_gt, 1)
    merge_rate = merge_events / denom_gt
    split_rate = split_events / denom_gt
    connectivity_error = (merge_events + split_events) / denom_gt

    out = {
        "gt_objects": n_gt,
        "pred_objects": int(n_pred),
        "obj_count_error": abs(int(n_pred) - n_gt),
        "obj_count_error_signed": int(n_pred) - n_gt,
        "merge_events": merge_events,
        "split_events": split_events,
        "merge_rate": float(merge_rate),
        "split_rate": float(split_rate),
        "connectivity_error": float(connectivity_error),
        "matched_iou_mean": matched_iou_mean,
    }

    for thr in iou_thresholds:
        tp = int((matched_iou >= thr).sum())
        fp = int(n_pred - tp)
        fn = int(n_gt - tp)
        tag = threshold_tag(thr)
        out[f"obj_precision_{tag}"] = safe_div(tp, tp + fp)
        out[f"obj_recall_{tag}"] = safe_div(tp, tp + fn)

    tp_pq = int((matched_iou > 0.5).sum())
    fp_pq = int(n_pred - tp_pq)
    fn_pq = int(n_gt - tp_pq)
    denom_pq = tp_pq + 0.5 * fp_pq + 0.5 * fn_pq
    if denom_pq == 0:
        out["pq_iou50"] = 1.0
    else:
        out["pq_iou50"] = float(matched_iou[matched_iou > 0.5].sum() / denom_pq)

    return out

def resolve_prediction_dirs(
    model: str,
    cfg: Dict,
    binary_dir_override: Optional[Path],
    probs_dir_override: Optional[Path],
) -> Tuple[Path, Path]:
    base = Path(cfg["paths"][f"predictions_{model}"])
    if binary_dir_override is None:
        binary_dir = base / "binary" if (base / "binary").exists() else base
    else:
        binary_dir = binary_dir_override
    if probs_dir_override is None:
        probs_dir = base / "probs" if (base / "probs").exists() else base
    else:
        probs_dir = probs_dir_override
    return binary_dir, probs_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified per-image evaluation for U-Net and nnU-Net.")
    parser.add_argument("--model", required=True, choices=["unet", "nnunet"], help="Model name.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_DEFAULT,
        help="Path to config.json.",
    )
    parser.add_argument("--binary-dir", type=Path, default=None, help="Override binary prediction directory.")
    parser.add_argument("--probs-dir", type=Path, default=None, help="Override probability directory.")
    parser.add_argument("--instances-dir", type=Path, default=None, help="Override instancesTs directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output CSV/JSON files.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Foreground thresholds for voxel-level metrics.",
    )
    parser.add_argument(
        "--obj-iou-thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
        help="IoU thresholds for object precision/recall.",
    )
    parser.add_argument("--ece-bins", type=int, default=15, help="Number of bins for ECE.")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke testing.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any prediction files are missing; default is to skip missing cases.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    binary_dir, probs_dir = resolve_prediction_dirs(
        model=args.model,
        cfg=cfg,
        binary_dir_override=args.binary_dir,
        probs_dir_override=args.probs_dir,
    )
    instances_dir = args.instances_dir or Path(cfg["paths"]["instances_ts"])
    output_dir = args.output_dir or Path(cfg["paths"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    default_threshold = float(cfg["shared"].get("prediction_threshold", 0.5))
    thresholds = sorted(set(args.thresholds))
    if default_threshold not in thresholds:
        thresholds.append(default_threshold)
        thresholds = sorted(thresholds)

    instance_files = sorted(instances_dir.glob("*_inst.nii.gz"))
    if args.limit is not None:
        instance_files = instance_files[: args.limit]
    if not instance_files:
        raise RuntimeError(f"No instance masks found in {instances_dir}")

    print("=" * 72)
    print("Evaluation")
    print(f"  model         : {args.model}")
    print(f"  binary dir    : {binary_dir}")
    print(f"  probs dir     : {probs_dir}")
    print(f"  instances dir : {instances_dir}")
    print(f"  cases         : {len(instance_files)}")
    print(f"  thresholds    : {thresholds}")
    print("=" * 72)

    rows: List[Dict[str, float]] = []
    missing_binary: List[str] = []
    missing_probs: List[str] = []
    failed_cases: List[str] = []

    for inst_path in _progress(instance_files, desc="Evaluating", leave=False):
        case_id = inst_path.name.replace("_inst.nii.gz", "")
        pred_bin_path = binary_dir / f"{case_id}.nii.gz"
        pred_prob_path = probs_dir / f"{case_id}.npz"

        if not pred_bin_path.exists():
            missing_binary.append(case_id)
            if args.strict:
                raise FileNotFoundError(f"Missing binary prediction: {pred_bin_path}")
            continue

        try:
            gt_instance = np.rint(load_nifti_array(inst_path)).astype(np.int32)
            gt_binary = (gt_instance > 0).astype(np.uint8)

            pred_binary = (load_nifti_array(pred_bin_path) > 0).astype(np.uint8)
            pred_binary = maybe_transpose_to_match(pred_binary, gt_binary.shape, "binary prediction").astype(np.uint8)

            prob_fg: Optional[np.ndarray] = None
            if pred_prob_path.exists():
                prob_fg = extract_probabilities(pred_prob_path)
                prob_fg = maybe_transpose_to_match(prob_fg, gt_binary.shape, "probabilities").astype(np.float32)
                prob_fg = align_prob_to_binary(prob_fg, pred_binary, default_threshold)
            else:
                missing_probs.append(case_id)

            row: Dict[str, float] = {
                "case_id": case_id,
                "model": args.model,
                "gt_fg_pixels": int(gt_binary.sum()),
                "pred_fg_pixels": int(pred_binary.sum()),
            }

            for thr in thresholds:
                tag = threshold_tag(thr)
                if prob_fg is not None:
                    pred_thr = (prob_fg >= thr).astype(np.uint8)
                elif math.isclose(thr, default_threshold, rel_tol=0.0, abs_tol=1e-8):
                    pred_thr = pred_binary
                else:
                    pred_thr = None

                if pred_thr is None:
                    row[f"dice_fg_{tag}"] = np.nan
                    row[f"dice_bg_{tag}"] = np.nan
                    row[f"dice_mean_{tag}"] = np.nan
                    row[f"iou_{tag}"] = np.nan
                    row[f"precision_{tag}"] = np.nan
                    row[f"recall_{tag}"] = np.nan
                    row[f"hd95_{tag}"] = np.nan
                    row[f"boundary_f1_2px_{tag}"] = np.nan
                    row[f"boundary_f1_5px_{tag}"] = np.nan
                    continue

                vm = voxel_metrics(pred_thr, gt_binary)
                row[f"dice_fg_{tag}"] = vm["dice_fg"]
                row[f"dice_bg_{tag}"] = vm["dice_bg"]
                row[f"dice_mean_{tag}"] = vm["dice_mean"]
                row[f"iou_{tag}"] = vm["iou"]
                row[f"precision_{tag}"] = vm["precision"]
                row[f"recall_{tag}"] = vm["recall"]
                row[f"hd95_{tag}"] = compute_hd95(pred_thr, gt_binary)
                row[f"boundary_f1_2px_{tag}"] = boundary_f1(pred_thr, gt_binary, tolerance_px=2)
                row[f"boundary_f1_5px_{tag}"] = boundary_f1(pred_thr, gt_binary, tolerance_px=5)

                if math.isclose(thr, default_threshold, rel_tol=0.0, abs_tol=1e-8):
                    row["dice_fg"] = vm["dice_fg"]
                    row["dice_bg"] = vm["dice_bg"]
                    row["dice_mean"] = vm["dice_mean"]
                    row["iou"] = vm["iou"]
                    row["precision"] = vm["precision"]
                    row["recall"] = vm["recall"]
                    row["hd95"] = row[f"hd95_{tag}"]
                    row["boundary_f1_2px"] = row[f"boundary_f1_2px_{tag}"]
                    row["boundary_f1_5px"] = row[f"boundary_f1_5px_{tag}"]

            row["ece"] = expected_calibration_error(prob_fg, gt_binary, n_bins=args.ece_bins) if prob_fg is not None else np.nan

            if prob_fg is not None:
                pred_for_objects = (prob_fg >= default_threshold).astype(np.uint8)
            else:
                pred_for_objects = pred_binary
            row.update(compute_object_metrics(pred_for_objects, gt_instance,
                                               iou_thresholds=args.obj_iou_thresholds))

            rows.append(row)
        except Exception:
            failed_cases.append(case_id)
            if args.strict:
                raise

    if not rows:
        raise RuntimeError("No cases were evaluated successfully.")

    df = pd.DataFrame(rows)
    per_image_path = output_dir / f"{args.model}_per_image.csv"
    df.to_csv(per_image_path, index=False)

    aggregate = build_aggregate(df)
    aggregate_path = output_dir / f"{args.model}_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=True)

    gt_positive_df = df[df["gt_fg_pixels"] > 0].copy()
    gt_empty_df = df[df["gt_fg_pixels"] == 0].copy()

    aggregate_gt_positive_path = output_dir / f"{args.model}_aggregate_gt_positive.csv"
    build_aggregate(gt_positive_df).to_csv(aggregate_gt_positive_path, index=True)

    aggregate_gt_empty_path = output_dir / f"{args.model}_aggregate_gt_empty.csv"
    build_aggregate(gt_empty_df).to_csv(aggregate_gt_empty_path, index=True)

    summary = {
        "model": args.model,
        "cases_total": len(instance_files),
        "cases_evaluated": len(df),
        "cases_missing_binary": len(missing_binary),
        "cases_missing_probs": len(missing_probs),
        "cases_failed": len(failed_cases),
        "binary_dir": str(binary_dir),
        "probs_dir": str(probs_dir),
        "instances_dir": str(instances_dir),
        "thresholds": thresholds,
        "default_threshold": default_threshold,
        "per_image_csv": str(per_image_path),
        "aggregate_csv": str(aggregate_path),
        "aggregate_gt_positive_csv": str(aggregate_gt_positive_path),
        "aggregate_gt_empty_csv": str(aggregate_gt_empty_path),
        "cases_gt_positive": int(len(gt_positive_df)),
        "cases_gt_empty": int(len(gt_empty_df)),
        "missing_binary_case_ids": missing_binary[:50],
        "missing_probs_case_ids": missing_probs[:50],
        "failed_case_ids": failed_cases[:50],
    }
    summary_path = output_dir / f"{args.model}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nEvaluation complete.")
    print(f"  per-image : {per_image_path}")
    print(f"  aggregate : {aggregate_path}")
    print(f"  gt>0 agg  : {aggregate_gt_positive_path}")
    print(f"  gt=0 agg  : {aggregate_gt_empty_path}")
    print(f"  summary   : {summary_path}")
    print(f"  evaluated : {len(df)} / {len(instance_files)}")
    print(f"  missing binary : {len(missing_binary)}")
    print(f"  missing probs  : {len(missing_probs)}")
    print(f"  failed cases   : {len(failed_cases)}")

if __name__ == "__main__":
    main()
