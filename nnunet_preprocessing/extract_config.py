"""
Write config.json after nnU-Net preprocessing.

Run this after `nnUNetv2_plan_and_preprocess`. It copies splits_final.json
into nnUNet_preprocessed so nnU-Net uses the dissertation split (fold 0),
then reads nnUNetPlans.json and writes config.json with the patch size,
batch size, and spacing both models share.

Reads:  nnunet/nnUNet_preprocessed/Dataset001_Mitochondria/nnUNetPlans.json
Output: config.json
"""

import json
import shutil
from pathlib import Path

REPO_ROOT               = Path(__file__).resolve().parents[1]
NNUNET_PREPROCESSED     = REPO_ROOT / "nnunet" / "nnUNet_preprocessed"
NNUNET_RAW              = REPO_ROOT / "nnunet" / "nnUNet_raw"
DATASET_NAME            = "Dataset001_Mitochondria"
CONFIG_OUT              = REPO_ROOT / "config.json"
SPLITS_JSON             = REPO_ROOT / "data" / "splits.json"

MAX_EPOCHS              = 1000
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_DELTA    = 0.001
AUGMENTATION_UNET       = False
RANDOM_SEED             = 42

def extract_config():
    preprocessed_dir = Path(NNUNET_PREPROCESSED) / DATASET_NAME
    raw_dir          = Path(NNUNET_RAW) / DATASET_NAME
    plans_path       = preprocessed_dir / "nnUNetPlans.json"
    with open(SPLITS_JSON) as f:
        active_splits = json.load(f)

    if not plans_path.exists():
        print(f"ERROR: nnUNetPlans.json not found at {plans_path}")
        print("Run this first:")
        print("  nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")
        return

    src_json = raw_dir / "splits_final.json"
    dst_json = preprocessed_dir / "splits_final.json"

    if not src_json.exists():
        print(f"ERROR: splits_final.json not found at {src_json}")
        print("Re-run data_preparation/create_nnunet_format.py")
        return

    shutil.copy2(src_json, dst_json)
    with open(src_json) as f:
        split_defs = json.load(f)
    fold_0 = split_defs[0]
    print(f"✓ splits_final.json copied → {dst_json}")
    print(f"  Fold 0 train : {len(fold_0['train'])} cases")
    print(f"  Fold 0 val   : {len(fold_0['val'])} cases")

    src_pkl = raw_dir / "splits_final.pkl"
    dst_pkl = preprocessed_dir / "splits_final.pkl"
    if src_pkl.exists():
        shutil.copy2(src_pkl, dst_pkl)
        print(f"✓ splits_final.pkl copied  → {dst_pkl}")
    else:
        print(f"WARNING: legacy splits_final.pkl not found at {src_pkl}")

    with open(plans_path) as f:
        plans = json.load(f)

    print(f"\n✓ nnUNetPlans.json loaded")

    configs = plans.get("configurations", {})
    if "2d" not in configs:
        print("ERROR: No 2d configuration found in nnUNetPlans.json")
        print("Available configurations:", list(configs.keys()))
        return

    cfg_2d     = configs["2d"]
    patch_size = cfg_2d.get("patch_size", [224, 224])
    batch_size = cfg_2d.get("batch_size", 12)

    fingerprint_path = preprocessed_dir / "dataset_fingerprint.json"
    spacing = [1.0, 1.0]
    if fingerprint_path.exists():
        with open(fingerprint_path) as f:
            fingerprint = json.load(f)
        raw_spacings = fingerprint.get("spacings", [[1.0, 1.0]])
        if raw_spacings:
            spacing = raw_spacings[0]
        print(f"✓ Pixel spacing from fingerprint: {spacing}")
    else:
        print("WARNING: dataset_fingerprint.json not found, using spacing [1.0, 1.0]")

    def rel(path: Path) -> str:
        return path.relative_to(REPO_ROOT).as_posix()

    config = {
        "dataset": {
            "name":            "CEM-MitoLab",
            "empiar_id":       "EMPIAR-11037",
            "n_classes":       2,
            "class_names":     ["background", "mitochondria"],
            "image_size":      [224, 224],
            "pixel_spacing":   spacing,
            "total_slices":    21871,
            "train_slices":    len(active_splits["train"]),
            "val_slices":      len(active_splits["val"]),
            "test_slices":     len(active_splits["test"]),
        },
        "nnunet": {
            "patch_size":      patch_size,
            "batch_size":      batch_size,
            "config":          "2d",
            "fold":            0,
            "max_epochs":      MAX_EPOCHS,
            "save_flag":       "--npz",
        },
        "unet": {
            "patch_size":      patch_size,
            "batch_size":      batch_size,
            "max_epochs":      MAX_EPOCHS,
            "augmentation":    AUGMENTATION_UNET,
            "early_stopping": {
                "patience":    EARLY_STOPPING_PATIENCE,
                "min_delta":   EARLY_STOPPING_DELTA,
            },
        },
        "shared": {
            "random_seed":              RANDOM_SEED,
            "early_stopping_patience":  EARLY_STOPPING_PATIENCE,
            "early_stopping_delta":     EARLY_STOPPING_DELTA,
            "max_epochs":               MAX_EPOCHS,
            "prediction_threshold":     0.5,
            "threshold_sensitivity":    [0.3, 0.4, 0.5, 0.6, 0.7],
            "min_object_size_px":       10,
            "hd95_units":               "pixels",
            "boundary_f1_tolerances":   [2, 5],
            "object_iou_thresholds":    [0.3, 0.5, 0.7, 0.9],
        },
        "paths": {
            "data_root":          rel(REPO_ROOT / "data" / "cem_mitolab"),
            "splits_json":        rel(REPO_ROOT / "data" / "splits.json"),
            "nnunet_id_map":      rel(REPO_ROOT / "data" / "nnunet_id_map.json"),
            "dataset_stats":      rel(REPO_ROOT / "data" / "dataset_stats.csv"),
            "nnunet_raw":         rel(REPO_ROOT / "nnunet" / "nnUNet_raw"),
            "nnunet_preproc":     rel(REPO_ROOT / "nnunet" / "nnUNet_preprocessed"),
            "nnunet_results":     rel(REPO_ROOT / "nnunet" / "nnUNet_results"),
            "instances_tr":       rel(REPO_ROOT / "nnunet" / "nnUNet_raw" / DATASET_NAME / "instancesTr"),
            "instances_ts":       rel(REPO_ROOT / "nnunet" / "nnUNet_raw" / DATASET_NAME / "instancesTs"),
            "predictions_unet":   rel(REPO_ROOT / "predictions" / "unet"),
            "predictions_nnunet": rel(REPO_ROOT / "predictions" / "nnunet"),
            "results_dir":        rel(REPO_ROOT / "evaluation" / "results"),
            "figures_dir":        rel(REPO_ROOT / "figures"),
            "logs_dir":           rel(REPO_ROOT / "logs"),
            "unet_checkpoint":    rel(REPO_ROOT / "models" / "unet" / "checkpoints"),
            "unet_training_log":  rel(REPO_ROOT / "models" / "unet" / "checkpoints" / "training_log.csv"),
        }
    }

    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ config.json saved → {CONFIG_OUT}")
    print(f"\n── Extracted from nnUNetPlans.json ──────────────────────────")
    print(f"  patch_size : {patch_size}")
    print(f"  batch_size : {batch_size}")
    print(f"  spacing    : {spacing}")
    print(f"\n── U-Net configuration ──────────────────────────────────────")
    print(f"  patch_size    : {patch_size}  (matched to nnU-Net)")
    print(f"  batch_size    : {batch_size}  (matched to nnU-Net)")
    print(f"  augmentation  : {AUGMENTATION_UNET}  (supervisor: keep simple)")
    print(f"  max_epochs    : {MAX_EPOCHS}")
    print(f"  early_stopping: patience={EARLY_STOPPING_PATIENCE}, delta={EARLY_STOPPING_DELTA}")
    print(f"\n── Shared evaluation settings ───────────────────────────────")
    print(f"  HD95 units           : pixels  (supervisor confirmed)")
    print(f"  min_object_size_px   : 10")
    print(f"  boundary_tolerances  : [2px, 5px]")
    print(f"  thresholds           : [0.3, 0.4, 0.5, 0.6, 0.7]")
    print(f"  object IoU thresholds: [0.3, 0.5, 0.7, 0.9]")

if __name__ == "__main__":
    extract_config()
