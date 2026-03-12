"""
Day 1 - Script 4: create_nnunet_format.py
==========================================
Converts raw tif dataset into nnU-Net's required 2D format
AND writes splits_final.json to override nnU-Net's default CV.

CEM-MitoLab specific:
  - Masks are instance labels (0=bg, 1-255=mitochondria instances)
  - Binary masks (0/1) are saved to imagesTr/labelsTr for training
  - Original instance masks are saved separately to instancesTr/
    for use in biological metrics during evaluation (DO NOT DELETE)

nnU-Net 2D structure produced:
  /home/s2292949/dissertation_resplit/nnunet/nnUNet_raw/Dataset001_Mitochondria/
      ├── dataset.json
      ├── imagesTr/            ← binary images, train + val
      │   ├── mito_0001_0000.nii.gz
      │   └── ...
      ├── labelsTr/            ← binary masks (0/1), train + val
      │   ├── mito_0001.nii.gz
      │   └── ...
      ├── instancesTr/         ← ORIGINAL instance masks, train + val
      │   ├── mito_0001_inst.nii.gz
      │   └── ...
      ├── imagesTs/            ← images only, test set
      │   ├── mito_XXXX_0000.nii.gz
      │   └── ...
      ├── instancesTs/         ← ORIGINAL instance masks, test set
      │   ├── mito_XXXX_inst.nii.gz
      │   └── ...
      ├── splits_final.json    ← nnU-Net v2 split file, fold 0 = splits.json train/val
      └── splits_final.pkl     ← legacy fallback copy of the same split

Reads:
  /home/s2292949/dissertation_resplit/data/cem_mitolab
  /home/s2292949/dissertation_resplit/data/splits.json

Outputs:
  /home/s2292949/dissertation_resplit/nnunet/nnUNet_raw/Dataset001_Mitochondria/
  /home/s2292949/dissertation_resplit/data/nnunet_id_map.json
"""

import json
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
from tifffile import imread

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_ROOT      = "/home/s2292949/dissertation_resplit/data/cem_mitolab"
SPLITS_JSON    = "/home/s2292949/dissertation_resplit/data/splits.json"
NNUNET_RAW_DIR = "/home/s2292949/dissertation_resplit/nnunet/nnUNet_raw"   # ← inside nnunet/
OUTPUT_DIR     = "/home/s2292949/dissertation_resplit/data"
DATASET_ID     = "001"
DATASET_NAME   = "Mitochondria"
# ─────────────────────────────────────────────────────────────────────────────

DATASET_FULL_NAME = f"Dataset{DATASET_ID}_{DATASET_NAME}"


def to_nifti(array: np.ndarray, is_label: bool = False) -> nib.Nifti1Image:
    """
    Convert 2D numpy array to NIfTI.
    nnU-Net expects (W, H, 1) layout.
    Binary labels → uint8. Images → float32.
    """
    if is_label:
        data = (array > 0).astype(np.uint8).T[:, :, np.newaxis]
    else:
        data = array.astype(np.float32).T[:, :, np.newaxis]
    return nib.Nifti1Image(data, np.eye(4))


def to_nifti_instance(array: np.ndarray) -> nib.Nifti1Image:
    """Save instance mask as-is (uint8, values 0-255)."""
    data = array.astype(np.uint8).T[:, :, np.newaxis]
    return nib.Nifti1Image(data, np.eye(4))


def create_nnunet_format(data_root, splits_json, nnunet_raw_dir, output_dir):
    data_root      = Path(data_root)
    output_dir     = Path(output_dir)
    nnunet_raw_dir = Path(nnunet_raw_dir)

    with open(splits_json) as f:
        splits = json.load(f)

    # ── Create directory structure ───────────────────────────────────────────
    dataset_dir   = nnunet_raw_dir / DATASET_FULL_NAME
    images_tr     = dataset_dir / "imagesTr"
    labels_tr     = dataset_dir / "labelsTr"
    instances_tr  = dataset_dir / "instancesTr"   # original instance masks
    images_ts     = dataset_dir / "imagesTs"
    instances_ts  = dataset_dir / "instancesTs"   # original instance masks
    for d in [images_tr, labels_tr, instances_tr, images_ts, instances_ts]:
        d.mkdir(parents=True, exist_ok=True)

    id_map         = {}
    case_counter   = 1
    train_case_ids = []
    val_case_ids   = []
    test_case_ids  = []

    # ── Train + Val → imagesTr / labelsTr / instancesTr ─────────────────────
    for split_name in ["train", "val"]:
        for entry in splits[split_name]:
            sample   = entry["sample"]
            filename = entry["filename"]
            case_id  = f"mito_{case_counter:04d}"
            case_counter += 1

            img  = imread(str(data_root / sample / "images" / filename))
            mask = imread(str(data_root / sample / "masks"  / filename))

            # Binary image + binary label for training
            nib.save(to_nifti(img,  is_label=False), images_tr / f"{case_id}_0000.nii.gz")
            nib.save(to_nifti(mask, is_label=True),  labels_tr / f"{case_id}.nii.gz")

            # Original instance mask for evaluation
            nib.save(to_nifti_instance(mask), instances_tr / f"{case_id}_inst.nii.gz")

            id_map[case_id] = {
                "sample":      sample,
                "filename":    filename,
                "split":       split_name,
                "fg_ratio":    entry["fg_ratio"],
                "n_instances": entry["n_instances"],
                "is_empty":    entry["is_empty"],
            }

            if split_name == "train":
                train_case_ids.append(case_id)
            else:
                val_case_ids.append(case_id)

            if case_counter % 500 == 1:
                print(f"  Converted {case_counter - 1} slices...")

    # ── Test → imagesTs / instancesTs (no labelsTr) ──────────────────────────
    for entry in splits["test"]:
        sample   = entry["sample"]
        filename = entry["filename"]
        case_id  = f"mito_{case_counter:04d}"
        case_counter += 1

        img  = imread(str(data_root / sample / "images" / filename))
        mask = imread(str(data_root / sample / "masks"  / filename))

        nib.save(to_nifti(img, is_label=False),   images_ts    / f"{case_id}_0000.nii.gz")
        nib.save(to_nifti_instance(mask),          instances_ts / f"{case_id}_inst.nii.gz")

        id_map[case_id] = {
            "sample":      sample,
            "filename":    filename,
            "split":       "test",
            "fg_ratio":    entry["fg_ratio"],
            "n_instances": entry["n_instances"],
            "is_empty":    entry["is_empty"],
        }
        test_case_ids.append(case_id)

    print(f"\nTotal converted   : {case_counter - 1} slices")
    print(f"  imagesTr train  : {len(train_case_ids)}")
    print(f"  imagesTr val    : {len(val_case_ids)}")
    print(f"  imagesTs test   : {len(test_case_ids)}")

    # ── Write dataset.json ───────────────────────────────────────────────────
    dataset_json = {
        "channel_names": {"0": "EM"},
        "labels": {
            "background":   0,
            "mitochondria": 1,
        },
        "numTraining": len(train_case_ids) + len(val_case_ids),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\ndataset.json saved     → {dataset_dir / 'dataset.json'}")

    # ── Write splits_final.{json,pkl} ────────────────────────────────────────
    # nnU-Net v2 expects a list of fold dicts in splits_final.json.
    # Fold 0 = our exact train/val from splits.json.
    # Folds 1–4 are copies — they will NEVER be trained.
    splits = [
        {"train": list(train_case_ids), "val": list(val_case_ids)}
        for _ in range(5)
    ]

    json_path = dataset_dir / "splits_final.json"
    with open(json_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"splits_final.json saved → {json_path}")

    pkl_path = dataset_dir / "splits_final.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"splits_final.pkl saved  → {pkl_path}")
    print(f"  Fold 0 train : {len(train_case_ids)} cases")
    print(f"  Fold 0 val   : {len(val_case_ids)} cases")

    # ── Save nnunet_id_map.json ──────────────────────────────────────────────
    id_map_path = output_dir / "nnunet_id_map.json"
    with open(id_map_path, "w") as f:
        json.dump(id_map, f, indent=2)
    print(f"ID map saved           → {id_map_path}")
    print("  (maps case IDs like mito_0001 back to original sample/filename)")

    # ── Verification ────────────────────────────────────────────────────────
    tr_images    = list(images_tr.glob("*_0000.nii.gz"))
    tr_labels    = list(labels_tr.glob("*.nii.gz"))
    tr_instances = list(instances_tr.glob("*_inst.nii.gz"))
    ts_images    = list(images_ts.glob("*_0000.nii.gz"))
    ts_instances = list(instances_ts.glob("*_inst.nii.gz"))

    print(f"\n── Verification ──────────────────────────────────────────────")
    print(f"imagesTr    : {len(tr_images)} images")
    print(f"labelsTr    : {len(tr_labels)} labels")
    print(f"instancesTr : {len(tr_instances)} instance masks")
    print(f"imagesTs    : {len(ts_images)} images")
    print(f"instancesTs : {len(ts_instances)} instance masks")

    assert len(tr_images) == len(tr_labels), \
        f"MISMATCH: {len(tr_images)} images vs {len(tr_labels)} labels in imagesTr"
    assert len(tr_images) == len(tr_instances), \
        f"MISMATCH: {len(tr_images)} images vs {len(tr_instances)} instance masks in instancesTr"
    assert len(ts_images) == len(ts_instances), \
        f"MISMATCH: {len(ts_images)} images vs {len(ts_instances)} instance masks in instancesTs"
    print("✓ All counts match")


if __name__ == "__main__":
    create_nnunet_format(DATA_ROOT, SPLITS_JSON, NNUNET_RAW_DIR, OUTPUT_DIR)
