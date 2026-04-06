#!/bin/bash
#SBATCH --job-name=nnunet_fold0
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/nnunet_fold0_%j.out
#SBATCH --error=logs/nnunet_fold0_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

echo "nnU-Net fold 0 training ($(date))"
echo ""

export PATH="$HOME/.local/bin:$PATH"
export nnUNet_raw="${REPO_ROOT}/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/nnunet/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/nnunet/nnUNet_results"
export NNUNET_COMPILE=0
export nnUNet_def_n_proc=2
export nnUNet_n_proc_DA=2
export OMP_NUM_THREADS=2

mkdir -p "${REPO_ROOT}/logs"

echo "Checking prerequisites..."

PLANS="$nnUNet_preprocessed/Dataset001_Mitochondria/nnUNetPlans.json"
SPLITS_RAW_JSON="$nnUNet_raw/Dataset001_Mitochondria/splits_final.json"
SPLITS_PREPROC_JSON="$nnUNet_preprocessed/Dataset001_Mitochondria/splits_final.json"
ID_MAP="${REPO_ROOT}/data/nnunet_id_map.json"
ACTIVE_SPLITS="${REPO_ROOT}/data/splits.json"

if [ ! -f "$PLANS" ]; then
    echo "ERROR: nnUNetPlans.json not found. Run nnunet_preprocessing/preprocess_nnunet.sh first."
    exit 1
fi

if [ ! -f "$SPLITS_RAW_JSON" ]; then
    echo "ERROR: splits_final.json not found in nnUNet_raw. Run data_preparation/create_nnunet_format.py first."
    exit 1
fi

if [ ! -f "$SPLITS_PREPROC_JSON" ]; then
    echo "ERROR: splits_final.json not found in nnUNet_preprocessed. Run extract_config.py first."
    exit 1
fi

if [ ! -f "$ID_MAP" ]; then
    echo "ERROR: nnunet_id_map.json not found at $ID_MAP"
    exit 1
fi

if [ ! -f "$ACTIVE_SPLITS" ]; then
    echo "ERROR: active splits.json not found at $ACTIVE_SPLITS"
    exit 1
fi

python - "$SPLITS_RAW_JSON" "$SPLITS_PREPROC_JSON" "$ID_MAP" "$ACTIVE_SPLITS" <<'PY'
import json
import sys

raw_json, preproc_json, id_map_json, active_splits_json = sys.argv[1:]

def fail(message: str) -> None:
    print(f"ERROR: {message}")
    sys.exit(1)

def load_fold0(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        splits = json.load(f)
    if not isinstance(splits, list) or not splits:
        fail(f"{path} does not contain a non-empty list of splits")
    fold0 = splits[0]
    if "train" not in fold0 or "val" not in fold0:
        fail(f"{path} fold 0 is missing train/val keys")
    return list(fold0["train"]), list(fold0["val"])

raw_train, raw_val = load_fold0(raw_json)
preproc_train, preproc_val = load_fold0(preproc_json)

if raw_train != preproc_train or raw_val != preproc_val:
    fail(
        "raw and preprocessed splits_final.json differ. "
        "Re-run nnunet_preprocessing/extract_config.py before training."
    )

with open(id_map_json) as f:
    id_map = json.load(f)
with open(active_splits_json) as f:
    active_splits = json.load(f)

expected_train = {
    (entry["sample"], entry["filename"])
    for entry in active_splits["train"]
}
expected_val = {
    (entry["sample"], entry["filename"])
    for entry in active_splits["val"]
}

def case_ids_to_pairs(case_ids: list[str], split_name: str) -> set[tuple[str, str]]:
    pairs = set()
    for case_id in case_ids:
        meta = id_map.get(case_id)
        if meta is None:
            fail(f"{case_id} is missing from nnunet_id_map.json")
        if meta.get("split") != split_name:
            fail(f"{case_id} is tagged as split={meta.get('split')} in nnunet_id_map.json, expected {split_name}")
        pairs.add((meta["sample"], meta["filename"]))
    return pairs

actual_train = case_ids_to_pairs(raw_train, "train")
actual_val = case_ids_to_pairs(raw_val, "val")

if actual_train != expected_train:
    fail(
        f"fold 0 train set mismatch: expected {len(expected_train)} cases, "
        f"got {len(actual_train)}"
    )

if actual_val != expected_val:
    fail(
        f"fold 0 val set mismatch: expected {len(expected_val)} cases, "
        f"got {len(actual_val)}"
    )

print(f"  ✓ splits_final.json validated")
print(f"    raw/preproc fold 0 train : {len(raw_train)}")
print(f"    raw/preproc fold 0 val   : {len(raw_val)}")
PY

echo "  ✓ nnUNetPlans.json found"
echo ""

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo "starting nnUNetv2_train (dataset 001, 2d, fold 0)"
echo ""

nnUNetv2_train 001 2d 0 --npz --c

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "training done: $(date)"

    FOLD_DIR="$nnUNet_results/Dataset001_Mitochondria/nnUNetTrainer__nnUNetPlans__2d/fold_0"
    CKPT="$FOLD_DIR/checkpoint_final.pth"
    if [ -f "$CKPT" ]; then
        SIZE=$(ls -lh "$CKPT" | awk '{print $5}')
        echo "checkpoint: $CKPT ($SIZE)"
    fi

    BEST_CKPT="$FOLD_DIR/checkpoint_best.pth"
    if [ -f "$BEST_CKPT" ]; then
        echo "best checkpoint: $BEST_CKPT"
    fi
else
    echo "training failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
