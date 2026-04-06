#!/bin/bash
#SBATCH --job-name=nnunet_predict
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/nnunet_predict_%j.out
#SBATCH --error=logs/nnunet_predict_%j.err
#SBATCH --exclude=damnii12

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export PATH="$HOME/.local/bin:$PATH"
export nnUNet_raw="${REPO_ROOT}/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/nnunet/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/nnunet/nnUNet_results"
export TQDM_DISABLE=1

DATASET="Dataset001_Mitochondria"
DATASET_ID="001"
CONFIG="2d"
FOLD="0"
CHECKPOINT_NAME="checkpoint_best.pth"

INPUT_DIR="${nnUNet_raw}/${DATASET}/imagesTs"
INSTANCES_DIR="${nnUNet_raw}/${DATASET}/instancesTs"
FOLD_DIR="${nnUNet_results}/${DATASET}/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}"
CHECKPOINT_PATH="${FOLD_DIR}/${CHECKPOINT_NAME}"

PRED_ROOT="${REPO_ROOT}/predictions/nnunet"
RAW_OUT="${PRED_ROOT}/_raw_predict"
BINARY_OUT="${PRED_ROOT}/binary"
PROBS_OUT="${PRED_ROOT}/probs"
META_OUT="${PRED_ROOT}/meta"

OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"

fail() {
    echo "ERROR: $1" >&2
    exit 1
}

[[ -d "${INPUT_DIR}" ]] || fail "Input directory not found: ${INPUT_DIR}"
[[ -d "${INSTANCES_DIR}" ]] || fail "Ground-truth instances directory not found: ${INSTANCES_DIR}"
[[ -d "${FOLD_DIR}" ]] || fail "Fold output directory not found: ${FOLD_DIR}"
[[ -f "${CHECKPOINT_PATH}" ]] || fail "Checkpoint not found: ${CHECKPOINT_PATH}"

mkdir -p "${PRED_ROOT}" "${RAW_OUT}" "${BINARY_OUT}" "${PROBS_OUT}" "${META_OUT}"

existing_files=$(find "${PRED_ROOT}" -maxdepth 2 -type f | wc -l)
if [[ "${existing_files}" -gt 0 ]]; then
    if [[ "${OVERWRITE_EXISTING}" != "1" ]]; then
        fail "Prediction directory ${PRED_ROOT} already contains files. Re-run with OVERWRITE_EXISTING=1 to replace them."
    fi
    find "${PRED_ROOT}" -maxdepth 2 -type f -delete
fi

num_inputs=$(find "${INPUT_DIR}" -maxdepth 1 -name '*.nii.gz' | wc -l)
num_instances=$(find "${INSTANCES_DIR}" -maxdepth 1 -name '*_inst.nii.gz' | wc -l)

[[ "${num_inputs}" -gt 0 ]] || fail "No test images found in ${INPUT_DIR}"
[[ "${num_instances}" -gt 0 ]] || fail "No test instance masks found in ${INSTANCES_DIR}"
[[ "${num_inputs}" -eq "${num_instances}" ]] || fail "imagesTs count (${num_inputs}) does not match instancesTs count (${num_instances})"

echo "nnU-Net test prediction ($(date))"
echo "input: ${INPUT_DIR}  checkpoint: ${CHECKPOINT_NAME}  cases: ${num_inputs}"
echo ""
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo ""

nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${RAW_OUT}" \
    -d "${DATASET_ID}" -c "${CONFIG}" -f "${FOLD}" \
    -chk "${CHECKPOINT_NAME}" \
    --save_probabilities

for f in "${RAW_OUT}"/*.nii.gz; do
    mv "${f}" "${BINARY_OUT}/"
done

for f in "${RAW_OUT}"/*.npz; do
    mv "${f}" "${PROBS_OUT}/"
done

for f in "${RAW_OUT}"/*; do
    [[ -e "${f}" ]] || continue
    mv "${f}" "${META_OUT}/"
done

rmdir "${RAW_OUT}" 2>/dev/null || true

num_binary=$(find "${BINARY_OUT}" -maxdepth 1 -name '*.nii.gz' | wc -l)
num_probs=$(find "${PROBS_OUT}" -maxdepth 1 -name '*.npz' | wc -l)

[[ "${num_binary}" -eq "${num_inputs}" ]] || fail "Expected ${num_inputs} binary predictions, found ${num_binary}"
[[ "${num_probs}" -eq "${num_inputs}" ]] || fail "Expected ${num_inputs} probability files, found ${num_probs}"

echo ""
echo "Prediction complete: $(date)"
echo "Binary output : ${BINARY_OUT}"
echo "Prob output   : ${PROBS_OUT}"
echo "Binary files  : ${num_binary}"
echo "Prob files    : ${num_probs}"
