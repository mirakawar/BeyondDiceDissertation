#!/bin/bash
#SBATCH --job-name=nnunet_preproc
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/nnunet_preprocess_%j.out
#SBATCH --error=logs/nnunet_preprocess_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export PATH="$HOME/.local/bin:$PATH"
export nnUNet_raw="${REPO_ROOT}/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/nnunet/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/nnunet/nnUNet_results"
export NNUNET_COMPILE=0

mkdir -p "${REPO_ROOT}/logs"
rm -rf "$nnUNet_preprocessed/Dataset001_Mitochondria"

echo "nnU-Net preprocessing ($(date))"
echo ""

nnUNetv2_plan_and_preprocess -d 001 -npfp 2 -np 2 --verify_dataset_integrity --verbose --clean
python "${REPO_ROOT}/nnunet_preprocessing/extract_config.py"

echo "done: $(date)"
