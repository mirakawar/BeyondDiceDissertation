#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --exclude=saxa
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/unet_train_%j.out
#SBATCH --error=logs/unet_train_%j.err

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${REPO_ROOT}/config.json"

echo "U-Net training ($(date))"
echo ""

export PATH="$HOME/.local/bin:$PATH"

mkdir -p "${REPO_ROOT}/logs"

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo "Python: $(which python)"
echo ""

python "${REPO_ROOT}/unet_pipeline/train_unet.py" \
    --config "${CONFIG}" \
    --batch-size 4 \
    --epochs 1000 \
    --learning-rate 1e-3 \
    --num-workers 8 \
    --augment \
    --loss-type bce_dice

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "done: $(date)"
else
    echo "training failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
