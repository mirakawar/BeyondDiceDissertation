#!/bin/bash
#SBATCH --job-name=unet_b2nd
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/unet_b2nd_%j.out
#SBATCH --error=logs/unet_b2nd_%j.err

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONFIG="${REPO_ROOT}/config.json"

echo "U-Net training ($(date))"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "${REPO_ROOT}/logs"

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "Python: $(which python)"
echo ""

python -u "${REPO_ROOT}/unet_pipeline/train_unet.py" \
    --config "${CONFIG}" \
    --batch-size 4 \
    --epochs 1000 \
    --learning-rate 1e-3 \
    --num-workers 2 \
    --augment \
    --loss-type ce_dice \
    --sample-output-dir "${REPO_ROOT}/samples/unet_val" \
    --loss-plot-path "${REPO_ROOT}/logs/unet_b2nd_loss.png"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "done: $(date)"
else
    echo "training failed (exit code $EXIT_CODE)"
    exit $EXIT_CODE
fi
