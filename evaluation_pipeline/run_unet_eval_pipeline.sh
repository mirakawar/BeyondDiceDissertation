#!/bin/bash
#SBATCH --job-name=unet_eval_pipe
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/unet_eval_pipeline_%j.out
#SBATCH --error=logs/unet_eval_pipeline_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

echo "U-Net prediction + evaluation + corner cases ($(date))"

export PATH="$HOME/.local/bin:$PATH"

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo ""
echo "Python: $(which python3)"
echo ""

CONFIG="${REPO_ROOT}/config.json"

echo "[1/4] U-Net test prediction"
python3 "${REPO_ROOT}/unet_pipeline/predict_unet.py" \
  --config "${CONFIG}" \
  --split test \
  --overwrite \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda \
  --no-compress-probs

echo "[2/4] Strict full evaluation (U-Net)"
python3 "${REPO_ROOT}/evaluation_pipeline/evaluate.py" \
  --model unet \
  --config "${CONFIG}" \
  --strict

echo "[3/4] Corner-case analysis"
python3 "${REPO_ROOT}/evaluation_pipeline/corner_case_analysis.py" \
  --config "${CONFIG}" \
  --models unet nnunet \
  --dice-threshold 0.85

echo "[4/4] Corner-case/scatter/threshold figures"
python3 "${REPO_ROOT}/figures/visualise.py" \
  --config "${CONFIG}" \
  --mode corner_cases scatter threshold_curves \
  --model both \
  --n 6

echo "done: $(date)"
