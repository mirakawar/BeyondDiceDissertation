#!/bin/bash
#SBATCH --job-name=nnunet_eval_pipe
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nnunet_eval_pipeline_%j.out
#SBATCH --error=logs/nnunet_eval_pipeline_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

echo "nnU-Net prediction + evaluation ($(date))"

export PATH="$HOME/.local/bin:$PATH"

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo ""
echo "Python: $(which python3)"
echo ""

CONFIG="${REPO_ROOT}/config.json"

echo "[1/2] nnU-Net test prediction"
OVERWRITE_EXISTING=1 bash "${REPO_ROOT}/nnunet_pipeline/predict_nnunet.sh"

echo "[2/2] Strict full evaluation (nnU-Net)"
python3 "${REPO_ROOT}/evaluation_pipeline/evaluate.py" \
  --model nnunet \
  --config "${CONFIG}" \
  --strict

echo "results in ${REPO_ROOT}/evaluation/results"
echo "done: $(date)"
