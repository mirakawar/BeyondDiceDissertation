#!/bin/bash
#SBATCH --job-name=nnunet_eval_pipe
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/s2292949/dissertation_resplit/logs/nnunet_eval_pipeline_%j.out
#SBATCH --error=/home/s2292949/dissertation_resplit/logs/nnunet_eval_pipeline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s2292949@ed.ac.uk

set -euo pipefail

echo "=================================================="
echo "nnU-Net Prediction + Evaluation"
echo "Job ID   : ${SLURM_JOB_ID:-interactive}"
echo "Node     : ${SLURM_NODELIST:-local}"
echo "Started  : $(date)"
echo "=================================================="

export PATH="$HOME/.local/bin:$PATH"

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo ""
echo "Python: $(which python3)"
echo ""

CONFIG=/home/s2292949/dissertation_resplit/config.json

echo "[1/2] nnU-Net test prediction"
OVERWRITE_EXISTING=1 bash /home/s2292949/dissertation_resplit/day3/predict_nnunet.sh

echo "[2/2] Strict full evaluation (nnU-Net)"
python3 /home/s2292949/dissertation_resplit/day5/evaluate.py \
  --model nnunet \
  --config "${CONFIG}" \
  --strict

echo ""
echo "Results written to:"
echo "  /home/s2292949/dissertation_resplit/evaluation/results"
echo ""
echo "=================================================="
echo "Pipeline complete : $(date)"
echo "=================================================="
