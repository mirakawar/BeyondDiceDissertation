#!/bin/bash
#SBATCH --job-name=unet_eval_pipe
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/s2292949/dissertation_resplit/logs/unet_eval_pipeline_%j.out
#SBATCH --error=/home/s2292949/dissertation_resplit/logs/unet_eval_pipeline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s2292949@ed.ac.uk

set -euo pipefail

echo "=================================================="
echo "U-Net Prediction + Evaluation + Corner Cases"
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

echo "[1/4] U-Net test prediction"
python3 /home/s2292949/dissertation_resplit/day4/predict_unet.py \
  --config "${CONFIG}" \
  --split test \
  --overwrite \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda \
  --no-compress-probs

echo "[2/4] Strict full evaluation (U-Net)"
python3 /home/s2292949/dissertation_resplit/day5/evaluate.py \
  --model unet \
  --config "${CONFIG}" \
  --strict

echo "[3/4] Corner-case analysis"
python3 /home/s2292949/dissertation_resplit/day5/corner_case_analysis.py \
  --config "${CONFIG}" \
  --models unet nnunet \
  --dice-threshold 0.85 \
  --quantile 0.75

echo "[4/4] Corner-case/scatter/threshold figures"
python3 /home/s2292949/dissertation_resplit/figures/visualise.py \
  --config "${CONFIG}" \
  --mode corner_cases scatter threshold_curves \
  --model both \
  --n 6

echo ""
echo "=================================================="
echo "Pipeline complete : $(date)"
echo "=================================================="
