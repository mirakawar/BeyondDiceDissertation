#!/bin/bash
#SBATCH --job-name=unet_b2nd
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/s2292949/dissertation_resplit/logs/unet_b2nd_%j.out
#SBATCH --error=/home/s2292949/dissertation_resplit/logs/unet_b2nd_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s2292949@ed.ac.uk

set -e

echo "=================================================="
echo "Job ID   : ${SLURM_JOB_ID:-interactive}"
echo "Node     : ${SLURM_NODELIST:-local}"
echo "Started  : $(date)"
echo "=================================================="

export PATH="$HOME/.local/bin:$PATH"
mkdir -p /home/s2292949/dissertation_resplit/logs

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "Python: $(which python)"
echo ""

python -u /home/s2292949/dissertation_resplit/day4/train_unet.py \
    --config /home/s2292949/dissertation_resplit/config.json \
    --batch-size 4 \
    --epochs 1000 \
    --learning-rate 1e-3 \
    --num-workers 2 \
    --augment \
    --loss-type ce_dice \
    --sample-output-dir /home/s2292949/dissertation_resplit/samples/unet_val \
    --loss-plot-path /home/s2292949/dissertation_resplit/logs/unet_b2nd_loss.png

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Training complete : $(date)"
else
    echo "  Training FAILED with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "=================================================="
