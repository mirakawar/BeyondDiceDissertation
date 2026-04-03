#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --exclude=saxa
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/s2292949/dissertation_resplit/logs/unet_train_%j.out
#SBATCH --error=/home/s2292949/dissertation_resplit/logs/unet_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s2292949@ed.ac.uk

set -e

echo "================================================================"
echo "  U-Net Training"
echo "  Job ID  : ${SLURM_JOB_ID:-interactive}"
echo "  Node    : ${SLURM_NODELIST:-local}"
echo "  Started : $(date)"
echo "================================================================"
echo ""

export PATH="$HOME/.local/bin:$PATH"

mkdir -p /home/s2292949/dissertation_resplit/logs

echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo "Python: $(which python)"
echo ""

python /home/s2292949/dissertation_resplit/day4/train_unet.py \
    --config /home/s2292949/dissertation_resplit/config.json \
    --batch-size 4 \
    --epochs 1000 \
    --learning-rate 1e-3 \
    --num-workers 8 \
    --augment \
    --loss-type bce_dice

EXIT_CODE=$?

echo ""
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ U-Net training complete : $(date)"
    echo "  Checkpoint : $(cat /home/s2292949/dissertation_resplit/config.json | python3 -c \
        'import json,sys; c=json.load(sys.stdin); print(c["paths"]["unet_checkpoint"])')/best_model.pt"
    echo "  Log        : $(cat /home/s2292949/dissertation_resplit/config.json | python3 -c \
        'import json,sys; c=json.load(sys.stdin); print(c["paths"]["unet_training_log"])')"
else
    echo "  ✗ U-Net training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "================================================================"
