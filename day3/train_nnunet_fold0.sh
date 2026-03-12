#!/bin/bash
#SBATCH --job-name=nnunet_fold0
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/s2292949/dissertation_resplit/logs/nnunet_fold0_%j.out
#SBATCH --error=/home/s2292949/dissertation_resplit/logs/nnunet_fold0_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s2292949@ed.ac.uk
# =============================================================================
# Day 3 — Train nnU-Net (fold 0 only)
# =============================================================================
# Uses the custom splits_final.pkl created in Day 1 which locks fold 0 to
# our stratified train/val split. This ensures both models see identical data.
#
# Usage:
#   bash train_nnunet_fold0.sh          # run interactively
#   sbatch train_nnunet_fold0.sh        # submit to SLURM
#
# The --npz flag tells nnU-Net to save softmax probability maps alongside
# binary predictions — required for ECE computation at evaluation time.
# =============================================================================

set -e

echo "================================================================"
echo "  nnU-Net Fold 0 Training"
echo "  Job ID  : ${SLURM_JOB_ID:-interactive}"
echo "  Node    : ${SLURM_NODELIST:-local}"
echo "  Started : $(date)"
echo "================================================================"
echo ""

# ── Environment ──────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"
export nnUNet_raw="/home/s2292949/dissertation_resplit/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/s2292949/dissertation_resplit/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/s2292949/dissertation_resplit/nnunet/nnUNet_results"
export NNUNET_COMPILE=0
export nnUNet_def_n_proc=2
export nnUNet_n_proc_DA=2
export OMP_NUM_THREADS=2

mkdir -p /home/s2292949/dissertation_resplit/logs

# ── Verify prerequisites ─────────────────────────────────────────────────────
echo "Checking prerequisites..."

PLANS="$nnUNet_preprocessed/Dataset001_Mitochondria/nnUNetPlans.json"
SPLITS="$nnUNet_preprocessed/Dataset001_Mitochondria/splits_final.pkl"

if [ ! -f "$PLANS" ]; then
    echo "ERROR: nnUNetPlans.json not found. Run Day 2 first."
    exit 1
fi

if [ ! -f "$SPLITS" ]; then
    echo "ERROR: splits_final.pkl not found in preprocessed folder. Run extract_config.py first."
    exit 1
fi

echo "  ✓ nnUNetPlans.json found"
echo "  ✓ splits_final.pkl found"
echo ""

# ── Check GPU ─────────────────────────────────────────────────────────────────
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Train fold 0 ──────────────────────────────────────────────────────────────
# Flags:
#   2d      — use the 2D configuration (auto-selected during preprocessing)
#   0       — fold 0 only (our standardised split)
#   --npz   — save softmax .npz files alongside binary predictions
#             needed for ECE metric at evaluation time
#   --c     — continue training if a checkpoint already exists
#
# nnU-Net will:
#   - read splits_final.pkl to get train/val assignment for fold 0
#   - apply its default augmentation pipeline
#   - train for up to 1000 epochs with its own early stopping logic
#   - save checkpoint_best.pth and checkpoint_final.pth

echo "================================================================"
echo "  Starting nnUNetv2_train"
echo "  Dataset  : 001 (Dataset001_Mitochondria)"
echo "  Config   : 2d"
echo "  Fold     : 0"
echo "================================================================"
echo ""

nnUNetv2_train 001 2d 0 --npz --c

EXIT_CODE=$?

echo ""
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ Training complete : $(date)"

    # Show checkpoint
    FOLD_DIR="$nnUNet_results/Dataset001_Mitochondria/nnUNetTrainer__nnUNetPlans__2d/fold_0"
    CKPT="$FOLD_DIR/checkpoint_final.pth"
    if [ -f "$CKPT" ]; then
        SIZE=$(ls -lh "$CKPT" | awk '{print $5}')
        echo "  ✓ Final checkpoint : $CKPT ($SIZE)"
    fi

    BEST_CKPT="$FOLD_DIR/checkpoint_best.pth"
    if [ -f "$BEST_CKPT" ]; then
        echo "  ✓ Best checkpoint  : $BEST_CKPT"
    fi

    echo ""
    echo "  Next step: run predict_nnunet_fold0.sh to generate test predictions"
else
    echo "  ✗ Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "================================================================"
