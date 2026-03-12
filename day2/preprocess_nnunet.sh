#!/bin/bash
#SBATCH --job-name=nnunet_preproc
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/s2292949/dissertation_resplit/logs/nnunet_preprocess_%j.out
#SBATCH --error=/home/s2292949/dissertation_resplit/logs/nnunet_preprocess_%j.err

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export nnUNet_raw="/home/s2292949/dissertation_resplit/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/s2292949/dissertation_resplit/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/s2292949/dissertation_resplit/nnunet/nnUNet_results"
export NNUNET_COMPILE=0

mkdir -p /home/s2292949/dissertation_resplit/logs
rm -rf "$nnUNet_preprocessed/Dataset001_Mitochondria"

echo "================================================================"
echo "  nnU-Net Preprocessing"
echo "  Job ID  : ${SLURM_JOB_ID:-interactive}"
echo "  Node    : ${SLURM_NODELIST:-local}"
echo "  Started : $(date)"
echo "  CPUs    : 2"
echo "================================================================"
echo ""

nnUNetv2_plan_and_preprocess -d 001 -npfp 2 -np 2 --verify_dataset_integrity --verbose --clean
python /home/s2292949/dissertation_resplit/day2/extract_config.py

echo ""
echo "================================================================"
echo "  Preprocessing complete : $(date)"
echo "================================================================"
