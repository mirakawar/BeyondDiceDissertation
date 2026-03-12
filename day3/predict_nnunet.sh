#!/bin/bash
#SBATCH --job-name=nnunet_predict
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/s2292949/dissertation_resplit/logs/nnunet_predict_%j.out
#SBATCH --error=/home/s2292949/dissertation_resplit/logs/nnunet_predict_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s2292949@ed.ac.uk

export PATH="$HOME/.local/bin:$PATH"
export nnUNet_raw="/home/s2292949/dissertation_resplit/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/home/s2292949/dissertation_resplit/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/s2292949/dissertation_resplit/nnunet/nnUNet_results"
export TQDM_DISABLE=1

mkdir -p /home/s2292949/dissertation_resplit/predictions/nnunet

echo "=================================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Started  : $(date)"
echo "=================================================="

nnUNetv2_predict \
    -i $nnUNet_raw/Dataset001_Mitochondria/imagesTs \
    -o /home/s2292949/dissertation_resplit/predictions/nnunet \
    -d 001 -c 2d -f 0 \
    --save_probabilities

echo "Prediction complete: $(date)"
echo "Output: /home/s2292949/dissertation_resplit/predictions/nnunet"