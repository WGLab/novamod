#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH -c 20
#SBATCH --mem=16G
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -J Otrain --out=logs/%x.out

TAG=$1

echo "Running training with tag: $TAG"

module load CUDA

echo "Running on node: $(hostname)"
nvidia-smi

python -u training-online.py
