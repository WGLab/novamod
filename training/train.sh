#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH -c 20
#SBATCH --mem=16G
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -J Otrain --out=training-online_test8.out

module load CUDA

echo "Running on node: $(hostname)"
nvidia-smi

python -u training-online.py
