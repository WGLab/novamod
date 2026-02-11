#!/bin/bash

#SBATCH --time=06:00:00
#SBATCH -c 20
#SBATCH --mem=16G
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -J Ovalid --out=validation-online_test3.out

module load CUDA

echo "Running on node: $(hostname)"
nvidia-smi

python -u validation-online_evalOnly.py
