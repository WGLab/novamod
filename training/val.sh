#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH -c 10
#SBATCH --mem=16G
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -J Oval --out=logs/%x.out

CONFIG_PATH=${1:-configs/val.example.json}

echo "Running validation with config: $CONFIG_PATH"

module load CUDA

echo "Running on node: $(hostname)"
nvidia-smi

python -u val.py --config "$CONFIG_PATH"
