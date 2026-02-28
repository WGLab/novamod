#!/bin/bash

#SBATCH --time=3-00:00:00
#SBATCH -c 20
#SBATCH --mem=16G
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -J otrain --out=logs/%x.out

CONFIG_PATH=${1:-configs/train_online.example.json}

echo "Running training with config: $CONFIG_PATH"

module load CUDA

echo "Running on node: $(hostname)"
nvidia-smi

python -u train.py --config "$CONFIG_PATH"
