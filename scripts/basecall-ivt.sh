#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -p gpuq
#SBATCH --gres=gpu:1
#SBATCH -J IVT

genome="20240411_HEK293T_IVT"
#genome="20240321_directRNA_HEK293T"

echo "Basecalling: ${genome}"
in="/home/zouy1/projects/RNAmod/datasets/rna004/${genome}"
out="/home/zouy1/projects/RNAmod/VAE/data/RNA/${genome}/"
mkdir -p $out

DORADO="/mnt/isilon/wang_lab/shared/apps/dorado/dorado-1.0.2-linux-x64"

$DORADO/bin/dorado basecaller $DORADO/models/rna004_130bps_sup@v5.2.0 $in --recursive --emit-moves --device cuda:0  > $out/$genome.basecall.dorado.bam
