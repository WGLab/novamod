#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -J rna.gft
#SBATCH --output=convert_%A_%a.out
#SBATCH --error=convert_%A_%a.out

#genome=20240411_HEK293T_IVT
genome=20240321_directRNA_HEK293T
dir=/home/zouy1/projects/RNAmod/VAE/data/RNA/$genome/
pod5=/home/zouy1/projects/RNAmod/datasets/rna004/$genome/

python /mnt/isilon/wang_lab/zac/software/deepmod2-training/full_signal_model/pod5_to_bam.py --bam $dir/$genome.aligned.GRCh38.splice.dorado.bam --input $pod5 --file_type pod5 --output $dir --prefix converted --threads 12


ls -1 $dir/converted*bam > $dir/bam_files

samtools cat -b $dir/bam_files -o -|samtools sort -m 8G -O BAM -o $dir/$genome.final.bam --write-index --threads 12
