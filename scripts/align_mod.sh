#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -J getmod

genome=20240321_directRNA_HEK293T
dir=/home/zouy1/projects/RNAmod/VAE/data/RNA/

samtools view $dir/$genome/$genome.GRCh38.splice.dorado.bam -O BAM -o $dir/$genome/$genome.aligned.GRCh38.splice.dorado-mm.bam --write-index --threads 12
