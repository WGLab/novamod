#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -J md.ext

sample=20240321_directRNA_HEK293T
dir=/home/zouy1/projects/RNAmod/VAE/data/RNA/${sample}
ref=/home/zouy1/software/reference/GRCh38.primary_assembly.genome.fa
bam=${dir}

/home/zouy1/software/modkit/modkit extract full ${dir}/${sample}.aligned.GRCh38.splice.dorado-mm.bam  ${dir}/${sample}.modkit.extract.chr20.bed --region chr20 --ref $ref -t 12 --force --edge-filter 3,3
