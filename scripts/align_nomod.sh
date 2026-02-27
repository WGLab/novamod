#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -J rna.aln

genome=20240321_directRNA_HEK293T
dir=/home/zouy1/projects/RNAmod/VAE/data/RNA/

samtools fastq $dir/$genome/$genome.basecall.dorado.bam -T "*"| minimap2 -ax splice --junc-bed /mnt/isilon/wang_lab/umair/projects/LR_cDNA/gencode.v46.basic.junc.bed -uf -k14 -t 20 /mnt/isilon/wang_lab/umair/data/GRCh38_noalts.fa - -y|samtools sort -O BAM -o $dir/$genome/$genome.GRCh38.splice.dorado.bam -@ 4 --write-index -m 4G

samtools view $dir/$genome/$genome.GRCh38.splice.dorado.bam -x "MM,ML" -O BAM -o $dir/$genome/$genome.aligned.GRCh38.splice.dorado.bam --write-index
